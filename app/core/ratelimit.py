"""Per-API-key rate limiting.

Fixed window + burst, implemented as a token bucket so a caller can spend ``burst``
requests instantly and then be paced at ``requests_per_minute``. The in-memory
limiter is the default (correct for a single container); the Redis backend shares the
same bucket semantics across workers via ``EVAL`` on a small script, so horizontal
scaling does not silently multiply the limit.

Chosen deliberately over a sliding-log counter: a bucket costs O(1) per key, which
matters when ``/moderate`` is the hot path.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class RateDecision:
    allowed: bool
    limit: int
    remaining: int
    retry_after_s: float = 0.0
    reset_at: float = 0.0

    def headers(self) -> dict[str, str]:
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(math.ceil(self.reset_at))),
            **({"Retry-After": str(max(1, math.ceil(self.retry_after_s)))} if not self.allowed else {}),
        }


class RateLimiter(Protocol):
    async def acquire(self, key: str, *, limit_per_minute: int, burst: int) -> RateDecision: ...

    async def reset(self, key: str | None = None) -> None: ...


class MemoryRateLimiter:
    """Token bucket per key, pruned on read. Single-process, no deps."""

    def __init__(self, *, max_keys: int = 10_000) -> None:
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_refill)
        self._max_keys = max_keys
        self._lock_seq = 0

    async def acquire(self, key: str, *, limit_per_minute: int, burst: int) -> RateDecision:
        now = time.monotonic()
        capacity = max(1, max(limit_per_minute, burst))
        refill_per_s = max(1e-6, limit_per_minute / 60.0)
        tokens, last = self._buckets.get(key, (float(capacity), now))
        tokens = min(capacity, tokens + (now - last) * refill_per_s)
        if tokens >= 1.0:
            tokens -= 1.0
            self._buckets[key] = (tokens, now)
            self._prune(now)
            return RateDecision(allowed=True, limit=capacity, remaining=int(tokens), reset_at=now + (capacity - tokens) / refill_per_s)
        self._buckets[key] = (tokens, now)
        wait = (1.0 - tokens) / refill_per_s
        return RateDecision(allowed=False, limit=capacity, remaining=0, retry_after_s=wait, reset_at=now + wait)

    def _prune(self, now: float) -> None:
        self._lock_seq += 1
        if len(self._buckets) <= self._max_keys and self._lock_seq % 512:
            return
        stale = [k for k, (_tokens, last) in self._buckets.items() if now - last > 900]
        for key in stale:
            self._buckets.pop(key, None)
        if len(self._buckets) > self._max_keys:  # hard cap safety valve
            for key in sorted(self._buckets, key=lambda k: self._buckets[k][1])[: len(self._buckets) - self._max_keys]:
                self._buckets.pop(key, None)

    async def reset(self, key: str | None = None) -> None:
        if key is None:
            self._buckets.clear()
        else:
            self._buckets.pop(key, None)

    def snapshot(self) -> dict[str, Any]:
        return {"backend": "memory", "tracked_keys": len(self._buckets)}


class RedisRateLimiter:
    """Shared bucket across workers. Requires ``redis>=5`` (optional extra).

    The script is a single atomic read-modify-write: no race between checking and
    decrementing, which a ``GET``/``SET`` pair would have.
    """

    _SCRIPT = """
    local key = KEYS[1]
    local now = tonumber(ARGV[1])
    local capacity = tonumber(ARGV[2])
    local refill = tonumber(ARGV[3])
    local state = redis.call('HMGET', key, 'tokens', 'ts')
    local tokens = tonumber(state[1])
    local last = tonumber(state[2])
    if tokens == nil or last == nil then tokens = capacity last = now end
    tokens = math.min(capacity, tokens + (now - last) * refill)
    local allowed = 1
    if tokens >= 1 then tokens = tokens - 1 else allowed = 0 end
    redis.call('HSET', key, 'tokens', tokens, 'ts', now)
    redis.call('EXPIRE', key, math.ceil(capacity / refill) + 60)
    return {allowed, tostring(tokens)}
    """

    def __init__(self, url: str) -> None:
        import redis.asyncio as aioredis  # imported here so the extra stays optional

        self._client = aioredis.from_url(url, decode_responses=True)
        self._sha: str | None = None

    async def acquire(self, key: str, *, limit_per_minute: int, burst: int) -> RateDecision:
        capacity = max(1, max(limit_per_minute, burst))
        refill = max(1e-6, limit_per_minute / 60.0)
        now = time.time()
        try:
            allowed, tokens = await self._invoke(key, capacity, refill, now)
        except Exception:  # pragma: no cover - redis outage must not block traffic
            limiter = get_limiter(backend="memory")
            return await limiter.acquire(key, limit_per_minute=limit_per_minute, burst=burst)
        remaining = int(float(tokens))
        if allowed:
            return RateDecision(allowed=True, limit=capacity, remaining=remaining, reset_at=now + (capacity - float(tokens)) / refill)
        wait = max(0.0, (1.0 - float(tokens)) / refill)
        return RateDecision(allowed=False, limit=capacity, remaining=0, retry_after_s=wait, reset_at=now + wait)

    async def _invoke(self, key: str, capacity: int, refill: float, now: float) -> tuple[int, str]:
        args = (f"rl:{key}", f"{now:.6f}", str(capacity), f"{refill:.9f}")
        if self._sha is None:
            self._sha = await self._client.script_load(self._SCRIPT)
        try:
            raw = await self._client.evalsha(self._sha, 1, *args)
        except Exception as exc:  # NOSCRIPT after a redis restart -> reload once
            if "NOSCRIPT" not in str(exc).upper():
                raise
            self._sha = await self._client.script_load(self._SCRIPT)
            raw = await self._client.evalsha(self._sha, 1, *args)
        return int(raw[0]), str(raw[1])

    async def reset(self, key: str | None = None) -> None:  # pragma: no cover - admin path
        if key is None:
            async for found in self._client.scan_iter(match="rl:*", count=500):
                await self._client.delete(found)
        else:
            await self._client.delete(f"rl:{key}")

    def snapshot(self) -> dict[str, Any]:  # pragma: no cover - debug helper
        return {"backend": "redis"}

    async def aclose(self) -> None:  # pragma: no cover - shutdown path
        await self._client.aclose()


_LIMITER: Any = None
_LIMITER_BACKEND: str | None = None


def get_limiter(backend: str | None = None, *, redis_url: str = "", limit_default: int = 60) -> Any:
    """Process-wide limiter; switches backend only when the config changes."""
    global _LIMITER, _LIMITER_BACKEND
    if backend is None:
        from app.core.config import get_settings

        settings = get_settings()
        backend = settings.rate_limit_backend
        url = settings.redis_url
    else:
        url = redis_url
    if _LIMITER is not None and _LIMITER_BACKEND == backend:
        return _LIMITER
    if backend == "redis" and url:
        try:
            _LIMITER = RedisRateLimiter(url)
            _LIMITER_BACKEND = "redis"
            return _LIMITER
        except Exception as exc:  # pragma: no cover - falls back rather than failing boot
            from app.core.logging import get_logger

            get_logger(__name__).warning("redis rate limiter unavailable, using in-memory", extra={"error": str(exc)[:160]})
    _LIMITER = MemoryRateLimiter()
    _LIMITER_BACKEND = "memory"
    return _LIMITER


def reset_limiter() -> None:  # pragma: no cover - test helper
    global _LIMITER, _LIMITER_BACKEND
    _LIMITER = None
    _LIMITER_BACKEND = None
