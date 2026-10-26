"""API-key authentication + per-key rate limiting (FastAPI dependency).

Keys look like ``gmk_<32 url-safe chars>`` and are stored only as
``sha256(pepper || key)`` - see :mod:`app.core.security`. Verification therefore does
one indexed digest lookup, and a short in-process cache (30s TTL) keeps that off the
hot path while still letting a revoke take effect quickly.

Scope model:

``moderate``    POST /moderate, POST /evaluate
``read_logs``   GET /logs, /logs/{id}, /explain/{id}, /stats, /logs/export.csv
``admin``       POST/DELETE /keys, POST /admin/retention, POST /admin/reload-policies

Auth failure modes are deliberately distinct (401 unknown/invalid, 403 inactive or
scope-less, 429 limited) so an operator can tell a misconfigured client from a revoked
credential, and each increments its own Prometheus counter.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security.api_key import APIKeyHeader, APIKeyQuery

from app.core.config import get_settings
from app.core.logging import bind_api_key, get_logger
from app.core.ratelimit import RateDecision, get_limiter
from app.core.security import hash_api_key, key_id_of
from app.monitoring.metrics import get_metrics

log = get_logger(__name__)

SCOPES = ("moderate", "read_logs", "admin")
_CACHE_TTL_S = 30.0
_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}


@dataclass(slots=True)
class AuthContext:
    key_id: str
    name: str
    scopes: list[str] = field(default_factory=list)
    rate_limit_rpm: int | None = None
    label_hint: str = ""

    def has(self, scope: str) -> bool:
        return "admin" in self.scopes or scope in self.scopes

    def as_dict(self) -> dict[str, Any]:
        return {"key_id": self.key_id, "name": self.name, "scopes": self.scopes}


header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
query_scheme = APIKeyQuery(name="api_key", auto_error=False)


_MISS: Any = object()


def _cache_get(digest: str) -> Any:
    """Cached record, cached ``None`` (a known-bad key), or ``_MISS``."""
    entry = _cache.get(digest)
    if entry is None:
        return _MISS
    expires, record = entry
    if expires < time.monotonic():
        _cache.pop(digest, None)
        return _MISS
    return record


def _cache_put(digest: str, record: dict[str, Any] | None) -> None:
    if len(_cache) > 4096:  # bounded: a spray of bad keys must not grow memory
        for stale in [k for k, (exp, _r) in _cache.items() if exp < time.monotonic()]:
            _cache.pop(stale, None)
        if len(_cache) > 4096:
            _cache.clear()
    _cache[digest] = (time.monotonic() + _CACHE_TTL_S, record)


def clear_auth_cache() -> None:
    _cache.clear()


async def _lookup(digest: str) -> dict[str, Any] | None:
    """Digest -> key record (``None`` = definitely not a valid key).

    Negative results are cached too: an attacker spraying random keys should cost the
    database nothing after the first attempt per key.
    """
    cached = _cache_get(digest)
    if cached is not _MISS:
        return cached if isinstance(cached, dict) else None
    from app.db.repository import find_api_key_by_hash

    try:
        record = await find_api_key_by_hash(digest)
    except Exception as exc:
        # Fail closed: if we cannot verify the caller we cannot authorise them either.
        log.error("api key lookup failed; rejecting request", extra={"error": str(exc)[:200]})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "auth_backend_unavailable", "message": "Authentication backend is unavailable; request rejected."},
        ) from exc
    _cache_put(digest, record)
    return record


async def _enforce_rate_limit(request: Request, identity: str, limit_override: int | None) -> None:
    settings = get_settings()
    if not settings.rate_limit_enabled:
        return
    limit = limit_override or settings.rate_limit_requests_per_minute
    decision: RateDecision = await get_limiter().acquire(
        identity, limit_per_minute=limit, burst=max(1, settings.rate_limit_burst)
    )
    request.state.rate_limit = decision
    for header, value in decision.headers().items():
        request.state.rate_headers = getattr(request.state, "rate_headers", {}) | {header: value}
    if not decision.allowed:
        get_metrics().rate_limited.inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "rate_limited",
                "message": f"Rate limit exceeded ({limit}/minute for this key). Retry in {decision.retry_after_s:.1f}s.",
                "retry_after_s": round(decision.retry_after_s, 2),
                "hints": ["Batch requests, or request a higher per-key limit from the operator."],
            },
            headers=decision.headers(),
        )


async def authenticate(
    request: Request,
    header_key: str | None = Depends(header_scheme),
    query_key: str | None = Depends(query_scheme),
) -> AuthContext:
    """Resolve + validate the caller. Raises 401/403/429/503 as appropriate."""
    settings = get_settings()
    if query_key and not settings.allow_query_string_api_key:
        # Rejected on purpose: URLs land in proxy logs, browser history and referrers.
        get_metrics().auth_failures.labels(reason="query_string").inc()
        raise _unauthenticated("query_string_keys_disabled", "Query-string API keys are disabled; send the X-API-Key header.")
    raw_key = header_key

    # Unauthenticated mode exists for local demos and automated tests only.
    if not settings.auth_enabled:
        context = AuthContext(key_id="anonymous", name="no-auth mode", scopes=list(SCOPES), rate_limit_rpm=None)
        bind_api_key(context.key_id)
        request.state.auth = context
        await _enforce_rate_limit(request, f"ip:{_client_ip(request)}", None)
        return context

    if not raw_key:
        get_metrics().auth_failures.labels(reason="missing").inc()
        raise _unauthenticated("missing_api_key", f"Send your key in the '{settings.api_key_header}' header.")
    if len(raw_key) > 256 or "\n" in raw_key or "\r" in raw_key:
        get_metrics().auth_failures.labels(reason="malformed").inc()
        raise _unauthenticated("malformed_api_key", "The key is malformed.")

    record = await _lookup(hash_api_key(raw_key))
    if record is None:
        get_metrics().auth_failures.labels(reason="unknown").inc()
        log.warning("rejected unknown api key", extra={"key_id_hint": key_id_of(raw_key)})
        raise _unauthenticated("unknown_api_key", "This key is not recognised.")
    if not record.get("is_active", False):
        get_metrics().auth_failures.labels(reason="revoked").inc()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "key_revoked", "message": "This API key has been revoked.", "hints": ["Create a new key or ask the operator to re-activate it."]},
        )

    context = AuthContext(
        key_id=str(record["key_id"]),
        name=str(record.get("name") or ""),
        scopes=list(record.get("scopes") or ["moderate"]),
        rate_limit_rpm=record.get("rate_limit_rpm"),
    )
    bind_api_key(context.key_id)
    request.state.auth = context
    await _enforce_rate_limit(request, f"key:{context.key_id}", context.rate_limit_rpm)
    # Usage bookkeeping is fire-and-forget: it must never delay or fail the response.
    request.state.touch_key = context.key_id
    return context


def _unauthenticated(code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"code": code, "message": message, "hints": [f"Set the {get_settings().api_key_header} header to a key issued by the operator."]},
        headers={"WWW-Authenticate": 'ApiKey realm="moderation-api"'},
    )


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def require_scope(scope: str) -> Callable[..., Awaitable[AuthContext]]:
    """``Depends(require_scope("read_logs"))`` - scope check after authentication."""

    async def _dependency(auth: AuthContext = Depends(authenticate)) -> AuthContext:
        if not auth.has(scope):
            get_metrics().auth_failures.labels(reason="scope").inc()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "insufficient_scope",
                    "message": f"This endpoint requires the '{scope}' scope; your key has {', '.join(auth.scopes) or 'none'}.",
                },
            )
        return auth

    return _dependency


async def bootstrap_keys_from_settings() -> int:
    """Idempotently create the keys listed in ``GUARD_BOOTSTRAP_API_KEYS``."""
    settings = get_settings()
    raw = settings.bootstrap_api_keys.get_secret_value()
    keys = [item.strip() for item in raw.replace(" ", "").split(",") if item.strip()]
    if not keys:
        return 0
    from app.core.security import key_id_of as _kid
    from app.db.repository import upsert_api_key

    created = 0
    for index, key in enumerate(keys):
        result = await upsert_api_key(
            key_id=_kid(key),
            key_hash=hash_api_key(key),
            label_hint=key[:9] + "…",
            name=f"bootstrap-{index + 1}",
            scopes=list(SCOPES),
        )
        created += 1 if result.get("created") else 0
    clear_auth_cache()
    # NOTE: never name an `extra` key "created" (or "message", "asctime", ...): those are
    # reserved LogRecord attributes and logging raises KeyError instead of emitting. The test
    # suite runs at WARNING, so an INFO-only crash here would surface first in production.
    log.info("bootstrap api keys ensured", extra={"total": len(keys), "keys_created": created})
    return created
