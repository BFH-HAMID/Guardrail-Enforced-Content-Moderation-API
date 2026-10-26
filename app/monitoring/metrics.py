"""Prometheus metrics for the moderation chain.

Names follow the conventional ``subsystem_metric_total`` scheme so PromQL reads
naturally (``rate(moderation_requests_total{status="blocked"}[5m])``).

Cardinality is the thing to watch on a metrics endpoint, so labels are enumerated and
bounded: ``status``/``mode``/``category``/``severity``/``stage``/``provider`` are all
short closed sets from config, never user input. Latency uses explicit buckets tuned
for a 3-stage LLM chain (10ms floor for rules-only requests, 60s ceiling for a slow
generation), and ``/stats`` can be served from these counters as well as from SQL.

If ``prometheus_client`` is missing, every function degrades to a no-op instead of
breaking the API - metrics must never be a reason a request fails.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import Any

from app.core.logging import get_logger

log = get_logger(__name__)

try:  # pragma: no cover - trivial import guard
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PROMETHEUS_AVAILABLE = False

# Latency buckets (seconds): rules-only fast path -> full LLM chain worst case.
_LATENCY_BUCKETS: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0)
_STAGE_BUCKETS: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 45.0)

STATUS_LABELS = ("allowed", "rewritten", "blocked")
CATEGORY_LABELS = (
    "hate", "harassment", "violence", "self_harm", "sexual", "nsfw", "child_safety", "weapons",
    "drugs", "fraud", "pii", "doxxing", "privacy", "copyright", "legal", "regulated_advice",
    "prompt_injection", "jailbreak", "obfuscation", "system_prompt_leak", "pii_leak",
    "unsafe_advice", "toxic", "toxic_output", "sensitive_topic", "dangerous_goods", "gore",
    "moderation_error", "other",
)
SEVERITY_LABELS = ("info", "low", "medium", "high", "critical")


class _NoopMetric:
    """Stand-in used when prometheus_client is unavailable."""

    def labels(self, *_a: Any, **_k: Any) -> _NoopMetric:
        return self

    def inc(self, *_a: Any, **_k: Any) -> None:
        return None

    def dec(self, *_a: Any, **_k: Any) -> None:
        return None

    def set(self, *_a: Any, **_k: Any) -> None:
        return None

    def observe(self, *_a: Any, **_k: Any) -> None:
        return None


class ModerationMetrics:
    """Bundle of every metric, created once per process."""

    #: trailing window for :attr:`block_rate`; keeps the gauge meaningful without
    #: requiring recording_rules on the Prometheus side for a demo dashboard.
    WINDOW_SECONDS = 300.0

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and PROMETHEUS_AVAILABLE
        self._outcomes: deque[tuple[float, bool]] = deque(maxlen=10000)
        self.registry: Any = CollectorRegistry() if self.enabled else None
        if not self.enabled:
            noop = _NoopMetric()
            for name in (
                "requests", "violations", "stage_latency", "pipeline_latency", "llm_errors", "llm_tokens",
                "llm_cost", "block_rate", "window_size", "in_flight", "text_length", "degraded", "fail_closed",
                "rate_limited", "auth_failures", "db_errors", "retention_purged", "http_requests", "http_latency", "info",
            ):
                setattr(self, name, noop)
            return

        registry_kwargs: dict[str, Any] = {"registry": self.registry}
        self.requests: Any = Counter(
            "moderation_requests_total",
            "Moderation requests completed, by outcome status and mode.",
            ["status", "mode"],
            **registry_kwargs,
        )
        self.violations: Any = Counter(
            "moderation_violations_total",
            "Policy violations recorded, by category and severity.",
            ["category", "severity"],
            **registry_kwargs,
        )
        self.stage_latency: Any = Histogram(
            "stage_latency_seconds",
            "Wall time of a single pipeline stage.",
            ["stage"],
            buckets=_STAGE_BUCKETS,
            **registry_kwargs,
        )
        self.pipeline_latency: Any = Histogram(
            "pipeline_latency_seconds",
            "End-to-end latency of the moderation chain.",
            ["mode", "status"],
            buckets=_LATENCY_BUCKETS,
            **registry_kwargs,
        )
        self.llm_errors: Any = Counter(
            "llm_errors_total",
            "LLM call failures by provider and error class.",
            ["provider", "kind"],
            **registry_kwargs,
        )
        self.llm_tokens: Any = Counter(
            "llm_tokens_total",
            "Tokens consumed by the moderation chain.",
            ["provider", "type"],
            **registry_kwargs,
        )
        self.llm_cost: Any = Counter(
            "llm_cost_usd_total",
            "Estimated LLM cost in USD (from configured per-1k prices).",
            ["provider"],
            **registry_kwargs,
        )
        self.block_rate: Any = Gauge(
            "moderation_block_rate_ratio",
            "Share of requests blocked over the trailing window (see block_rate_window_seconds).",
            **registry_kwargs,
        )
        self.window_size: Any = Gauge(
            "moderation_block_rate_sample_size",
            "Requests inside the trailing block-rate window.",
            **registry_kwargs,
        )
        self.in_flight: Any = Gauge(
            "moderation_requests_in_flight",
            "Requests currently inside the moderation chain.",
            **registry_kwargs,
        )
        self.text_length: Any = Histogram(
            "moderation_input_chars",
            "Input length in characters after normalization.",
            buckets=(16, 64, 128, 256, 512, 1024, 2048, 4096, 8192),
            **registry_kwargs,
        )
        self.degraded: Any = Counter(
            "moderation_degraded_total",
            "Responses produced with a degraded detector layer (judge/classifier unavailable).",
            ["stage"],
            **registry_kwargs,
        )
        self.fail_closed: Any = Counter(
            "moderation_fail_closed_total",
            "Requests blocked because moderation itself failed.",
            ["stage"],
            **registry_kwargs,
        )
        self.rate_limited: Any = Counter(
            "moderation_rate_limited_total",
            "Requests rejected by the per-key rate limiter.",
            **registry_kwargs,
        )
        self.auth_failures: Any = Counter(
            "moderation_auth_failures_total",
            "Authentication failures (missing/unknown/revoked key).",
            ["reason"],
            **registry_kwargs,
        )
        self.db_errors: Any = Counter(
            "moderation_db_errors_total",
            "Failures writing or reading the moderation log.",
            ["operation"],
            **registry_kwargs,
        )
        self.retention_purged: Any = Counter(
            "moderation_retention_purged_total",
            "Log rows deleted by the retention sweep.",
            **registry_kwargs,
        )
        self.http_requests: Any = Counter(
            "http_requests_total",
            "HTTP requests served, by method, route and status code.",
            ["method", "path", "status"],
            **registry_kwargs,
        )
        self.http_latency: Any = Histogram(
            "http_request_duration_seconds",
            "HTTP request latency including auth, moderation and persistence.",
            ["path"],
            buckets=_LATENCY_BUCKETS,
            **registry_kwargs,
        )
        self.info: Any = Info(
            "moderation_build_info",
            "Build/version metadata for this service.",
            **registry_kwargs,
        )

    # ------------------------------------------------------------------ record
    def record(self, result: Any) -> None:
        """One call per completed request - the single write path for request metrics."""
        if not self.enabled:
            return
        try:
            status = getattr(result.status, "value", result.status)
            mode = result.mode
            self.requests.labels(status=status, mode=mode).inc()
            self.pipeline_latency.labels(mode=mode, status=status).observe(max(0.0, result.latency_ms / 1000.0))
            for stage, ms in (result.stage_latency_ms or {}).items():
                self.stage_latency.labels(stage=stage).observe(max(0.0, float(ms) / 1000.0))
            for violation in result.violations:
                self.violations.labels(
                    category=violation.category if violation.category in CATEGORY_LABELS else "other",
                    severity=violation.severity if violation.severity in SEVERITY_LABELS else "medium",
                ).inc()
            if result.degraded:
                self.degraded.labels(stage="pipeline").inc()
            if result.fail_closed:
                self.fail_closed.labels(stage="pipeline").inc()
            usage = result.token_usage or {}
            provider = str(usage.get("provider") or "none")
            for kind in ("prompt", "completion"):
                count = int(usage.get(f"{kind}_tokens") or 0)
                if count:
                    self.llm_tokens.labels(provider=provider, type=kind).inc(count)
            if result.estimated_cost_usd:
                self.llm_cost.labels(provider=provider).inc(float(result.estimated_cost_usd))
            self.text_length.observe(float(len(result.original_text or "")))
            self._note_outcome(status == "blocked")
        except Exception as exc:  # pragma: no cover - metrics must never break a request
            log.debug("metric record failed", extra={"error": str(exc)[:160]})

    def _note_outcome(self, blocked: bool) -> None:
        now = time.monotonic()
        self._outcomes.append((now, blocked))
        cutoff = now - self.WINDOW_SECONDS
        while self._outcomes and self._outcomes[0][0] < cutoff:
            self._outcomes.popleft()
        total = len(self._outcomes)
        if total:
            self.block_rate.set(sum(1 for _t, b in self._outcomes if b) / total)
            self.window_size.set(total)

    def window_stats(self) -> dict[str, Any]:
        cutoff = time.monotonic() - self.WINDOW_SECONDS
        window = [b for t, b in self._outcomes if t >= cutoff]
        total = len(window)
        return {
            "window_seconds": self.WINDOW_SECONDS,
            "requests": total,
            "blocked": sum(window),
            "block_rate": round(sum(window) / total, 4) if total else 0.0,
        }

    def describe_build(self, *, version: str, provider: str, policy_version: str) -> None:
        if self.enabled:
            self.info.info({"version": version, "llm_provider": provider, "policy_version": policy_version})


_METRICS: ModerationMetrics | None = None


def get_metrics() -> ModerationMetrics:
    global _METRICS
    if _METRICS is None:
        from app.core.config import get_settings

        _METRICS = ModerationMetrics(enabled=get_settings().metrics_enabled)
    return _METRICS


def reset_metrics() -> None:  # pragma: no cover - test helper
    global _METRICS
    _METRICS = None


def render_prometheus() -> bytes:
    """Payload for ``GET /metrics`` (text exposition format)."""
    metrics = get_metrics()
    if not metrics.enabled:
        return b"# prometheus_client not installed or metrics disabled\n"
    return generate_latest(metrics.registry)


def metrics_middleware_wrapper(handler: Callable[..., Any]) -> Callable[..., Any]:  # pragma: no cover - thin helper
    """Expose ``in_flight`` for callers that want it without importing the bundle."""
    return handler
