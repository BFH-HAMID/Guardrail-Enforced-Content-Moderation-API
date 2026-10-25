"""Post-processing layer: metadata enrichment, storage policy, DB write, metrics.

Runs after the chain, before the response is formatted. Three things happen here and
all three are independent by design:

1. **Metadata** - a structured record (status, categories, aggregate confidence,
   action, policy version, model versions, timings) is attached to the result.
2. **Storage** - the row is written according to the *storage policy*, which is where
   privacy lives: ``full`` / ``truncated`` / ``redacted`` (hash only) / ``encrypted``
   (Fernet at rest). ``user_id`` is always HMAC-hashed.
3. **Metrics** - Prometheus counters/histograms are updated.

Order matters: metrics are updated *before* the DB write is awaited, so a database
outage cannot blind the dashboard, and a metrics failure cannot fail a request. A DB
write failure is recorded and returned in the response as ``storage="failed"`` - the
moderation decision still stands, because the caller's content was already filtered.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.security import get_cipher, hash_text, hash_user_id
from app.core.text import redact_middle, truncate_for_storage
from app.monitoring.metrics import get_metrics
from app.pipeline.base import PipelineResult, Status

log = get_logger(__name__)

TEXT_STORAGE_MODES = ("full", "truncated", "redacted", "encrypted")


@dataclass(slots=True)
class PostProcessReport:
    """What post-processing actually did - surfaced in /logs and the response."""

    metadata: dict[str, Any] = field(default_factory=dict)
    stored: bool = False
    storage: str = "skipped"  # ok | failed | skipped
    log_id: str | None = None
    input_text_mode: str = "full"
    metrics_updated: bool = False
    duration_ms: float = 0.0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stored": self.stored,
            "storage": self.storage,
            "log_id": self.log_id,
            "input_text_mode": self.input_text_mode,
            "metrics_updated": self.metrics_updated,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
        }


def aggregate_confidence(result: PipelineResult) -> float:
    """Single 0..1 number for the response and the progress bar in the UI.

    Combines the per-stage confidences with the agreement between them: three stages
    all near "clean" is a confident allow; a stage-1 score of 0.69 that slipped past a
    0.72 threshold should not read as 0.99 confident.
    """
    scores = result.scores or {}
    peak = max((value for value in scores.values() if isinstance(value, (int, float))), default=0.0)
    blocked = result.status is Status.BLOCKED
    stages_present = sum(1 for name in ("stage1", "stage2", "stage3") if result.trace and any(s.name.startswith(name) and s.status == "ok" for s in result.trace.stages))
    coverage = stages_present / 3.0
    base = 0.9 if blocked else 0.95
    proximity_penalty = 0.35 * max(0.0, 1.0 - abs(1.0 - peak)) if not blocked else 0.0
    value = base * (0.6 + 0.4 * coverage) - proximity_penalty - (0.12 if result.degraded else 0.0)
    if blocked:
        value += 0.15 * min(peak, 1.0)
    return round(max(0.05, min(0.99, value)), 4)


def build_metadata(result: PipelineResult, *, ctx_extras: dict[str, Any] | None = None) -> dict[str, Any]:
    """Structured, log-ready metadata for one decision."""
    now = dt.datetime.now(dt.timezone.utc)
    categories = sorted({v.category for v in result.violations} | set(result.categories))
    severities = [v.severity for v in result.violations]
    severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
    top_severity = max(severities, key=lambda s: severity_rank.get(s, 0)) if severities else None
    settings = get_settings()
    metadata = {
        "moderation_status": result.status.value,
        "violation_categories": categories,
        "violation_count": len(result.violations),
        "top_severity": top_severity,
        "confidence_score": round(result.confidence, 4),
        "action_taken": result.action_taken,
        "policy_version": result.policy_version,
        "model_versions": result.model_versions,
        "request_id": result.request_id,
        "mode": result.mode,
        "timestamp": now.isoformat(),
        "timestamp_epoch": now.timestamp(),
        "latency_ms": round(result.latency_ms, 2),
        "stage_latency_ms": result.stage_latency_ms,
        "scores": result.scores,
        "reason": result.reason,
        "blocked_reason": result.blocked_reason,
        "disclaimer_attached": bool(result.disclaimers),
        "degraded": result.degraded,
        "fail_closed": result.fail_closed,
        "token_usage": result.token_usage,
        "estimated_cost_usd": result.estimated_cost_usd,
        "text_pipeline": {
            "normalized": bool((ctx_extras or {}).get("normalized")),
            "obfuscation_score": (ctx_extras or {}).get("obfuscation_score", 0.0),
            "flags": (ctx_extras or {}).get("obfuscation_flags", []),
            "payloads": (ctx_extras or {}).get("payload_signals", {}),
            "script_mix": (ctx_extras or {}).get("script_mix", {}),
        },
        "retention_days": settings.retention_days,
        "input_storage": settings.store_raw_input and "on" or "off",
        "service": {"name": settings.app_name, "version": settings.version, "environment": settings.environment},
    }
    if ctx_extras:
        for key in ("key_id", "user_id_hash", "channel", "language", "request_headers"):
            if key in ctx_extras and ctx_extras[key] is not None:
                metadata[key] = ctx_extras[key]
        metadata["client_metadata"] = ctx_extras.get("metadata", {})
    return metadata


def text_storage_payload(result: PipelineResult) -> dict[str, Any]:
    """Apply the storage policy to the input/output text before it hits SQL."""
    settings = get_settings()
    text = result.original_text or ""
    mode = "full"
    stored_input: str | None = text

    if not settings.store_raw_input:
        mode = "redacted"
        stored_input = redact_middle(text, max(40, settings.trace_preview_chars))
    elif settings.store_max_chars and len(text) > settings.store_max_chars:
        mode = "truncated"
        stored_input, _changed = truncate_for_storage(text, settings.store_max_chars)

    cipher = get_cipher()
    if mode in {"full", "truncated"} and getattr(cipher, "scheme", "none") != "none":
        try:
            stored_input = cipher.encrypt(stored_input or "")
            mode = "encrypted"
        except Exception as exc:  # pragma: no cover - misconfigured key must not drop the row
            log.warning("text encryption unavailable; storing under storage policy without it", extra={"error": str(exc)[:160]})
    if stored_input is not None and mode != "encrypted":
        stored_input = stored_input[: max(settings.store_max_chars, 256) + 128]

    output = result.safe_output
    if output and mode == "encrypted":
        try:
            output = cipher.encrypt(output)
        except Exception:  # pragma: no cover
            output = output[: max(settings.store_max_chars, 256)]
    return {
        "input_text": stored_input,
        "input_text_mode": mode,
        "input_hash": hash_text(text),
        "input_chars": len(text),
        "output_text": (output or "")[: max(settings.store_max_chars, 256) + 128] if output else None,
    }


def _row_payload(result: PipelineResult, metadata: dict[str, Any], storage: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    created = _parse_ts(metadata.get("timestamp")) or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    cutoff = created + dt.timedelta(days=settings.retention_days) if settings.retention_days > 0 else None
    return {
        "request_id": result.request_id,
        "api_key_id": metadata.get("key_id"),
        "user_id_hash": metadata.get("user_id_hash"),
        "created_at": created,
        "finished_at": created,
        "latency_ms": result.latency_ms,
        "stage_latency": result.stage_latency_ms,
        "status": result.status.value,
        "action_taken": result.action_taken,
        "reason": (result.reason or "")[:120],
        "confidence": result.confidence,
        "mode": result.mode,
        "channel": metadata.get("channel"),
        "language": metadata.get("language"),
        "scores": result.scores,
        "trace_json": result.trace.as_dict() if (result.trace and settings.store_trace) else None,
        "model_versions": result.model_versions,
        "policy_version": result.policy_version,
        "metadata_json": {
            "violation_categories": metadata["violation_categories"],
            "top_severity": metadata["top_severity"],
            "disclaimer_attached": metadata["disclaimer_attached"],
            "text_pipeline": metadata["text_pipeline"],
            "client_metadata": metadata.get("client_metadata", {}),
        },
        "prompt_tokens": int((result.token_usage or {}).get("prompt_tokens") or 0),
        "completion_tokens": int((result.token_usage or {}).get("completion_tokens") or 0),
        "estimated_cost_usd": result.estimated_cost_usd,
        "degraded": result.degraded,
        "fail_closed": result.fail_closed,
        "error": result.error,
        "obfuscation_score": result.scores.get("obfuscation", 0.0),
        "retention_expires_at": cutoff,
        "violations": [v.as_dict() for v in result.violations],
        **storage,
    }


def _parse_ts(value: Any) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:  # pragma: no cover - defensive
        return None


async def finalize(
    result: PipelineResult,
    *,
    ctx_extras: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> tuple[PipelineResult, PostProcessReport]:
    """Enrich -> store -> meter. Returns the (possibly updated) result and a report."""
    started = time.perf_counter()
    settings = settings or get_settings()
    metrics = get_metrics()

    metadata = build_metadata(result, ctx_extras=ctx_extras)
    if metadata.get("user_id_hash") is None and (ctx_extras or {}).get("user_id"):
        metadata["user_id_hash"] = hash_user_id(str(ctx_extras["user_id"]))
    result.confidence = aggregate_confidence(result)
    metadata["confidence_score"] = result.confidence

    storage = text_storage_payload(result)
    report = PostProcessReport(metadata=metadata, input_text_mode=storage["input_text_mode"])

    # 1) metrics first: never let a DB problem blind the dashboard
    try:
        metrics.record(result)
        report.metrics_updated = True
    except Exception as exc:  # pragma: no cover - metrics are already defensive
        log.debug("metrics update failed", extra={"error": str(exc)[:160]})

    # 2) then persistence
    if settings.database_url:
        try:
            from app.db.repository import insert_moderation_log

            row_id = await insert_moderation_log(_row_payload(result, metadata, storage))
            report.stored = row_id is not None
            report.storage = "ok" if row_id else "duplicate"
            report.log_id = row_id
        except Exception as exc:
            report.storage = "failed"
            report.error = f"{exc.__class__.__name__}: {exc}"[:300]
            metrics.db_errors.labels(operation="insert").inc()
            log.error("moderation log could not be persisted", extra={"request_id": result.request_id, "error": report.error[:200]})
    else:
        report.storage = "skipped"

    report.duration_ms = (time.perf_counter() - started) * 1000
    metadata["postprocess"] = report.as_dict()
    log.info(
        "moderation completed",
        extra={
            "request_id": result.request_id,
            "status": result.status.value,
            "action": result.action_taken,
            "categories": metadata["violation_categories"][:6],
            "latency_ms": round(result.latency_ms, 2),
            "confidence": result.confidence,
            "stored": report.stored,
        },
    )
    return result, report


async def retention_sweep() -> dict[str, Any]:
    """Manual/CLI entry point for the retention job (also run by the app scheduler)."""
    from app.db.session import purge_expired_logs
    from app.monitoring.metrics import get_metrics

    deleted = await purge_expired_logs()
    if deleted:
        get_metrics().retention_purged.inc(deleted)
    return {"deleted": deleted, "at": dt.datetime.now(dt.timezone.utc).isoformat()}
