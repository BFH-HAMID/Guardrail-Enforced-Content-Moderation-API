"""HTTP routes.

Endpoint map (all JSON unless noted):

======================================  ==========  =========================================
Route                                   Scope       Purpose
======================================  ==========  =========================================
``POST /moderate``                      moderate    run the 3-stage chain, return v1 envelope
``POST /evaluate``                      moderate    run bundled/passed datasets, score modes
``GET  /logs``                          read_logs   filtered, paginated decision log
``GET  /logs/export.csv``               read_logs   CSV for spreadsheet analysis
``GET  /logs/{request_id}``             read_logs   one full record incl. trace
``GET  /explain/{request_id}``          read_logs   human-readable pipeline trace
``GET  /stats``                         read_logs   aggregates powering the UI stats tab
``GET  /policies``                      -           active policy bundle summary
``GET  /health``                        -           liveness + component readiness
``GET  /metrics``                       -           Prometheus exposition
``POST /keys`` ``DELETE /keys/{id}``     admin       issue / revoke API keys
``POST /admin/retention`` ``POST /admin/reload-policies``  admin  ops triggers
``GET  /``                              -           service metadata + docs links
======================================  ==========  =========================================

The ``/moderate`` handler is deliberately thin: normalize -> run chain -> post-process
-> format. Every decision lives in the pipeline, so the HTTP layer cannot become a
second, divergent moderation implementation.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import time
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, Response, status

from app.api.auth import AuthContext, authenticate, bootstrap_keys_from_settings, clear_auth_cache, require_scope
from app.api.schemas import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    EvaluateRequest,
    ErrorResponse,
    HealthResponse,
    LogListResponse,
    ModerateRequest,
    ModerationResponse,
    StatsResponse,
)
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.security import generate_api_key, hash_api_key, hash_user_id, key_id_of
from app.monitoring.metrics import get_metrics, render_prometheus
from app.pipeline.base import ModerationMode, PipelineResult, Status
from app.pipeline.orchestrator import ModerationPipeline
from app.post.formatter import SCHEMA_VERSION, format_response, openapi_examples
from app.post.postprocess import finalize, retention_sweep

log = get_logger(__name__)

router = APIRouter()

DESCRIPTIONS = {
    "/moderate": (
        "Run the guardrail chain over untrusted text. Returns a `v1` envelope. A blocked "
        "response never contains the original text; a pipeline failure returns HTTP 503 "
        "with `status=blocked` and `reason=moderation_error` (fail closed) - clients must "
        "treat any non-2xx on this route as 'blocked' and must not display the input."
    ),
}

_RESPONSE_EXAMPLES = openapi_examples()


@router.post(
    "/moderate",
    response_model=ModerationResponse,
    response_model_by_alias=True,
    status_code=status.HTTP_200_OK,
    summary="Moderate text through the 3-stage chain",
    description=DESCRIPTIONS["/moderate"],
    responses={
        200: {
            "model": ModerationResponse,
            "description": "Moderation decision. See the four examples: allowed / rewritten / blocked(policy) / blocked(fail-closed).",
            "content": {"application/json": {"examples": _RESPONSE_EXAMPLES}},
        },
        422: {"model": ErrorResponse, "description": "Input too long / blank / invalid schema"},
        429: {"model": ErrorResponse, "description": "Per-key rate limit exceeded"},
        503: {"model": ModerationResponse, "description": "Fail-closed: moderation could not run. Treat as blocked."},
    },
)
async def moderate(request: Request, payload: ModerateRequest, auth: AuthContext = Depends(require_scope("moderate"))) -> Response:
    settings = get_settings()
    pipeline: ModerationPipeline = request.app.state.pipeline
    started = time.perf_counter()

    metadata = payload.metadata.flattened() if payload.metadata else {}
    user_id_hash = hash_user_id(str(metadata["user_id"])) if metadata.get("user_id") and settings.hash_user_ids else None
    # "anonymous" (auth disabled) is not a row in api_keys, so it must not be written
    # into the FK column - it is recorded in metadata instead.
    key_id = auth.key_id if auth.key_id and auth.key_id != "anonymous" else None

    try:
        result = await pipeline.run(
            payload.text,
            request_id=request.state.request_id,
            mode=ModerationMode(payload.mode),
            metadata=metadata,
            return_reasoning=payload.return_reasoning,
            key_id=key_id,
            user_id_hash=user_id_hash,
        )
    except Exception as exc:  # pragma: no cover - the chain has its own guards; this is the backstop
        # Raised *before* a verdict exists (bad config, policy file unreadable, monkeypatched
        # stage). The one response that must never happen here is a pass.
        log.exception("moderation chain raised; returning the fail-closed envelope", extra={"error": f"{exc.__class__.__name__}: {exc}"[:200]})
        return await fail_closed_response(request.state.request_id, f"{exc.__class__.__name__}: {exc}"[:240], stage="route")
    if not isinstance(result, PipelineResult):  # pragma: no cover - orchestrator contract
        result = await _synthetic_blocked(request.state.request_id, "orchestrator returned a non-result object")

    ctx_extras = {
        "key_id": key_id,
        "caller": auth.key_id,
        "user_id_hash": user_id_hash,
        "channel": metadata.get("channel"),
        "language": metadata.get("language"),
        "metadata": metadata,
        "obfuscation_score": result.scores.get("obfuscation", 0.0),
        "script_mix": metadata.get("_script_mix", {}),
        "payload_signals": {},
        "obfuscation_flags": [],
        "normalized": True,
    }
    result, report = await finalize(result, ctx_extras=ctx_extras, settings=settings)
    body = format_response(
        result,
        include_trace=payload.include_trace or settings.store_trace and payload.return_reasoning,
        postprocess_report=report.as_dict(),
        request_url=str(request.url),
    )
    body["timing"] = {
        "pipeline_ms": round(result.latency_ms, 2),
        "postprocess_ms": round(report.duration_ms, 2),
        "handler_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    if payload.include_trace and result.trace is not None:
        body["pipeline_trace"] = result.trace.as_dict()
    headers = dict(getattr(request.state, "rate_headers", None) or {})
    headers["X-Moderation-Status"] = str(result.status.value)
    headers["X-Policy-Version"] = str(result.policy_version)
    if report.storage == "failed":
        # Keep the failure visible to ops without changing the moderation verdict.
        headers["X-Moderation-Storage"] = "degraded"
    return Response(
        content=json.dumps(body, ensure_ascii=False, default=str).encode("utf-8"),
        media_type="application/json",
        headers=headers,
    )


async def fail_closed_response(request_id: str, detail: str, *, stage: str = "http") -> Response:
    """The single fail-closed envelope, shared by ``/moderate`` and the ASGI backstop.

    Two paths can produce it: the route's own guard around ``pipeline.run`` and the global
    exception handler in ``app.main``. They must be byte-for-byte compatible or a client ends
    up parsing two different shapes for the same failure - which is how "we could not check
    this" quietly turns into "the caller's retry logic treated it as a transient network error".

    The contract: a *blocked* verdict (never a pass), persisted so reviewers see the gap,
    counted in Prometheus, with an empty ``safe_output`` and ``retry_allowed=true`` plus
    ``Retry-After`` - the failure is ours, so the caller should back off and try again.
    """
    get_metrics().fail_closed.labels(stage=stage).inc()
    result = await _synthetic_blocked(request_id, detail)
    try:
        result, _report = await finalize(result, ctx_extras={"fail_closed_handler": True})
    except Exception as exc:  # pragma: no cover - a broken DB must not compound a moderation failure
        log.warning("fail-closed logging failed", extra={"request_id": request_id, "error": str(exc)[:160]})
    return Response(
        content=json.dumps(format_response(result), default=str),
        media_type="application/json",
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        headers={
            "X-Moderation-Status": "blocked",
            "X-Moderation-Fail-Closed": "true",
            "Retry-After": "5",
        },
    )


async def _synthetic_blocked(request_id: str, detail: str) -> PipelineResult:
    return PipelineResult(
        request_id=request_id,
        status=Status.BLOCKED,
        safe_output="",
        action_taken="blocked",
        reason="moderation_error",
        blocked_reason="moderation_error",
        fail_closed=True,
        degraded=True,
        confidence=0.99,
        categories=["moderation_error"],
        error=detail[:300],
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )


# ------------------------------------------------------------------ logs / stats


@router.get("/logs", response_model=LogListResponse, summary="List moderation decisions", description="Filterable, paginated audit log. Newest first unless `order=asc`.")
async def list_logs(
    auth: AuthContext = Depends(require_scope("read_logs")),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0, le=1_000_000),
    status_filter: str | None = Query(None, alias="status", pattern="^(allowed|rewritten|blocked)$"),
    category: str | None = Query(None, max_length=48),
    policy_id: str | None = Query(None, max_length=64),
    mode: str | None = Query(None, pattern="^(strict|balanced|lenient)$"),
    key_id: str | None = Query(None, max_length=32),
    since: dt.datetime | None = Query(None, description="ISO-8601 timestamp"),
    until: dt.datetime | None = Query(None),
    min_confidence: float | None = Query(None, ge=0.0, le=1.0),
    search: str | None = Query(None, max_length=80),
    order: str = Query("desc", pattern="^(asc|desc)$"),
) -> dict[str, Any]:
    from app.db.repository import list_logs as _list_logs

    try:
        return await _list_logs(
            limit=limit,
            offset=offset,
            status=status_filter,
            category=category,
            policy_id=policy_id,
            mode=mode,
            key_id=key_id,
            since=_naive(since),
            until=_naive(until),
            min_confidence=min_confidence,
            search=search,
            order=order,
        )
    except Exception as exc:
        raise _unavailable("logs", exc) from exc


@router.get("/logs/export.csv", summary="Download moderation logs as CSV", response_class=Response)
async def export_logs_csv(
    auth: AuthContext = Depends(require_scope("read_logs")),
    limit: int = Query(500, ge=1, le=5000),
    status_filter: str | None = Query(None, alias="status", pattern="^(allowed|rewritten|blocked)$"),
    category: str | None = Query(None, max_length=48),
    mode: str | None = Query(None, pattern="^(strict|balanced|lenient)$"),
    since: dt.datetime | None = Query(None),
) -> Response:
    from app.db.repository import list_logs as _list_logs

    page_size = 500
    rows: list[dict[str, Any]] = []
    offset = 0
    while len(rows) < limit:
        batch = await _list_logs(
            limit=min(page_size, limit - len(rows)),
            offset=offset,
            status=status_filter,
            category=category,
            mode=mode,
            since=_naive(since),
        )
        items = batch.get("items") or []
        rows.extend(items)
        if not items or not batch.get("has_more"):
            break
        offset += len(items)

    buffer = io.StringIO()
    columns = [
        "request_id", "created_at", "status", "action_taken", "reason", "mode", "confidence",
        "latency_ms", "categories", "violation_count", "input_chars", "input_hash", "input_preview",
        "output_preview", "channel", "language", "policy_version", "degraded", "fail_closed", "error",
    ]
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        flat = {key: value for key, value in row.items() if key in columns}
        flat["categories"] = "|".join(row.get("categories") or [])
        for key, value in list(flat.items()):
            if isinstance(value, (dict, list)):
                flat[key] = json.dumps(value, ensure_ascii=False)
        writer.writerow(flat)
    filename = f"moderation-logs-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}.csv"
    return Response(
        content=buffer.getvalue().encode("utf-8-sig"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/logs/{request_id}", summary="One moderation record including its trace")
async def get_log(request_id: str, auth: AuthContext = Depends(require_scope("read_logs"))) -> dict[str, Any]:
    from app.db.repository import get_log_by_request_id

    record = await get_log_by_request_id(_safe_id(request_id))
    if record is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": f"No moderation log for request_id {request_id!r}."})
    return record


@router.get("/explain/{request_id}", summary="Readable pipeline trace for one request")
async def explain(request_id: str, auth: AuthContext = Depends(require_scope("read_logs"))) -> dict[str, Any]:
    from app.db.repository import get_log_trace

    record = await get_log_trace(_safe_id(request_id))
    if record is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "No such request id."})
    trace = record.get("trace") or {}
    stages = []
    for stage in trace.get("stages") or []:
        stages.append(
            {
                "stage": stage.get("name"),
                "status": stage.get("status"),
                "decision": stage.get("decision"),
                "latency_ms": stage.get("latency_ms"),
                "engine": stage.get("engine"),
                "categories": stage.get("categories"),
                "confidence": stage.get("confidence"),
                "scores": stage.get("scores"),
                "reasons": (stage.get("detail") or {}).get("reasons", []),
                "error": stage.get("error"),
            }
        )
    return {
        "request_id": record["request_id"],
        "status": record["status"],
        "short_circuited_at": trace.get("short_circuited_at"),
        "total_latency_ms": trace.get("total_latency_ms"),
        "mode": trace.get("mode"),
        "policy_version": trace.get("policy_version"),
        "stages": stages,
        "violations": record.get("violations") or [],
        "narrative": _narrative(record, stages),
    }


def _narrative(record: dict[str, Any], stages: list[dict[str, Any]]) -> str:
    if record.get("status") == "blocked":
        culprit = next((s for s in stages if s.get("decision") == "block"), None)
        name = (culprit or {}).get("stage", "the pipeline")
        cats = ", ".join((culprit or {}).get("categories") or []) or "policy violation"
        return f"Blocked by {name} ({cats}). Later stages were skipped, so no model saw the raw input."
    applied = [s for s in stages if s.get("decision") == "flag"]
    if applied:
        return "Allowed after transformations in " + ", ".join(s["stage"] for s in applied) + "."
    return "No policy fired; the text passed through normalisation only."


@router.get("/stats", response_model=StatsResponse, summary="Aggregates for the dashboard")
async def stats(
    request: Request,
    auth: AuthContext = Depends(require_scope("read_logs")),
    window_days: int = Query(7, ge=1, le=365),
    key_id: str | None = Query(None, max_length=32),
) -> dict[str, Any]:
    from app.db.repository import compute_stats

    try:
        payload = await compute_stats(window_days=window_days, key_id=key_id)
    except Exception as exc:
        raise _unavailable("stats", exc) from exc
    # The in-process window complements SQL: it reflects the *current* traffic, which is
    # what the "live" panel and alerting on block-rate spikes care about.
    payload["metrics_window"] = get_metrics().window_stats()
    payload["pipeline"] = request.app.state.pipeline.describe()
    return payload


# ------------------------------------------------------------------- ops / meta


@router.get("/health", response_model=HealthResponse, summary="Liveness + component readiness")
async def health(request: Request) -> dict[str, Any]:
    from app.db.session import db_readiness
    from app.pipeline.injection import classifier_status
    from app.pipeline.policy import get_policy_bundle

    settings = get_settings()
    checks: list[str] = []
    db = await db_readiness() if settings.database_url else {"ok": True, "url": "disabled"}
    if not db.get("ok"):
        checks.append("database_unavailable")
    bundle = get_policy_bundle(settings.policies_file)
    if not bundle.policies:
        checks.append("policy_bundle_empty")
    classifier = classifier_status()
    if settings.stage1_use_classifier:
        loaded = [info.get("state") for info in classifier.values() if isinstance(info, dict)]
        if loaded and all(state in {"unavailable", "not_loaded"} for state in loaded):
            checks.append("injection_classifier_unavailable")
    provider = settings.resolved_provider()
    if provider == "local" and settings.stage1_use_llm_judge:
        checks.append("llm_judge_offline_rules_only")
    started = getattr(request.app.state, "started_at", time.time())
    status_value = "ok" if not checks else "degraded"
    return {
        "status": status_value,
        "version": settings.version,
        "environment": settings.environment,
        "uptime_s": round(time.time() - started, 2),
        "checks": checks,
        "components": {
            "database": db,
            "llm": {"provider": provider, "small_model": settings.small_model, "main_model": settings.main_model, "chain": ">".join(request.app.state.pipeline.describe()["provider_chain"])},
            "classifier": classifier,
            "metrics": {"enabled": settings.metrics_enabled, "collector": "prometheus_client"},
            "rate_limit": {"enabled": settings.rate_limit_enabled, "rpm": settings.rate_limit_requests_per_minute, "burst": settings.rate_limit_burst},
        },
        "policies": {
            "version": bundle.version,
            "count": len(bundle.policies),
            "checksum": bundle.checksum[:16],
            "source": str(bundle.source_path) if bundle.source_path else "builtin",
        },
    }


@router.get("/metrics", summary="Prometheus exposition", response_class=Response, include_in_schema=False)
async def metrics() -> Response:
    if not get_settings().metrics_enabled:
        return Response(content="# metrics disabled\n", media_type="text/plain; version=0.0.4")
    return Response(content=render_prometheus(), media_type="text/plain; version=0.0.4; charset=utf-8")


@router.get("/policies", summary="Active policy bundle", description="Non-secret view of the loaded YAML: rules, thresholds, version.")
async def policies(
    request: Request,
    include_rules: bool = Query(False, description="Include the full (redacted) policy definitions"),
) -> dict[str, Any]:
    from app.pipeline.policy import get_policy_bundle

    settings = get_settings()
    bundle = get_policy_bundle(settings.policies_file)
    payload: dict[str, Any] = {"summary": bundle.summary(), "pipeline": request.app.state.pipeline.describe()}
    if include_rules:
        payload["policies"] = [
            {
                "id": policy.id,
                "name": policy.name,
                "stage": policy.stage,
                "category": policy.category,
                "severity": policy.severity,
                "action": policy.action,
                "min_score": policy.min_score,
                "description": policy.description.strip()[:300],
                "clause_types": [
                    name
                    for name in ("regex", "lexicon", "classifier", "llm_categories", "obfuscation", "payloads", "script", "entities", "pii_entities", "stage1")
                    if getattr(policy, name, None)
                ],
            }
            for policy in bundle.policies
        ]
    return payload


@router.post("/evaluate", summary="Score the pipeline against a labelled dataset")
async def evaluate(payload: EvaluateRequest, request: Request, auth: AuthContext = Depends(require_scope("moderate"))) -> dict[str, Any]:
    from scripts.evaluate import run_evaluation  # local import: script is optional at runtime

    try:
        return await run_evaluation(
            dataset=payload.dataset,
            limit=payload.limit,
            modes=list(payload.modes),
            include_failures=payload.include_failures,
            pipeline=request.app.state.pipeline,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "dataset_not_found", "message": str(exc)[:200]}) from exc


# ------------------------------------------------------------------------ admin


@router.post("/keys", response_model=ApiKeyCreateResponse, status_code=status.HTTP_201_CREATED, summary="Issue a new API key (admin)")
async def create_key(
    payload: ApiKeyCreateRequest = Body(...),
    auth: AuthContext = Depends(require_scope("admin")),
) -> dict[str, Any]:
    from app.db.repository import upsert_api_key

    key = generate_api_key()
    await upsert_api_key(
        key_id=key_id_of(key),
        key_hash=hash_api_key(key),
        label_hint=key[:9] + "…",
        name=payload.name,
        scopes=payload.scopes,
        rate_limit_rpm=payload.rate_limit_rpm,
        owner_email=payload.owner_email or "",
    )
    clear_auth_cache()
    log.info("api key issued", extra={"key_id": key_id_of(key), "by": auth.key_id})
    return {"key": key, "key_id": key_id_of(key), "name": payload.name, "created": True}


@router.get("/keys", summary="List key metadata (admin; never the keys themselves)")
async def list_keys(auth: AuthContext = Depends(require_scope("admin"))) -> dict[str, Any]:
    from app.db.repository import list_api_keys

    return {"items": await list_api_keys()}


@router.delete("/keys/{key_id}", summary="Revoke an API key (admin)")
async def revoke_key(key_id: str, auth: AuthContext = Depends(require_scope("admin"))) -> dict[str, Any]:
    from app.db.repository import revoke_api_key

    revoked = await revoke_api_key(key_id[:12])
    clear_auth_cache()
    if not revoked:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "No such key id."})
    log.info("api key revoked", extra={"key_id": key_id, "by": auth.key_id})
    return {"revoked": True, "key_id": key_id[:12]}


@router.post("/admin/retention", summary="Run the retention sweep now (admin)")
async def run_retention(auth: AuthContext = Depends(require_scope("admin"))) -> dict[str, Any]:
    return await retention_sweep()


@router.post("/admin/reload-policies", summary="Reload policies.yaml without a restart (admin)")
async def reload_policies(request: Request, auth: AuthContext = Depends(require_scope("admin"))) -> dict[str, Any]:
    from app.pipeline.policy import get_registry

    bundle = get_registry(get_settings().policies_file).reload()
    get_metrics().describe_build(version=get_settings().version, provider=get_settings().resolved_provider(), policy_version=bundle.version)
    return {"reloaded": True, "version": bundle.version, "policies": len(bundle.policies), "checksum": bundle.checksum[:16]}


@router.get("/", summary="Service metadata")
async def index(request: Request) -> dict[str, Any]:
    settings = get_settings()
    return {
        "name": settings.app_name,
        "version": settings.version,
        "schema_version": SCHEMA_VERSION,
        "environment": settings.environment,
        "docs": {"openapi": "/openapi.json", "swagger": "/docs", "redoc": "/redoc"},
        "endpoints": ["/moderate", "/evaluate", "/logs", "/logs/export.csv", "/logs/{request_id}", "/explain/{request_id}", "/stats", "/policies", "/health", "/metrics", "/keys"],
        "auth": {"required": settings.auth_enabled, "header": settings.api_key_header, "how_to_get_a_key": "GUARD_BOOTSTRAP_API_KEYS or POST /keys (admin)"},
        "pipeline": request.app.state.pipeline.describe(),
        "notice": "User text is always treated as data; blocked responses never echo it.",
    }


# --------------------------------------------------------------------- utilities


def _naive(value: dt.datetime | None) -> dt.datetime | None:
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return value


def _safe_id(value: str) -> str:
    cleaned = "".join(ch for ch in value if ch.isalnum() or ch in "_-.")[:64]
    if not cleaned:
        raise HTTPException(status_code=422, detail={"code": "invalid_id", "message": "request_id contains no usable characters."})
    return cleaned


def _unavailable(operation: str, exc: Exception) -> HTTPException:
    get_metrics().db_errors.labels(operation=operation).inc()
    log.error(f"{operation} query failed", extra={"error": f"{exc.__class__.__name__}: {exc}"[:240]})
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": "storage_unavailable", "message": f"The moderation {operation} store is unavailable right now.", "retry": True},
    )


__all__ = ["router", "DESCRIPTIONS", "authenticate", "bootstrap_keys_from_settings"]
