"""FastAPI application factory and entrypoint.

Responsibilities kept here (rather than scattered in handlers):

* middleware order: CORS outermost -> request id -> access log/metrics -> security
  headers -> optional ASGI timeout backstop;
* lifespan: DB bootstrap, bootstrap API keys, policy warm-up, background retention
  sweep, clean shutdown;
* the **global exception handler**. For ``/moderate`` it returns a fail-closed
  moderation envelope (``status=blocked``, ``reason=moderation_error``, empty
  ``safe_output``) with HTTP 503, so an unexpected crash can never turn into "the
  content was fine".

Run with: ``uvicorn app.main:app --host 0.0.0.0 --port 8000``
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, RedirectResponse, Response

from app.api import routes
from app.api.auth import bootstrap_keys_from_settings
from app.api.middleware import AccessLogMiddleware, RequestIdMiddleware, SecurityHeadersMiddleware, TimeoutMiddleware
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.monitoring.metrics import get_metrics
from app.pipeline.orchestrator import ModerationPipeline

log = get_logger(__name__)

OPENAPI_TAGS = [
    {"name": "moderation", "description": "Run the guardrail chain over untrusted text."},
    {"name": "audit", "description": "Moderation log, traces and aggregates."},
    {"name": "operations", "description": "Health, metrics, policies, key management."},
]


def create_app(*, settings: Any | None = None, inject_pipeline: ModerationPipeline | None = None) -> FastAPI:
    """Build the app. ``inject_pipeline`` is how tests swap in a stubbed chain."""
    settings = settings or get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json, max_field_chars=settings.max_log_field_chars)

    with contextlib.suppress(Exception):  # metrics must never block boot
        get_metrics().describe_build(version=settings.version, provider=settings.resolved_provider(), policy_version="loading")

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):  # pragma: no cover - exercised via TestClient in integration tests
        from app.db.session import dispose_engine, init_db

        app.state.started_at = time.time()
        app.state.should_stop = lambda: False
        try:
            await init_db()
            await bootstrap_keys_from_settings()
        except Exception as exc:
            log.error("startup data prep failed", extra={"error": f"{exc.__class__.__name__}: {exc}"[:240]})
            raise
        from app.pipeline.policy import get_policy_bundle

        bundle = get_policy_bundle(settings.policies_file)
        get_metrics().describe_build(version=settings.version, provider=settings.resolved_provider(), policy_version=bundle.version)
        sweep_task = asyncio.create_task(_retention_loop(app, settings))
        log.info(
            "service ready",
            extra={
                "version": settings.version,
                "environment": settings.environment,
                "provider": settings.resolved_provider(),
                "policies": bundle.version,
                "stage3_mode": settings.stage3_mode,
                "auth": settings.auth_enabled,
            },
        )
        try:
            yield
        finally:
            app.state.should_stop = lambda: True
            sweep_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await sweep_task
            await dispose_engine()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=_DESCRIPTION,
        summary="Guardrail-enforced, 3-stage (Model Chain Prompting) text moderation with audit trail.",
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={"name": "Guardrail Moderation API", "url": "https://github.com/BFH-HAMID/Guardrail-Enforced-Content-Moderation-API"},
        license_info={"name": "MIT"},
        servers=[{"url": "", "description": "This instance"}],
    )
    app.state.settings = settings
    app.state.pipeline = inject_pipeline or ModerationPipeline(settings)

    # --- middleware (added innermost-first: last add runs first) ---------------
    if settings.pipeline_request_timeout_s:
        app.add_middleware(TimeoutMiddleware, timeout_s=settings.pipeline_request_timeout_s + 15.0)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestIdMiddleware)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=settings.cors_allow_credentials,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "X-API-Key", "Content-Type", settings.request_id_header],
            expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset", settings.request_id_header, "X-Moderation-Status", "X-Process-Time-Ms"],
            max_age=600,
        )

    app.include_router(routes.router)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> Response:
        detail = exc.detail
        if isinstance(detail, dict):
            payload: dict[str, Any] = {"error": {"code": detail.get("code", "http_error"), "message": detail.get("message", str(detail))}}
            if detail.get("hints"):
                payload["error"]["hints"] = detail["hints"]
            if detail.get("retry_after_s"):
                payload["error"]["retry_after_s"] = detail["retry_after_s"]
        else:
            payload = {"error": {"code": f"http_{exc.status_code}", "message": str(detail)}}
        payload["error"]["request_id"] = getattr(request.state, "request_id", None)
        return JSONResponse(payload, status_code=exc.status_code, headers=dict(getattr(exc, "headers", None) or {}))

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"field": ".".join(str(part) for part in err.get("loc", [])[1:]) or "body", "message": err.get("msg", "invalid"), "type": err.get("type", "")}
            for err in exc.errors()[:12]
        ]
        limit = settings.max_input_chars
        for error in errors:
            if "too_long" in error["type"] and error["field"] == "text":
                error["message"] = f"text exceeds the {limit} character limit"
        return JSONResponse(
            {"error": {"code": "validation_error", "message": "Request validation failed.", "request_id": getattr(request.state, "request_id", None), "details": errors, "hints": [f"Keep 'text' between 1 and {limit} characters."]}},
            status_code=422,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> Response:
        """Fail closed. A crash is never interpreted as a pass."""
        request_id = getattr(request.state, "request_id", "unknown")
        log.exception(
            "unhandled exception",
            extra={"path": request.url.path, "method": request.method, "error": f"{exc.__class__.__name__}: {exc}"[:240]},
        )
        if request.url.path.rstrip("/").endswith("/moderate"):
            # Identical envelope, headers and counters as the route's own guard - both paths call
            # the same function so a fail-closed response cannot drift depending on where the
            # exception surfaced. It also owns the `fail_closed` metric increment.
            return await routes.fail_closed_response(request_id, f"unhandled exception: {exc.__class__.__name__}")
        return JSONResponse(
            {"error": {"code": "internal_error", "message": "Internal server error.", "request_id": request_id}},
            status_code=500,
        )

    @app.get("/ui", include_in_schema=False)
    async def ui_redirect() -> Response:
        return RedirectResponse("/docs")

    def custom_openapi() -> dict[str, Any]:  # pragma: no cover - cosmetic enrichment
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes, tags=OPENAPI_TAGS)
        components = schema.setdefault("components", {})
        components.setdefault("securitySchemes", {})["ApiKeyHeader"] = {
            "type": "apiKey",
            "in": "header",
            "name": settings.api_key_header,
            "description": "Issued by the operator. Stored only as a salted SHA-256 digest.",
        }
        for path_item in schema.get("paths", {}).values():
            for operation in path_item.values():
                if isinstance(operation, dict) and operation.get("summary"):
                    operation.setdefault("security", [{"ApiKeyHeader": []}])
        examples = routes.openapi_examples()
        moderate_responses = schema.get("paths", {}).get("/moderate", {}).get("post", {}).get("responses", {})
        if "200" in moderate_responses:
            content = moderate_responses["200"].setdefault("content", {}).setdefault("application/json", {})
            content["examples"] = examples
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]
    return app


async def _retention_loop(app: FastAPI, settings: Any) -> None:
    """Background retention sweep (auto-delete logs after N days)."""
    from app.db.session import retention_loop

    if settings.retention_days <= 0:
        log.info("retention sweep disabled (GUARD_RETENTION_DAYS<=0)")
        return
    await retention_loop(lambda: getattr(app.state, "should_stop", lambda: False)(), interval_s=float(settings.retention_sweep_interval_s))


_DESCRIPTION = """
Guardrail-enforced content moderation for LLM applications, built as a **Model Chain
Prompting (MCP)** pipeline. Untrusted text never reaches the main model directly:

1. **Stage 1 - toxicity & prompt-injection gate.** A transformer classifier plus a
   deterministic rulebook (encoded payloads, obfuscation, multilingual lexicons) are
   fused with a strict-JSON small-LLM judge that is explicitly forbidden from following
   instructions inside the payload. A block here short-circuits the chain.
2. **Stage 2 - content policy.** YAML-defined rules (`policies.yaml`) mask PII
   (Presidio + Bangladesh-format regex: mobile, NID, birth certificate, passport,
   IBAN, card+Luhn), detect NSFW/privacy/legal risk, and rewrite only the unsafe
   fragments - then re-scan their own output.
3. **Stage 3 - safe final output.** Sanitized text is answered (`assistant` mode) or
   passed through (`passthrough`, the default), and the answer is checked for toxicity,
   PII leakage and prompt disclosure before it is returned.

**Post-processing** writes metadata + full per-stage trace to SQL and Prometheus.

Design guarantees: *fail closed* (any pipeline error returns `blocked`), *user text is
always data* (quarantined behind per-request delimiters), *policies are data* (no code
change to re-tune), *blocked content is never echoed*.
"""

app = create_app()
