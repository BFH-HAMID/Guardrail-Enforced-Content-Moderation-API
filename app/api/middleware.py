"""HTTP middleware: request id, structured access log, timing, security headers, metrics.

Ordering matters and is fixed in :func:`app.main.create_app`:

    CORSMiddleware -> RequestIdMiddleware -> AccessLogMiddleware -> SecurityHeaders -> route

so the request id exists before anything logs, and CORS headers are added on the way
out of the outermost layer (a 429/500 response still needs them, or the browser demo
cannot read the error body).
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import get_settings
from app.core.logging import bind_request_id, get_logger

log = get_logger(__name__)

_VALID_INCOMING_ID = re.compile(r"^[A-Za-z0-9._:-]{6,64}$")
SKIP_LOG_PATHS = frozenset({"/metrics", "/health", "/favicon.ico", "/docs", "/openapi.json", "/redoc"})

SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.gstatic.com; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; img-src 'self' data:; font-src 'self' data: https://fonts.gstatic.com; connect-src 'self'",
}


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Ensure every request has a request id, echo it, and bind it to the log context."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        settings = get_settings()
        incoming = request.headers.get(settings.request_id_header) or request.headers.get("x-correlation-id")
        request_id = incoming if incoming and _VALID_INCOMING_ID.match(incoming) else f"req_{uuid.uuid4().hex[:24]}"
        bind_request_id(request_id)
        request.state.request_id = request_id
        request.state.started_at = time.perf_counter()
        response = await call_next(request)
        response.headers[settings.request_id_header] = request_id
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """One JSON line per request + HTTP-level Prometheus counters."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        settings = get_settings()
        started = getattr(request.state, "started_at", None) or time.perf_counter()
        metrics = _metrics()
        if metrics is not None:
            metrics.in_flight.inc()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            # Let the app-level handler produce the fail-closed body; we only log here.
            duration_ms = (time.perf_counter() - started) * 1000
            log.exception(
                "unhandled request error",
                extra={"method": request.method, "path": request.url.path, "duration_ms": round(duration_ms, 2), "status": 500},
            )
            raise
        finally:
            if metrics is not None:
                metrics.in_flight.dec()
        duration_ms = (time.perf_counter() - started) * 1000
        request.state.duration_ms = duration_ms

        for header, value in (getattr(request.state, "rate_headers", None) or {}).items():
            response.headers.setdefault(header, str(value))
        response.headers.setdefault(settings.request_id_header, str(getattr(request.state, "request_id", "-")))
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"

        if request.url.path not in SKIP_LOG_PATHS:
            log.info(
                f"{request.method} {request.url.path} -> {status_code}",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "duration_ms": round(duration_ms, 2),
                    "client": _client(request),
                    "api_key": getattr(getattr(request.state, "auth", None), "key_id", "-"),
                    "query": dict(request.query_params) or None,
                    "ua": (request.headers.get("user-agent") or "")[:120],
                },
            )
        if metrics is not None:
            metrics.http_requests.labels(method=request.method, path=_route_label(request), status=str(status_code)).inc()
            metrics.http_latency.labels(path=_route_label(request)).observe(duration_ms / 1000.0)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Baseline hardening headers. The API returns text, never markup."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response


def _route_label(request: Request) -> str:
    """Cardinality-safe path label: the matched route template, or ``other``."""
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if path:
        return path
    known = {"/moderate", "/logs", "/stats", "/health", "/metrics", "/policies", "/evaluate", "/keys"}
    clean = request.url.path.rstrip("/")
    return clean if clean in known else "other"


def _client(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _metrics() -> Any:
    try:
        from app.monitoring.metrics import get_metrics

        return get_metrics()
    except Exception:  # pragma: no cover - metrics must never break serving
        return None


class TimeoutMiddleware(BaseHTTPMiddleware):
    """ASGI-level backstop for handlers that ignore their own budget.

    The pipeline already has per-stage and global timeouts; this exists so a *different*
    slow path (a stuck DB call in /logs, say) cannot pin a worker forever behind a proxy.
    """

    def __init__(self, app: Any, *, timeout_s: float = 90.0) -> None:
        super().__init__(app)
        self.timeout_s = timeout_s

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        import anyio

        try:
            with anyio.fail_after(self.timeout_s):
                return await call_next(request)
        except TimeoutError:
            log.error("request exceeded the ASGI timeout", extra={"path": request.url.path, "timeout_s": self.timeout_s})
            return Response(
                content='{"error":{"code":"gateway_timeout","message":"Request timed out; treat this as a blocked moderation result."}}',
                status_code=504,
                media_type="application/json",
            )
