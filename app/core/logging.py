"""Structured logging with a request-id correlation key.

Every log line is JSON (or a readable single line in dev) and carries the current
``request_id``, so a single moderation decision can be followed across middleware,
pipeline stages, LLM calls and the DB write.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from typing import Any

REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
USER_KEY: contextvars.ContextVar[str] = contextvars.ContextVar("api_key", default="-")

_RESERVED = frozenset(
    {"args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName", "levelname", "levelno",
     "lineno", "module", "msecs", "message", "msg", "name", "pathname", "process", "processName",
     "relativeCreated", "stack_info", "threadName", "taskName"}
)

_LOGGED_ATTEMPTS: set[int] = set()


def bind_request_id(request_id: str) -> contextvars.Token[str]:
    return REQUEST_ID.set(request_id)


def bind_api_key(key_id: str) -> contextvars.Token[str]:
    return USER_KEY.set(key_id)


class JsonLineFormatter(logging.Formatter):
    """One JSON object per line - what Loki/CloudWatch/Promtail want."""

    def __init__(self, *, max_field_chars: int = 2000, include_extra: bool = True) -> None:
        super().__init__(datefmt="%Y-%m-%dT%H:%M:%S%z")
        self.max_field_chars = max_field_chars
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": REQUEST_ID.get(),
            "api_key": USER_KEY.get(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key in _RESERVED or key.startswith("_"):
                    continue
                payload[key] = self._shrink(value)
        return json.dumps(payload, ensure_ascii=False, default=self._default)

    def _shrink(self, value: Any) -> Any:
        if isinstance(value, str) and len(value) > self.max_field_chars:
            return value[: self.max_field_chars] + f"…(+{len(value) - self.max_field_chars})"
        if isinstance(value, dict):
            return {k: self._shrink(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)) and len(value) > 50:
            return [self._shrink(v) for v in value[:50]] + [f"…+{len(value) - 50} more"]
        return value

    @staticmethod
    def _default(obj: Any) -> Any:
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)
        if isinstance(obj, bytes):
            return obj.decode("utf-8", "replace")
        return repr(obj)


class DevFormatter(logging.Formatter):
    """Human-readable fallback for local runs (``GUARD_LOG_JSON=false``)."""

    def format(self, record: logging.LogRecord) -> str:
        extra = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_")
        }
        suffix = " ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""
        rid = REQUEST_ID.get()
        base = f"{time.strftime('%H:%M:%S', time.gmtime(record.created))} {record.levelname:<7} [{rid[:8]}] {record.getMessage()}"
        return f"{base} {suffix}".rstrip()


def configure_logging(*, level: str = "INFO", json_logs: bool = True, max_field_chars: int = 2000) -> None:
    """Idempotent root-logger setup; safe to call from tests, uvicorn and Streamlit."""
    root = logging.getLogger()
    handler_key = id(root)
    if handler_key in _LOGGED_ATTEMPTS and root.handlers:
        root.setLevel(level.upper())
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter: logging.Formatter
    if json_logs:
        formatter = JsonLineFormatter(max_field_chars=max_field_chars)
    else:
        formatter = DevFormatter()
    handler.setFormatter(formatter)

    root.handlers = [handler]
    root.setLevel(level.upper())
    # uvicorn installs its own handlers; route everything through ours instead.
    for noisy in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi", "httpx", "asyncio"):
        lg = logging.getLogger(noisy)
        lg.handlers = []
        lg.propagate = True
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    _LOGGED_ATTEMPTS.add(handler_key)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class LogContext:
    """``with LogContext(stage="stage1", chars=120):`` -> timed, structured lines."""

    __slots__ = ("logger", "fields", "started")

    def __init__(self, logger: logging.Logger | str, **fields: Any) -> None:
        self.logger = logging.getLogger(logger) if isinstance(logger, str) else logger
        self.fields = fields
        self.started = 0.0

    def __enter__(self) -> "LogContext":
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        elapsed_ms = round((time.perf_counter() - self.started) * 1000, 2)
        if exc_type is None:
            self.logger.info(
                str(self.fields.pop("event", "completed")) + f" in {elapsed_ms}ms",
                extra={**self.fields, "duration_ms": elapsed_ms},
            )
        else:
            self.logger.error(
                f"{self.fields.pop('event', 'completed')} failed after {elapsed_ms}ms",
                extra={**self.fields, "duration_ms": elapsed_ms},
                exc_info=exc,
            )
        return False
