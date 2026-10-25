"""Logging and startup-path hygiene.

Why this file exists: ``bootstrap_keys_from_settings()`` logged
``extra={"total": ..., "created": ...}``, and ``created`` is a reserved ``logging.LogRecord``
attribute - so the *first boot of a real deployment* raised ``KeyError: Attempt to overwrite
'created' in LogRecord`` inside startup, which ``app.main`` re-raises, which made uvicorn exit.
The test suite never saw it, because conftest pins ``GUARD_LOG_LEVEL=WARNING`` and the offending
call is at INFO. A crash that only happens on the path the README tells people to run is exactly
the bug class worth a gate.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime", "taskName"}
_EXTRA = re.compile(r"extra=\{(?P<body>[^}]*)\}", re.S)
_KEY = re.compile(r"[\"']([a-zA-Z_][a-zA-Z0-9_]*)[\"']\s*:")


def _log_calls() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for folder in ("app", "scripts", "ui"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            src = path.read_text(encoding="utf-8")
            for match in _EXTRA.finditer(src):
                for key in _KEY.findall(match.group("body")):
                    out.append((f"{path.relative_to(ROOT)}:{src[: match.start()].count(chr(10)) + 1}", key))
    return out


def test_no_log_extra_dict_shadows_a_logrecord_attribute() -> None:
    collisions = sorted({where: key for where, key in _log_calls() if key in RESERVED})
    assert not collisions, f"reserved LogRecord keys used in `extra=`: {collisions}"


def test_log_call_count_is_sane() -> None:
    """Guards the scan itself: a broken regex would make the test above pass vacuously."""
    assert len(_log_calls()) > 20


async def test_bootstrap_keys_are_idempotent_and_log_at_info(caplog: pytest.LogCaptureFixture) -> None:
    """The documented cold start, exercised at the log level production actually uses."""
    from app.api.auth import bootstrap_keys_from_settings, clear_auth_cache
    from app.db.session import init_db

    await init_db()
    with caplog.at_level(logging.INFO, logger="app.api.auth"):
        first = await bootstrap_keys_from_settings()
        second = await bootstrap_keys_from_settings()
    assert first >= 1, "conftest pins bootstrap keys"
    assert second == 0, "the second boot must not re-create them"
    assert any("bootstrap api keys ensured" in record.message for record in caplog.records)


async def test_startup_logging_survives_every_level(caplog: pytest.LogCaptureFixture) -> None:
    """Boot the service at DEBUG/INFO/WARNING: no log call may be level-dependent in a way that raises."""
    import asyncio

    from app.main import create_app

    for level in ("DEBUG", "INFO", "WARNING"):
        app = create_app()
        async with app.router.lifespan_context(app):  # the real startup path, including key bootstrap
            assert app.state.pipeline is not None, level
            health = await app.state.pipeline.run("hello", request_id=f"boot-{level}")
            assert health.status.value in {"allowed", "rewritten", "blocked"}
