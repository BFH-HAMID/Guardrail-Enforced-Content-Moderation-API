"""Shared fixtures.

Environment is pinned *before* importing the app so tests never inherit a developer's
``.env``: rules-only Stage 1 (no torch), ``local`` LLM provider, a throwaway SQLite
file, short timeouts, and bootstrap API keys. Every test gets an empty database, so
assertions about counts are exact rather than relative.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="guard-tests-"))
_BOOTSTRAP_KEY = "gmk_testkeyaaaaaaaaaaaaaaaaaaaaaaaaaa1"
_SECOND_KEY = "gmk_testkeybbbbbbbbbbbbbbbbbbbbbbbbbbbb2"

os.environ.update(
    {
        "GUARD_ENV_FILE": str(_TMP / "does-not-exist.env"),
        "GUARD_DATABASE_URL": f"sqlite+aiosqlite:///{_TMP / 'test.db'}",
        "GUARD_AUTH_ENABLED": "true",
        "GUARD_API_KEY_PEPPER": "test-pepper",
        "GUARD_BOOTSTRAP_API_KEYS": f"{_BOOTSTRAP_KEY},{_SECOND_KEY}",
        "GUARD_RATE_LIMIT_REQUESTS_PER_MINUTE": "5000",
        "GUARD_RATE_LIMIT_BURST": "5000",
        "GUARD_LOG_JSON": "false",
        "GUARD_LOG_LEVEL": "WARNING",
        "GUARD_LLM_PROVIDER": "local",
        "GUARD_STAGE1_USE_CLASSIFIER": "false",
        "GUARD_STAGE1_USE_LLM_JUDGE": "true",
        "GUARD_STAGE2_USE_PRESIDIO": "false",
        "GUARD_RETENTION_DAYS": "7",
        "GUARD_STORE_TRACE": "true",
        "GUARD_MAX_INPUT_CHARS": "8000",
        "GUARD_USER_ID_HASH_SALT": "test-salt",
    }
)

from app.core.config import reload_settings  # noqa: E402

reload_settings()


@pytest.fixture(scope="session", autouse=True)
def _cleanup_tmpdir() -> Iterator[None]:
    yield
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture(autouse=True)
def clean_database() -> Iterator[None]:
    """Empty tables before each test, so count assertions stay exact."""
    import sqlite3

    db_path = _TMP / "test.db"
    if not db_path.exists():
        yield
        return
    connection = sqlite3.connect(db_path)
    try:
        for table in ("moderation_violations", "moderation_logs", "api_keys"):
            try:
                connection.execute(f"DELETE FROM {table}")
            except sqlite3.OperationalError:
                pass  # tables not created yet (first test of the session)
        connection.commit()
    finally:
        connection.close()
    yield


@pytest.fixture(scope="session")
def bootstrap_key() -> str:
    return _BOOTSTRAP_KEY


@pytest.fixture(scope="session")
def second_key() -> str:
    return _SECOND_KEY


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def prepared_app() -> AsyncIterator[object]:
    """App with schema created and bootstrap keys seeded (no lifespan needed)."""
    from app.api.auth import bootstrap_keys_from_settings, clear_auth_cache
    from app.db.session import init_db
    from app.main import create_app
    from app.pipeline.orchestrator import ModerationPipeline
    from app.core.config import get_settings

    await init_db()
    await bootstrap_keys_from_settings()
    clear_auth_cache()
    settings = get_settings()
    app = create_app(settings=settings, inject_pipeline=ModerationPipeline(settings))
    yield app


@pytest.fixture
async def client(prepared_app: object) -> AsyncIterator[Any]:  # type: ignore[name-defined]  # noqa: F821
    import httpx

    from app.core.config import get_settings

    transport = httpx.ASGITransport(app=prepared_app)  # type: ignore[attr-defined]
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=30.0) as http_client:
        yield http_client


@pytest.fixture
async def anon_client(prepared_app: object) -> AsyncIterator[Any]:  # type: ignore[name-defined]  # noqa: F821
    """Client without an API key header."""
    import httpx

    transport = httpx.ASGITransport(app=prepared_app)  # type: ignore[attr-defined]
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=30.0) as http_client:
        yield http_client


@pytest.fixture
def auth_headers(bootstrap_key: str) -> dict[str, str]:
    return {"X-API-Key": bootstrap_key}
