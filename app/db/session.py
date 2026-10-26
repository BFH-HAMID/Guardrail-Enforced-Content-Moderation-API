"""Async DB engine + session lifecycle, retention sweep, and a SQLite bootstrap.

One engine per process, created lazily so importing the app never touches the disk.
SQLite gets ``check_same_thread=False`` + WAL-ish pragmas (a single-container demo must
not fall over on concurrent writes); Postgres gets a real pool. The parent directory of
a file-based SQLite URL is created automatically because ``./data/`` is not in the repo.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import event, func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.models import Base, ModerationLog

log = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_engine_url: str | None = None


def _sqlite_path(url: str) -> Path | None:
    marker = "///"
    if "sqlite" not in url or marker not in url:
        return None
    raw = url.split(marker, 1)[1]
    if raw.startswith(":memory:"):
        return None
    return Path(raw)


def build_engine(settings: Settings | None = None) -> AsyncEngine:
    settings = settings or get_settings()
    url = settings.database_url
    kwargs: dict[str, Any] = {"echo": settings.db_echo, "future": True}
    if "sqlite" in url:
        target = _sqlite_path(url)
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
    else:
        kwargs.update(pool_size=settings.db_pool_size, max_overflow=settings.db_max_overflow, pool_pre_ping=True)
    engine = create_async_engine(url, **kwargs)

    if "sqlite" in url:

        @event.listens_for(engine.sync_engine, "connect")
        def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:  # pragma: no cover - driver level
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=5000")
            finally:
                cursor.close()

    return engine


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    global _engine, _session_factory, _engine_url
    settings = settings or get_settings()
    if _engine is None or _engine_url != settings.database_url:
        dispose_engine_sync()
        _engine_url = settings.database_url
        _engine = build_engine(settings)
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    get_engine()
    assert _session_factory is not None
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """``async with session_scope() as session:`` with commit/rollback handled."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency wrapper (yields a session, commits at the end of the request)."""
    async with session_scope() as session:
        yield session


async def init_db(*, create_all: bool | None = None) -> None:
    """Create tables on demand. Production should run migrations instead (see README)."""
    settings = get_settings()
    if create_all is None:
        create_all = settings.db_create_all_on_startup
    if not create_all:
        return
    engine = get_engine(settings)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    log.info("database ready", extra={"url": _safe_url(settings.database_url), "tables": sorted(Base.metadata.tables)})


def _safe_url(url: str) -> str:
    if "@" in url:
        scheme, _, rest = url.partition("://")
        credentials, _, host = rest.rpartition("@")
        return f"{scheme}://{credentials.split(':')[0]}:***@{host}"
    return url


async def dispose_engine() -> None:
    global _engine, _session_factory, _engine_url
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
    _engine_url = None


def dispose_engine_sync() -> None:  # pragma: no cover - teardown helper
    global _engine, _session_factory, _engine_url
    _engine = None
    _session_factory = None
    _engine_url = None


# ------------------------------------------------------------------- retention


def retention_cutoff(now: dt.datetime | None = None) -> dt.datetime | None:
    settings = get_settings()
    if settings.retention_days <= 0:
        return None  # 0 disables the sweep entirely (logs kept until deleted by hand)
    return (now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)) - dt.timedelta(days=settings.retention_days)


async def purge_expired_logs(now: dt.datetime | None = None) -> int:
    """Delete moderation logs past their retention window. Returns the row count.

    ``moderation_violations`` rows disappear via ``ON DELETE CASCADE``; on SQLite we
    delete the children explicitly because ``PRAGMA foreign_keys`` is per-connection.
    """
    cutoff = retention_cutoff(now)
    if cutoff is None:
        return 0
    async with session_scope() as session:
        ids = (await session.execute(select(ModerationLog.id).where(ModerationLog.created_at < cutoff).limit(5000))).scalars().all()
        if not ids:
            return 0
        await session.execute(text("DELETE FROM moderation_violations WHERE log_id IN :ids").bindparams(ids=tuple(ids)))
        await session.execute(ModerationLog.__table__.delete().where(ModerationLog.id.in_(ids)))
        deleted = len(ids)
    log.info("retention sweep removed expired logs", extra={"count": deleted, "cutoff": cutoff.isoformat()})
    return deleted


async def retention_loop(should_stop: Any, *, interval_s: float) -> None:
    """Background sweep: hourly by default, exits when ``should_stop()`` is true."""
    while not should_stop():
        try:
            await _sleep_or_stop(should_stop, interval_s)
            if should_stop():
                return
            await purge_expired_logs()
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            return
        except Exception as exc:  # pragma: no cover - resilience guard
            log.warning("retention sweep failed", extra={"error": str(exc)[:200]})
            await _sleep_or_stop(should_stop, min(60.0, interval_s))


async def _sleep_or_stop(should_stop: Any, seconds: float) -> None:
    steps = max(1, int(seconds // 5))
    for _ in range(steps):
        if should_stop():
            return
        await asyncio.sleep(min(5.0, seconds / steps))


async def db_readiness() -> dict[str, Any]:
    """Used by ``GET /health``: engine + dialect + row counts + oldest log."""
    settings = get_settings()
    info: dict[str, Any] = {"url": _safe_url(settings.database_url), "ok": False}
    try:
        engine = get_engine(settings)
        async with engine.connect() as connection:
            result = await connection.execute(text("SELECT 1"))
            info["probe"] = result.scalar() == 1
            try:
                info["dialect"] = engine.dialect.name
                counts = await connection.execute(
                    select(func.count()).select_from(ModerationLog.__table__)
                )
                info["log_rows"] = int(counts.scalar() or 0)
                oldest = await connection.execute(select(func.min(ModerationLog.created_at)))
                first = oldest.scalar()
                info["oldest_log"] = first.isoformat() + "Z" if first else None
            except Exception as exc:  # pragma: no cover - partial schema
                info["schema_warning"] = str(exc)[:160]
        info["ok"] = bool(info.get("probe"))
    except Exception as exc:
        info["error"] = f"{exc.__class__.__name__}: {exc}"[:200]
    return info
