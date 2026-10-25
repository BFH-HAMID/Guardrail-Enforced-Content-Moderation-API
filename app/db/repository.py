"""Data access for API keys, moderation logs and stats aggregation.

All queries are async and dialect-portable (SQL for the aggregates; percentiles are
computed from a bounded recent window because ``PERCENTILE_CONT`` does not exist on
SQLite). Read paths never raise into the request path - the routes map exceptions to
503s - and the write path is a single transaction that inserts the log row plus its
normalised violation rows.
"""

from __future__ import annotations

import datetime as dt
import statistics
from typing import Any, Sequence

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.exc import IntegrityError

from app.core.logging import get_logger
from app.db.models import ApiKey, ModerationLog, ModerationViolationRow, new_uuid
from app.db.session import session_scope

log = get_logger(__name__)


# ------------------------------------------------------------------- api keys


async def upsert_api_key(
    *,
    key_id: str,
    key_hash: str,
    label_hint: str = "",
    name: str = "unnamed",
    scopes: Sequence[str] = ("moderate", "read_logs"),
    rate_limit_rpm: int | None = None,
    owner_email: str = "",
) -> dict[str, Any]:
    """Create a key row, or re-activate an existing one (idempotent bootstrap)."""
    async with session_scope() as session:
        existing = (await session.execute(select(ApiKey).where(ApiKey.key_hash == key_hash))).scalar_one_or_none()
        if existing is not None:
            if not existing.is_active:
                existing.is_active = True
                existing.revoked_at = None
            existing.scopes = list(scopes)
            existing.rate_limit_rpm = rate_limit_rpm
            return {"created": False, "key_id": existing.key_id, "name": existing.name}
        row = ApiKey(
            id=new_uuid(),
            key_id=key_id,
            key_hash=key_hash,
            label_hint=label_hint,
            name=name,
            owner_email=owner_email,
            scopes=list(scopes),
            rate_limit_rpm=rate_limit_rpm,
            is_active=True,
        )
        session.add(row)
        return {"created": True, "key_id": key_id, "name": name}


async def find_api_key_by_hash(key_hash: str) -> dict[str, Any] | None:
    async with session_scope() as session:
        row = (await session.execute(select(ApiKey).where(ApiKey.key_hash == key_hash))).scalar_one_or_none()
        if row is None:
            return None
        return {
            "key_id": row.key_id,
            "name": row.name,
            "is_active": bool(row.is_active),
            "scopes": list(row.scopes or []),
            "rate_limit_rpm": row.rate_limit_rpm,
            "request_count": row.request_count,
        }


async def touch_api_key(key_id: str) -> None:
    """Best-effort usage bookkeeping; never blocks or fails the request path."""
    try:
        async with session_scope() as session:
            await session.execute(
                update(ApiKey)
                .where(ApiKey.key_id == key_id)
                .values(last_used_at=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None), request_count=ApiKey.request_count + 1)
            )
    except Exception as exc:  # pragma: no cover - analytics must not break serving
        log.debug("api key touch failed", extra={"error": str(exc)[:120]})


async def revoke_api_key(key_id: str) -> bool:
    async with session_scope() as session:
        row = (await session.execute(select(ApiKey).where(ApiKey.key_id == key_id))).scalar_one_or_none()
        if row is None:
            return False
        row.is_active = False
        row.revoked_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        return True


async def list_api_keys() -> list[dict[str, Any]]:
    async with session_scope() as session:
        rows = (await session.execute(select(ApiKey).order_by(ApiKey.created_at.desc()).limit(200))).scalars().all()
        return [row.to_public_dict() for row in rows]


# ----------------------------------------------------------------------- logs


async def insert_moderation_log(payload: dict[str, Any]) -> str | None:
    """Persist one decision. Returns the row id, or ``None`` when the write failed.

    The caller (post-processing) treats ``None`` as "recorded in metrics, not in SQL";
    it must never escalate into the client response, because failing to *log* is not a
    safety failure - failing to *moderate* is.
    """
    violations = payload.pop("violations", []) or []
    row = ModerationLog(
        id=payload.get("id") or new_uuid(),
        **{k: v for k, v in payload.items() if k in ModerationLog.__table__.columns.keys() and k != "id"},
    )
    for violation in violations:
        row.violations.append(
            ModerationViolationRow(
                category=str(violation.get("category", "other"))[:48],
                severity=str(violation.get("severity", "medium"))[:16],
                action=str(violation.get("action", "flag"))[:16],
                policy_id=(str(violation["policy_id"])[:64] if violation.get("policy_id") else None),
                stage=str(violation.get("stage", "stage1"))[:24],
                detector=(str(violation["detector"])[:48] if violation.get("detector") else None),
                confidence=float(violation.get("confidence") or 0.0),
                reason=(str(violation.get("reason") or ""))[:1000] or None,
                excerpt=(str(violation.get("excerpt"))[:500] if violation.get("excerpt") else None),
            )
        )
    try:
        async with session_scope() as session:
            session.add(row)
            await session.flush()
            return row.id
    except IntegrityError as exc:
        # Duplicate request_id (client retried): keep the first record, log the collision.
        log.info("duplicate request_id ignored", extra={"request_id": payload.get("request_id"), "error": str(exc.orig)[:120]})
        return None
    except Exception as exc:
        log.error("moderation log write failed", extra={"request_id": payload.get("request_id"), "error": f"{exc.__class__.__name__}: {exc}"[:240]})
        raise


async def get_log_by_request_id(request_id: str) -> dict[str, Any] | None:
    async with session_scope() as session:
        row = (await session.execute(select(ModerationLog).where(ModerationLog.request_id == request_id))).scalar_one_or_none()
        return row.to_dict(include_trace=True) if row else None


async def list_logs(
    *,
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
    category: str | None = None,
    policy_id: str | None = None,
    mode: str | None = None,
    key_id: str | None = None,
    since: dt.datetime | None = None,
    until: dt.datetime | None = None,
    min_confidence: float | None = None,
    search: str | None = None,
    order: str = "desc",
) -> dict[str, Any]:
    """Paginated log listing with the filters the Streamlit viewer exposes."""
    conditions: list[Any] = []
    if status:
        conditions.append(ModerationLog.status == status)
    if mode:
        conditions.append(ModerationLog.mode == mode)
    if key_id:
        conditions.append(ModerationLog.api_key_id == key_id)
    if min_confidence is not None:
        conditions.append(ModerationLog.confidence >= min_confidence)
    if since is not None:
        conditions.append(ModerationLog.created_at >= since)
    if until is not None:
        conditions.append(ModerationLog.created_at <= until)
    if search:
        pattern = f"%{search.strip()[:80]}%"
        conditions.append(
            or_(
                ModerationLog.request_id.ilike(pattern),
                ModerationLog.input_text.ilike(pattern),
                ModerationLog.reason.ilike(pattern),
            )
        )
    join_violations = bool(category or policy_id)
    if category:
        conditions.append(ModerationViolationRow.category == category)
    if policy_id:
        conditions.append(ModerationViolationRow.policy_id == policy_id)

    order_by = ModerationLog.created_at.desc() if order.lower().startswith("d") else ModerationLog.created_at.asc()
    async with session_scope() as session:
        base = select(ModerationLog).distinct()
        if join_violations:
            base = base.join(ModerationViolationRow, ModerationViolationRow.log_id == ModerationLog.id)
        if conditions:
            base = base.where(and_(*conditions))
        rows = (await session.execute(base.order_by(order_by).limit(max(1, min(limit, 500))).offset(max(0, offset)))).scalars().all()

        count_stmt = select(func.count(func.distinct(ModerationLog.id)))
        if join_violations:
            count_stmt = count_stmt.select_from(ModerationLog).join(ModerationViolationRow, ModerationViolationRow.log_id == ModerationLog.id)
        if conditions:
            count_stmt = count_stmt.where(and_(*conditions)) if join_violations is False else count_stmt.where(and_(*conditions))
        total = int((await session.execute(count_stmt)).scalar() or 0)

    return {
        "items": [row.to_dict() for row in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": total > offset + len(rows),
    }


async def get_log_trace(request_id: str) -> dict[str, Any] | None:
    async with session_scope() as session:
        row = (await session.execute(select(ModerationLog).where(ModerationLog.request_id == request_id))).scalar_one_or_none()
        if row is None:
            return None
        return {"request_id": row.request_id, "status": row.status, "trace": row.trace_json, "violations": [v.to_dict() for v in row.violations]}


# ---------------------------------------------------------------------- stats


async def compute_stats(*, window_days: int = 7, key_id: str | None = None, percentile_sample: int = 1000) -> dict[str, Any]:
    """Aggregate dashboard numbers straight from SQL + a bounded latency sample."""
    since = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(days=max(1, window_days))
    conditions = [ModerationLog.created_at >= since]
    if key_id:
        conditions.append(ModerationLog.api_key_id == key_id)

    async with session_scope() as session:
        by_status = dict(
            (
                await session.execute(
                    select(ModerationLog.status, func.count()).where(and_(*conditions)).group_by(ModerationLog.status)
                )
            ).all()
        )
        by_mode = dict((await session.execute(select(ModerationLog.mode, func.count()).where(and_(*conditions)).group_by(ModerationLog.mode))).all())
        # one aggregate row: count / avg+max latency / cost / tokens / avg confidence
        totals = (
            await session.execute(
                select(
                    func.count(ModerationLog.id),
                    func.avg(ModerationLog.latency_ms),
                    func.max(ModerationLog.latency_ms),
                    func.sum(ModerationLog.estimated_cost_usd),
                    func.sum(ModerationLog.prompt_tokens + ModerationLog.completion_tokens),
                    func.avg(ModerationLog.confidence),
                ).where(and_(*conditions))
            )
        ).one()
        category_rows = (
            await session.execute(
                select(ModerationViolationRow.category, func.count(), func.avg(ModerationViolationRow.confidence))
                .select_from(ModerationViolationRow)
                .join(ModerationLog, ModerationLog.id == ModerationViolationRow.log_id)
                .where(ModerationLog.created_at >= since)
                .group_by(ModerationViolationRow.category)
                .order_by(func.count().desc())
                .limit(20)
            )
        ).all()
        policy_rows = (
            await session.execute(
                select(ModerationViolationRow.policy_id, func.count())
                .select_from(ModerationViolationRow)
                .join(ModerationLog, ModerationLog.id == ModerationViolationRow.log_id)
                .where(ModerationLog.created_at >= since)
                .group_by(ModerationViolationRow.policy_id)
                .order_by(func.count().desc())
                .limit(15)
            )
        ).all()
        action_rows = dict((await session.execute(select(ModerationLog.action_taken, func.count()).where(and_(*conditions)).group_by(ModerationLog.action_taken))).all())
        latency_sample = [
            float(value)
            for value in (
                await session.execute(
                    select(ModerationLog.latency_ms).where(and_(*conditions)).order_by(ModerationLog.created_at.desc()).limit(percentile_sample)
                )
            ).scalars().all()
            if value is not None
        ]
        stage_rows = (
            await session.execute(
                select(ModerationLog.stage_latency, ModerationLog.latency_ms).where(and_(*conditions)).order_by(ModerationLog.created_at.desc()).limit(percentile_sample)
            )
        ).all()
        fail_closed_count = int((await session.execute(select(func.count()).select_from(ModerationLog).where(and_(*conditions), ModerationLog.fail_closed.is_(True)))).scalar() or 0)
        degraded_count = int((await session.execute(select(func.count()).select_from(ModerationLog).where(and_(*conditions), ModerationLog.degraded.is_(True)))).scalar() or 0)
        errors = int((await session.execute(select(func.count()).select_from(ModerationLog).where(and_(*conditions), ModerationLog.error.is_not(None)))).scalar() or 0)

    total = int(totals[0] or 0)
    blocked = int(by_status.get("blocked", 0))
    p50, p95, p99 = _percentiles(latency_sample)
    stage_latency = _stage_averages([row[0] for row in stage_rows])
    return {
        "window_days": window_days,
        "since": since.isoformat() + "Z",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total_requests": total,
        "status_counts": {status: int(count) for status, count in by_status.items()},
        "block_rate": round(blocked / total, 4) if total else 0.0,
        "mode_counts": {mode or "balanced": int(count) for mode, count in by_mode.items()},
        "action_counts": {action or "none": int(count) for action, count in action_rows.items()},
        "violations_by_category": [
            {"category": category, "count": int(count), "avg_confidence": round(float(avg or 0.0), 4)} for category, count, avg in category_rows
        ],
        "violations_by_policy": [{"policy_id": policy, "count": int(count)} for policy, count in policy_rows],
        "latency_ms": {
            "avg": round(float(totals[1] or 0.0), 2),
            "max": round(float(totals[2] or 0.0), 2),
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "samples": len(latency_sample),
        },
        "avg_confidence": round(float(totals[5] or 0.0), 4),
        "stage_latency_ms": stage_latency,
        "cost": {"llm_tokens": int(totals[4] or 0), "estimated_cost_usd": round(float(totals[3] or 0.0), 6)},
        "reliability": {
            "fail_closed": fail_closed_count,
            "degraded": degraded_count,
            "with_error": errors,
            "error_rate": round(errors / total, 4) if total else 0.0,
        },
    }


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    ordered = sorted(values)

    def pct(share: float) -> float:
        if len(ordered) == 1:
            return round(ordered[0], 2)
        index = min(len(ordered) - 1, max(0, int(round(share * (len(ordered) - 1)))))
        return round(ordered[index], 2)

    return pct(0.50), pct(0.95), pct(0.99)


def _stage_averages(payloads: Sequence[Any]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for stage, value in payload.items():
            try:
                buckets.setdefault(str(stage), []).append(float(value))
            except (TypeError, ValueError):
                continue
    return {stage: round(statistics.fmean(values), 2) for stage, values in buckets.items() if values}
