"""SQLAlchemy models.

Two tables:

``api_keys``      hashed credentials + per-key limits (no plaintext key, ever)
``moderation_logs`` one row per ``POST /moderate`` - the audit record
``moderation_violations`` normalised child rows so "violations by category" is a real
                   ``GROUP BY`` instead of scanning JSON blobs in Python.

Deliberate denormalisation: ``status``/``mode``/``confidence``/latency live on the log
row so the log viewer and /stats never need to parse ``trace_json``.

Text storage is policy-driven (see :mod:`app.post.postprocess`): ``input_text`` may be
truncated, withheld (hash only) or Fernet-encrypted, and a row can be expired by the
retention job. Works on SQLite and PostgreSQL without dialect-specific column types -
JSON, not JSONB, and string UUID primary keys - so the same migration runs everywhere.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def new_uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Declarative base shared by the app and the tests."""


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    key_id: Mapped[str] = mapped_column(String(12), unique=True, index=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    label_hint: Mapped[str] = mapped_column(String(24), default="")  # "gmk_ab12…" only
    name: Mapped[str] = mapped_column(String(120), default="unnamed")
    owner_email: Mapped[str] = mapped_column(String(200), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")
    scopes: Mapped[list[str]] = mapped_column(JSON, default=list)
    rate_limit_rpm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow, server_default=func.now())
    last_used_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    revoked_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    request_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    #: cascade is disabled on delete: revoking a key must not destroy its audit trail
    logs: Mapped[list[ModerationLog]] = relationship(
        "ModerationLog",
        foreign_keys="ModerationLog.api_key_id",
        primaryjoin="ApiKey.key_id == foreign(ModerationLog.api_key_id)",
        viewonly=True,
    )

    __table_args__ = (Index("ix_api_keys_active_created", "is_active", "created_at"),)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "key_id": self.key_id,
            "label_hint": self.label_hint,
            "name": self.name,
            "is_active": self.is_active,
            "scopes": self.scopes or [],
            "rate_limit_rpm": self.rate_limit_rpm,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "request_count": self.request_count,
        }


class ModerationLog(Base):
    __tablename__ = "moderation_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    request_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    api_key_id: Mapped[str | None] = mapped_column(ForeignKey("api_keys.key_id", ondelete="SET NULL"), nullable=True, index=True)
    user_id_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow, index=True, server_default=func.now())
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    stage_latency: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)

    status: Mapped[str] = mapped_column(String(16), index=True)
    action_taken: Mapped[str] = mapped_column(String(32), default="none")
    reason: Mapped[str | None] = mapped_column(String(120), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    mode: Mapped[str] = mapped_column(String(16), default="balanced", index=True)
    channel: Mapped[str | None] = mapped_column(String(48), nullable=True)
    language: Mapped[str | None] = mapped_column(String(32), nullable=True)

    input_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_text_mode: Mapped[str] = mapped_column(String(16), default="full")  # full|truncated|redacted|encrypted
    input_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    input_chars: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    output_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    scores: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    trace_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    model_versions: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    policy_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")

    degraded: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    fail_closed: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    obfuscation_score: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")

    retention_expires_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True, index=True)

    violations: Mapped[list[ModerationViolationRow]] = relationship(
        back_populates="log", cascade="all, delete-orphan", passive_deletes=True, lazy="selectin"
    )

    __table_args__ = (
        Index("ix_logs_status_created", "status", "created_at"),
        Index("ix_logs_key_created", "api_key_id", "created_at"),
    )

    def to_dict(self, *, include_trace: bool = False) -> dict[str, Any]:
        """Serialisation used by ``GET /logs`` (input text only when we stored it)."""
        payload = {
            "id": self.id,
            "request_id": self.request_id,
            "created_at": self.created_at.isoformat() + "Z" if self.created_at else None,
            "status": self.status,
            "action_taken": self.action_taken,
            "reason": self.reason,
            "confidence": round(self.confidence or 0.0, 4),
            "mode": self.mode,
            "latency_ms": round(self.latency_ms or 0.0, 2),
            "stage_latency_ms": self.stage_latency or {},
            "categories": sorted({v.category for v in self.violations}),
            "violation_count": len(self.violations or []),
            "input_preview": (self.input_text[:280] + "…") if self.input_text and len(self.input_text) > 280 else self.input_text,
            "input_chars": self.input_chars,
            "input_text_mode": self.input_text_mode,
            "input_hash": self.input_hash,
            "output_preview": (self.output_text[:280] + "…") if self.output_text and len(self.output_text) > 280 else self.output_text,
            "scores": self.scores or {},
            "policy_version": self.policy_version,
            "model_versions": self.model_versions or {},
            "channel": self.channel,
            "language": self.language,
            "user_id_hash": self.user_id_hash,
            "api_key_id": self.api_key_id,
            "tokens": {"prompt": self.prompt_tokens, "completion": self.completion_tokens},
            "estimated_cost_usd": self.estimated_cost_usd,
            "degraded": bool(self.degraded),
            "fail_closed": bool(self.fail_closed),
            "error": self.error,
            "obfuscation_score": self.obfuscation_score,
        }
        if include_trace:
            payload["trace"] = self.trace_json
            payload["violations"] = [v.to_dict() for v in self.violations or []]
        return payload


class ModerationViolationRow(Base):
    """One policy hit per log row, normalised for GROUP BY analytics."""

    __tablename__ = "moderation_violations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    log_id: Mapped[str] = mapped_column(ForeignKey("moderation_logs.id", ondelete="CASCADE"), index=True)
    category: Mapped[str] = mapped_column(String(48), index=True)
    severity: Mapped[str] = mapped_column(String(16), default="medium")
    action: Mapped[str] = mapped_column(String(16), default="flag")
    policy_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    stage: Mapped[str] = mapped_column(String(24), default="stage1")
    detector: Mapped[str | None] = mapped_column(String(48), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    log: Mapped[ModerationLog] = relationship(back_populates="violations")

    __table_args__ = (Index("ix_violations_category_created", "category", "created_at"),)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "action": self.action,
            "policy_id": self.policy_id,
            "stage": self.stage,
            "detector": self.detector,
            "confidence": round(self.confidence or 0.0, 4),
            "reason": self.reason,
            "excerpt": self.excerpt,
        }
