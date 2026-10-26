"""Shared pipeline vocabulary: context, typed stage results, traces, Stage interface.

The orchestrator only ever talks to :class:`Stage` objects through ``run()`` and the
:class:`PipelineContext`, which is what makes stages pluggable: adding "Stage 1.5:
brand-safety" is a new subclass plus one line in the chain, with no change to the
HTTP layer, the DB schema or the formatter.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover
    from app.core.config import Settings
    from app.pipeline.policy import PolicyBundle


class ModerationMode(str, Enum):
    """Threshold presets, tuned in ``config/policies.yaml`` (see ``modes:``)."""

    STRICT = "strict"
    BALANCED = "balanced"
    LENIENT = "lenient"

    @classmethod
    def coerce(cls, value: Any) -> ModerationMode:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.BALANCED


class Status(str, Enum):
    """Top-level outcome exposed to API consumers."""

    ALLOWED = "allowed"
    REWRITTEN = "rewritten"
    BLOCKED = "blocked"


class Decision(str, Enum):
    PASS = "pass"
    FLAG = "flag"
    BLOCK = "block"


class StageName(str, Enum):
    STAGE1 = "stage1_toxicity"
    STAGE2 = "stage2_policy"
    STAGE3 = "stage3_generation"
    POST = "post_processing"


class Violation(BaseModel):
    """One concrete policy finding, safe to serialize into logs and responses."""

    model_config = ConfigDict(extra="ignore")

    category: str
    severity: Literal["info", "low", "medium", "high", "critical"] = "medium"
    confidence: float = 0.0
    action: Literal["block", "rewrite", "mask", "allow", "flag", "disclaimer"] = "flag"
    policy_id: str | None = None
    stage: str = "stage1"
    detector: str = "heuristic"
    reason: str = ""
    excerpt: str | None = None  # short matched snippet, redacted at store time
    span: tuple[int, int] | None = None

    def as_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data["span"] = list(self.span) if self.span else None
        return data


class Scorecard(BaseModel):
    """Numeric detector outputs for one stage (auditable, plotted as histograms)."""

    model_config = ConfigDict(extra="ignore")

    toxicity: float = 0.0
    prompt_injection: float = 0.0
    jailbreak: float = 0.0
    nsfw: float = 0.0
    obfuscation: float = 0.0
    pii_risk: float = 0.0
    legal_risk: float = 0.0
    sensitive_topic: float = 0.0

    def top(self) -> tuple[str, float]:
        items = self.model_dump()
        return max(items.items(), key=lambda kv: kv[1]) if items else ("toxicity", 0.0)

    def as_dict(self) -> dict[str, float]:
        return {k: round(float(v), 4) for k, v in self.model_dump().items()}


class StageResult(BaseModel):
    """Base for every stage result: decision + scores + timing are mandatory."""

    model_config = ConfigDict(extra="ignore")

    stage: str
    decision: Decision = Decision.PASS
    latency_ms: float = 0.0
    confidence: float = 0.0
    scores: Scorecard = Field(default_factory=Scorecard)
    categories: list[str] = Field(default_factory=list)
    violations: list[Violation] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    engine: str = "n/a"
    degraded: bool = False
    error: str | None = None

    @property
    def blocked(self) -> bool:
        return self.decision is Decision.BLOCK

    def summary(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "decision": self.decision.value,
            "latency_ms": round(self.latency_ms, 2),
            "confidence": round(self.confidence, 4),
            "categories": self.categories,
            "scores": self.scores.as_dict(),
            "engine": self.engine,
            "degraded": self.degraded,
            "violation_count": len(self.violations),
            "error": self.error,
        }


class Stage1Result(StageResult):
    """Toxicity + prompt-injection verdict."""

    model_config = ConfigDict(extra="ignore")

    stage: str = StageName.STAGE1.value
    toxicity: float = 0.0
    prompt_injection: float = 0.0
    jailbreak: float = 0.0
    #: full Layer-A evidence: {heuristics: {...}, classifier: {...}}
    detector_detail: dict[str, Any] = Field(default_factory=dict)
    judge: dict[str, Any] | None = None
    judge_used: bool = False
    judge_skipped_reason: str | None = None
    threshold: float = 0.0
    heuristics: list[str] = Field(default_factory=list)
    llm_model: str | None = None
    llm_usage: dict[str, Any] | None = None


class Stage2Action(BaseModel):
    """One concrete transformation applied to the text (fully auditable diff entry)."""

    model_config = ConfigDict(extra="ignore")

    policy_id: str
    rule_name: str = ""
    action: Literal["mask", "rewrite", "disclaimer", "blocked_span_removed", "normalize"]
    original: str
    replacement: str
    start: int | None = None
    end: int | None = None
    confidence: float = 0.0
    reason: str = ""
    detector: str = "policy"


class Stage2Result(StageResult):
    """Content-policy verdict + rewritten text."""

    model_config = ConfigDict(extra="ignore")

    stage: str = StageName.STAGE2.value
    status: Literal["clean", "rewritten", "blocked"] = "clean"
    text: str = ""
    original_text: str = ""
    actions: list[Stage2Action] = Field(default_factory=list)
    diff: list[dict[str, Any]] = Field(default_factory=list)
    entities: list[dict[str, Any]] = Field(default_factory=list)
    disclaimers: list[str] = Field(default_factory=list)
    #: redacted-before-echo summary of every Stage-2 transformation (formatter input)
    transformations: list[dict[str, Any]] = Field(default_factory=list)
    rewrite_engine: str | None = None


class Stage3Result(StageResult):
    """Final answer + the output check that guarded it."""

    model_config = ConfigDict(extra="ignore")

    stage: str = StageName.STAGE3.value
    output: str = ""
    reasoning: str | None = None
    output_check_passed: bool = True
    output_check_violations: list[Violation] = Field(default_factory=list)
    provider: str = "n/a"
    model: str = "n/a"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    attempts: int = 1
    mode: str = "passthrough"
    output_sanitized: bool = False


class StageTrace(BaseModel):
    """What the trace viewer shows for a single stage."""

    model_config = ConfigDict(extra="ignore")

    name: str
    status: Literal["ok", "skipped", "short_circuited", "error", "timeout"] = "ok"
    decision: str = "pass"
    latency_ms: float = 0.0
    started_at: str = ""
    engine: str = ""
    input_preview: str = ""
    output_preview: str = ""
    scores: dict[str, float] = Field(default_factory=dict)
    categories: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class PipelineTrace(BaseModel):
    """Full chain evidence: the thing a reviewer actually reads in the UI."""

    model_config = ConfigDict(extra="ignore")

    request_id: str
    mode: str = "balanced"
    policy_version: str = ""
    stages: list[StageTrace] = Field(default_factory=list)
    short_circuited_at: str | None = None
    total_latency_ms: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    engines: dict[str, str] = Field(default_factory=dict)

    def add(self, trace: StageTrace) -> None:
        self.stages.append(trace)
        self.engines[trace.name] = trace.engine

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class PipelineContext(BaseModel):
    """Mutable per-request state handed to every stage.

    Carries both representations of the text (sanitized + detection view) so that
    spans found in the obfuscation-folded view remain valid on the returned text.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    request_id: str
    mode: ModerationMode = ModerationMode.BALANCED
    raw_text: str = ""
    text: str = ""
    view: str = ""
    key_id: str | None = None
    user_id_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    return_reasoning: bool = False
    script_mix: dict[str, float] = Field(default_factory=dict)
    obfuscation_score: float = 0.0
    obfuscation_flags: list[str] = Field(default_factory=list)
    payload_signals: dict[str, Any] = Field(default_factory=dict)
    started_at: float = Field(default_factory=time.perf_counter)
    started_wall: str = ""
    truncated: bool = False

    # populated as stages run
    stage1: Stage1Result | None = None
    stage2: Stage2Result | None = None
    stage3: Stage3Result | None = None
    trace: PipelineTrace | None = None

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000

    @property
    def text_for_next_stage(self) -> str:
        if self.stage2 is not None and self.stage2.text:
            return self.stage2.text
        return self.text


class PipelineResult(BaseModel):
    """The orchestrator's contract to the post-processing + formatter layers."""

    model_config = ConfigDict(extra="ignore")

    request_id: str
    status: Status = Status.ALLOWED
    decision: Decision = Decision.PASS
    safe_output: str = ""
    original_text: str = ""
    action_taken: Literal["none", "masked_pii", "rewrote_segments", "masked_pii_and_rewrote", "added_safety_annotations", "blocked"] = "none"
    violations: list[Violation] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""
    blocked_reason: str | None = None
    disclaimers: list[str] = Field(default_factory=list)
    #: redacted-before-echo summary of every Stage-2 transformation (formatter input)
    transformations: list[dict[str, Any]] = Field(default_factory=list)
    user_message: str | None = None
    mode: str = "balanced"
    policy_version: str = ""
    model_versions: dict[str, str] = Field(default_factory=dict)
    latency_ms: float = 0.0
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    trace: PipelineTrace | None = None
    error: str | None = None
    fail_closed: bool = False
    degraded: bool = False
    token_usage: dict[str, Any] = Field(default_factory=dict)
    estimated_cost_usd: float = 0.0
    created_at: str = ""

    @property
    def blocked(self) -> bool:
        return self.status is Status.BLOCKED


class StageTimeout(Exception):
    """Raised when a stage exceeds its configured budget."""

    def __init__(self, stage: str, budget: float) -> None:
        super().__init__(f"{stage} exceeded its {budget:.1f}s budget")
        self.stage = stage
        self.budget = budget


class Stage(ABC):
    """Common interface every pipeline stage implements.

    Subclasses provide ``name`` and ``async def _run(ctx)``; the base class supplies
    timing, the per-stage timeout, fail-closed error wrapping and trace building, so
    no stage has to re-implement any of that (and cannot forget to).
    """

    name: str = "stage"
    #: whether a failure of this stage must block the request (it always does; the
    #: flag documents intent for stages that could legitimately be optional)
    fail_closed: bool = True

    def __init__(self, settings: Settings | None = None, *, timeout_s: float | None = None) -> None:
        from app.core.config import get_settings

        self.settings = settings or get_settings()
        self.timeout_s = timeout_s if timeout_s is not None else self._default_timeout()

    def _default_timeout(self) -> float:  # pragma: no cover - overridden
        return self.settings.stage1_timeout_s

    @property
    def policy_bundle(self) -> PolicyBundle:
        from app.pipeline.policy import get_policy_bundle

        return get_policy_bundle(self.settings.policies_file)

    @abstractmethod
    async def _run(self, ctx: PipelineContext) -> StageResult:
        """Stage body. May assume ``ctx.text``/``ctx.view`` are already sanitized."""

    async def execute(self, ctx: PipelineContext) -> StageResult:
        """Run with timing + timeout + fail-closed wrapping. Never raises."""
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(self._run(ctx), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            result = self.on_timeout(ctx, StageTimeout(self.name, self.timeout_s))
        except Exception as exc:  # fail closed: a broken stage must not pass content
            result = self.on_error(ctx, exc)
        result.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return result

    # ------------------------------------------------------- failure behaviour
    def on_timeout(self, ctx: PipelineContext, exc: StageTimeout) -> StageResult:  # pragma: no cover - overridden
        return StageResult(
            stage=self.name,
            decision=Decision.BLOCK if self.fail_closed else Decision.FLAG,
            error=f"timeout after {exc.budget:.1f}s",
            reasons=[f"{self.name} timed out; request blocked fail-closed"],
            degraded=True,
        )

    def on_error(self, ctx: PipelineContext, exc: Exception) -> StageResult:  # pragma: no cover - overridden
        return StageResult(
            stage=self.name,
            decision=Decision.BLOCK if self.fail_closed else Decision.FLAG,
            error=f"{exc.__class__.__name__}: {exc}"[:400],
            reasons=[f"{self.name} failed; request blocked fail-closed"],
            degraded=True,
        )

    def trace_from(self, result: StageResult, ctx: PipelineContext, *, status: str = "ok", detail: dict[str, Any] | None = None) -> StageTrace:
        text_in = ctx.text
        text_out = getattr(result, "text", None) or getattr(result, "output", None) or ""
        limit = self.settings.trace_preview_chars
        return StageTrace(
            name=self.name,
            status=status,
            decision=result.decision.value,
            latency_ms=result.latency_ms,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            engine=result.engine,
            input_preview=text_in[:limit],
            output_preview=(text_out or "")[:limit],
            scores=result.scores.as_dict(),
            categories=result.categories,
            confidence=result.confidence,
            detail=detail or result.summary(),
            error=result.error,
        )
