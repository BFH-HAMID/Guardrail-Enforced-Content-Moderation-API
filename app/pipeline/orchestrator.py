"""Orchestrator - the Model Chain Prompting runner.

``Stage 1 -> Stage 2 -> Stage 3`` with short-circuiting, per-stage timeouts, a global
budget, a pluggable :class:`app.pipeline.base.Stage` interface and a full
:class:`~app.pipeline.base.PipelineTrace`.

Chain semantics (the parts that matter for safety):

* **Short-circuit on Stage 1 block.** The harmful prompt never reaches the policy
  rewriter or the generation model, so neither can be steered by it.
* **Stage 2 block** stops generation for the same reason (e.g. doxxing cluster,
  residual illegality after masking).
* **Every failure path blocks.** A timeout, an exception, an unparsable judge or a
  missing DB write all end in ``status=blocked`` with ``reason=moderation_error``.
  "We couldn't check it" is never treated as "it was fine".
* **Nothing is returned unmoderated.** The response text always comes from Stage 2's
  sanitized output or Stage 3's checked output - never from the raw request.

Stages are a list, so a new stage is one line here plus a subclass.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.text import is_multilingual, sanitize_text
from app.pipeline.base import (
    Decision,
    ModerationMode,
    PipelineContext,
    PipelineResult,
    PipelineTrace,
    Stage,
    StageTrace,
    Status,
    Violation,
)
from app.pipeline.stage1_toxicity import Stage1ToxicityStage
from app.pipeline.stage2_policy import Stage2PolicyStage
from app.pipeline.stage3_generation import SAFE_FALLBACKS, Stage3GenerationStage

log = get_logger(__name__)


#: Hard cap on the detection view (original text + folded-in decoded payloads).
_MAX_DETECTION_VIEW_CHARS = 12_000


class ModerationPipeline:
    """Sequential chain runner with trace assembly and fail-closed guarantees."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        stages: Sequence[Stage] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.stages: list[Stage] = list(stages) if stages is not None else self.default_stages(self.settings)

    # ------------------------------------------------------------------ factory
    @staticmethod
    def default_stages(settings: Settings) -> list[Stage]:
        """The 3-stage MCP chain. Replace/extend here to add a stage."""
        return [
            Stage1ToxicityStage(settings),
            Stage2PolicyStage(settings),
            Stage3GenerationStage(settings),
        ]

    # ---------------------------------------------------------------------- run
    async def run(
        self,
        text: str,
        *,
        request_id: str,
        mode: ModerationMode | str = ModerationMode.BALANCED,
        metadata: dict[str, Any] | None = None,
        return_reasoning: bool = False,
        key_id: str | None = None,
        user_id_hash: str | None = None,
    ) -> PipelineResult:
        mode = ModerationMode.coerce(mode)
        started_wall = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        bundle = self.policy_bundle()
        trace = PipelineTrace(
            request_id=request_id,
            mode=mode.value,
            policy_version=f"{bundle.version}+{bundle.checksum[:8]}",
            started_at=started_wall,
        )
        ctx = self.build_context(
            text,
            request_id=request_id,
            mode=mode,
            metadata=metadata,
            return_reasoning=return_reasoning,
            key_id=key_id,
            user_id_hash=user_id_hash,
            trace=trace,
        )

        log.info(
            "pipeline start",
            extra={"request_id": request_id, "mode": mode.value, "chars": ctx.text.__len__(), "stages": [s.name for s in self.stages]},
        )
        try:
            remaining = self.settings.pipeline_request_timeout_s - (time.perf_counter() - ctx.started_at)
            if remaining <= 0:
                return self._fail_closed(ctx, "global request timeout elapsed before the chain started")
            return await self._run_chain(ctx, list(self.stages))
        except Exception as exc:  # pragma: no cover - defensive: chain-level catch-all
            log.exception("pipeline crashed", extra={"request_id": request_id})
            return self._fail_closed(ctx, f"{exc.__class__.__name__}: {exc}"[:300])

    async def _run_chain(self, ctx: PipelineContext, stages: list[Stage]) -> PipelineResult:
        bundle = self.policy_bundle()
        for stage in stages:
            budget_left = self.settings.pipeline_request_timeout_s - ctx.elapsed_ms() / 1000.0
            if budget_left <= 0:
                ctx.trace and ctx.trace.add(
                    StageTrace(name=stage.name, status="short_circuited", decision="skipped", detail={"reason": "global timeout before stage"})
                )
                return self._fail_closed(ctx, "global request timeout while moderating", trace_only_names=[s.name for s in stages])
            stage.timeout_s = min(stage.timeout_s, max(1.0, budget_left))
            result = await stage.execute(ctx)
            self._attach(ctx, stage.name, result)

            if ctx.trace is not None:
                ctx.trace.add(
                    stage.trace_from(
                        result,
                        ctx,
                        status="error" if result.error else ("timeout" if "timeout" in (result.error or "") else "ok"),
                        detail={**result.summary(), "reasons": result.reasons[:6]},
                    )
                )

            if result.decision is Decision.BLOCK:
                if ctx.trace is not None:
                    ctx.trace.short_circuited_at = stage.name
                    for skipped in _remaining_stages(stages, stage.name):
                        ctx.trace.add(
                            StageTrace(
                                name=skipped,
                                status="short_circuited",
                                decision="skipped",
                                detail={"reason": f"not executed: {stage.name} blocked this request"},
                            )
                        )
                log.info(
                    "pipeline short-circuited",
                    extra={"request_id": ctx.request_id, "stage": stage.name, "categories": result.categories[:6]},
                )
                return self._blocked_result(ctx, stage.name, result)

        return self._allowed_result(ctx, bundle)

    # ------------------------------------------------------------------ context
    def build_context(
        self,
        text: str,
        *,
        request_id: str,
        mode: ModerationMode,
        metadata: dict[str, Any] | None,
        return_reasoning: bool,
        key_id: str | None = None,
        user_id_hash: str | None = None,
        trace: PipelineTrace | None = None,
    ) -> PipelineContext:
        sanitized = sanitize_text(
            text or "",
            max_chars=self.settings.max_input_chars,
            normalize_unicode=self.settings.normalize_unicode,
            strip_control_chars=self.settings.strip_control_chars,
        )
        from app.core.text import build_detection_view, detect_payloads

        view = build_detection_view(sanitized.text)
        payloads = detect_payloads(sanitized.text, view)
        if payloads.decoded_language_spans:
            # Moderate what is *inside* the envelope. Encoding is the oldest evasion trick in
            # this space, and it only works if the filter treats the blob as opaque; decoded
            # text is appended to the detection view (never to the returned text) so policies
            # see it while spans on the original text stay valid.
            extra = "\n".join(span[2][:2000] for span in payloads.decoded_language_spans[:4])
            # Bounded so a pile of decoded blobs cannot become a CPU amplifier; the
            # detector cost is linear in the view, not in the caller's patience.
            view = f"{view}\n{build_detection_view(extra)}"[: _MAX_DETECTION_VIEW_CHARS]
        return PipelineContext(
            request_id=request_id,
            mode=mode,
            raw_text=text or "",
            text=sanitized.text,
            view=view,
            metadata=metadata or {},
            return_reasoning=return_reasoning,
            key_id=key_id,
            user_id_hash=user_id_hash,
            script_mix=is_multilingual(sanitized.text),
            obfuscation_score=sanitized.obfuscation_score,
            obfuscation_flags=list(sanitized.flags),
            payload_signals=payloads.as_dict(),
            truncated=sanitized.truncated,
            trace=trace,
            started_wall=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    # ------------------------------------------------------------------ results
    @staticmethod
    def _attach(ctx: PipelineContext, stage_name: str, result: Any) -> None:
        if stage_name == Stage1ToxicityStage.name:
            ctx.stage1 = result
        elif stage_name == Stage2PolicyStage.name:
            ctx.stage2 = result
        elif stage_name == Stage3GenerationStage.name:
            ctx.stage3 = result

    def _common(self, ctx: PipelineContext) -> dict[str, Any]:
        bundle = self.policy_bundle()
        from app.pipeline.llm import get_router

        router_provider = get_router(self.settings).candidate_providers()
        model_versions = {
            "policy_version": f"{bundle.version}+{bundle.checksum[:8]}",
            "stage1_classifier": self.settings.stage1_classifier_model if self.settings.stage1_use_classifier else "disabled",
            "stage1_judge": (ctx.stage1.llm_model or "skipped") if ctx.stage1 else "n/a",
            "stage2_pii": "presidio+regex" if (ctx.stage2 and "presidio" in (ctx.stage2.engine or "")) else "regex",
            "stage3_model": (ctx.stage3.model or "n/a") if ctx.stage3 else "n/a",
            "stage3_provider": (ctx.stage3.provider or "n/a") if ctx.stage3 else "n/a",
            "pipeline_version": self.settings.version,
            "llm_provider_chain": ">".join(router_provider),
        }
        scores: dict[str, float] = {}
        for result in (ctx.stage1, ctx.stage2):
            if result is not None:
                for key, value in result.scores.as_dict().items():
                    scores[key] = max(scores.get(key, 0.0), value)
        if ctx.stage3 is not None:
            for key, value in ctx.stage3.scores.as_dict().items():
                scores[key] = max(scores.get(key, 0.0), value)
        latency = {
            "stage1": round(ctx.stage1.latency_ms, 2) if ctx.stage1 else 0.0,
            "stage2": round(ctx.stage2.latency_ms, 2) if ctx.stage2 else 0.0,
            "stage3": round(ctx.stage3.latency_ms, 2) if ctx.stage3 else 0.0,
        }
        return {
            "request_id": ctx.request_id,
            "mode": ctx.mode.value,
            "policy_version": model_versions["policy_version"],
            "model_versions": model_versions,
            "original_text": ctx.text,
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "latency_ms": round(ctx.elapsed_ms(), 2),
            "stage_latency_ms": latency,
            "trace": ctx.trace,
            "created_at": ctx.started_wall or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": ctx.metadata,
        }

    def _blocked_result(self, ctx: PipelineContext, stage_name: str, result: Any) -> PipelineResult:
        violations = _merge_violations(ctx)
        categories = sorted({v.category for v in violations} | set(result.categories))
        blocked_by_judge_error = bool(result.error) and "moderation_error" not in categories
        reason = (
            "moderation_error"
            if blocked_by_judge_error or "moderation_error" in categories
            else _first_reason(result) or "policy_violation"
        )
        user_message = SAFE_FALLBACKS["self_harm"] if "self_harm" in categories else SAFE_FALLBACKS["child_safety"] if "child_safety" in categories else SAFE_FALLBACKS["default"]
        if reason == "moderation_error":
            user_message = (
                "Moderation could not complete safely, so this request was not processed. "
                "Please try again - if it persists, contact support with the request id."
            )
        if ctx.trace is not None:
            ctx.trace.total_latency_ms = round(ctx.elapsed_ms(), 2)
            ctx.trace.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # Stage3Result exposes flat counters, not a usage dict - reading a field that does not
        # exist here turned every blocked-with-output-violation response into a fail-closed
        # moderation_error, which is the worst possible outcome for a moderation *failure*
        # because it hides the real verdict from the reviewer.
        stage3 = ctx.stage3
        token_usage = (
            {
                "prompt_tokens": stage3.prompt_tokens,
                "completion_tokens": stage3.completion_tokens,
                "provider": stage3.provider,
                "model": stage3.model,
                "attempts": stage3.attempts,
            }
            if stage3 is not None
            else {}
        )
        return PipelineResult(
            transformations=self._transformations(ctx, reveal_source=False),
            **self._common(ctx),
            status=Status.BLOCKED,
            decision=Decision.BLOCK,
            safe_output="",
            action_taken="blocked",
            violations=violations,
            categories=categories,
            confidence=round(max((result.confidence, _aggregate_confidence(ctx, blocking=True), *(v.confidence for v in violations)), default=0.9), 4),
            reason=reason,
            blocked_reason=reason,
            user_message=user_message,
            fail_closed=reason == "moderation_error",
            degraded=bool(ctx.stage1 and ctx.stage1.degraded) or bool(ctx.stage2 and ctx.stage2.degraded),
            token_usage=token_usage,
            estimated_cost_usd=float((ctx.stage3.estimated_cost_usd if ctx.stage3 else 0.0) or 0.0),
            error=result.error,
        )

    @staticmethod
    def _transformations(ctx: PipelineContext, *, reveal_source: bool) -> list[dict[str, Any]]:
        """Compact action rows for the response/UI.

        ``reveal_source=False`` (blocked responses) keeps the *what changed* shape but
        drops the original text, so a blocked request cannot leak its own payload.
        """
        rows: list[dict[str, Any]] = []
        for action in (ctx.stage2.actions if ctx.stage2 else [])[:20]:
            row = {
                "policy_id": action.policy_id,
                "rule": action.rule_name,
                "action": action.action,
                "replacement": action.replacement,
                "confidence": round(action.confidence, 4),
                "reason": action.reason[:200],
                "span": [action.start, action.end] if action.start is not None and action.end is not None else None,
            }
            if reveal_source:
                row["original"] = action.original[:160]
            rows.append(row)
        return rows

    def _allowed_result(self, ctx: PipelineContext, bundle: Any) -> PipelineResult:
        stage2, stage3 = ctx.stage2, ctx.stage3
        violations = _merge_violations(ctx)
        categories = sorted({v.category for v in violations})
        actions = list(stage2.actions) if stage2 else []
        masked = any(a.action == "mask" for a in actions)
        rewrote = any(a.action in {"rewrite", "blocked_span_removed", "normalize"} for a in actions) or (stage2 is not None and stage2.status == "rewritten")
        # ``rewritten`` is a contract, not a description of the byte diff: it has to mean "your
        # own words were changed", because that is what a client can act on (show a diff, warn
        # the author). Disclaimers are *appended* after the caller's text, so compare the text
        # with the disclaimers removed - otherwise every crisis-resources response would be
        # reported as a rewrite of the user's message.
        user_text = stage2.text if stage2 else ""
        for disclaimer in (stage2.disclaimers if stage2 else []):
            user_text = user_text.replace(disclaimer, "")
        text_changed = bool(user_text.strip() and user_text.rstrip(" \n-") != ctx.text.rstrip(" \n-"))

        if stage3 is not None and not stage3.output_check_passed:
            status, action, reason = Status.BLOCKED, "blocked", "output_policy_violation"
            safe_output = stage3.output  # already the canned safe message
        elif rewrote or masked or text_changed or (stage2 and stage2.disclaimers) or (stage3 and stage3.mode == "assistant"):
            annotations = bool(stage2 and stage2.disclaimers)
            status = Status.REWRITTEN if (masked or rewrote or text_changed) else Status.ALLOWED
            action = (
                "masked_pii_and_rewrote"
                if masked and rewrote
                else "masked_pii"
                if masked
                else "rewrote_segments"
                if rewrote
                else "added_safety_annotations"
                if annotations
                else "none"
            )
            reason = "policy_transformations_applied" if (masked or rewrote or text_changed) else "policy_annotations_added"
            safe_output = stage3.output if stage3 else (stage2.text if stage2 else ctx.text)
        else:
            status, action, reason = Status.ALLOWED, "none", "no_violations"
            safe_output = stage3.output if stage3 else (stage2.text if stage2 else ctx.text)

        if ctx.trace is not None:
            ctx.trace.total_latency_ms = round(ctx.elapsed_ms(), 2)
            ctx.trace.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        confidence = _aggregate_confidence(ctx, blocking=False)
        degraded = bool((ctx.stage1 and ctx.stage1.degraded) or (ctx.stage2 and ctx.stage2.degraded) or (stage3 and stage3.degraded))
        return PipelineResult(
            transformations=self._transformations(ctx, reveal_source=status is not Status.BLOCKED),
            **self._common(ctx),
            status=status,
            decision=Decision.FLAG if categories else Decision.PASS,
            safe_output=safe_output,
            action_taken=action,  # type: ignore[arg-type]
            violations=violations,
            categories=categories,
            confidence=confidence,
            reason=reason,
            disclaimers=list(stage2.disclaimers) if stage2 else [],
            degraded=degraded,
            token_usage={
                "prompt_tokens": stage3.prompt_tokens if stage3 else 0,
                "completion_tokens": stage3.completion_tokens if stage3 else 0,
                "provider": stage3.provider if stage3 else "none",
                "model": stage3.model if stage3 else "none",
                "attempts": stage3.attempts if stage3 else 0,
            },
            estimated_cost_usd=float(stage3.estimated_cost_usd if stage3 else 0.0),
        )

    def _fail_closed(self, ctx: PipelineContext, detail: str, *, trace_only_names: list[str] | None = None) -> PipelineResult:
        """Total-failure result: blocked, no content, reason ``moderation_error``."""
        log.error("pipeline fail-closed", extra={"request_id": ctx.request_id, "detail": detail[:300]})
        for name in trace_only_names or []:
            if ctx.trace is not None:
                ctx.trace.add(StageTrace(name=name, status="error", decision="skipped", detail={"reason": "pipeline timeout"}))
        if ctx.trace is not None:
            ctx.trace.total_latency_ms = round(ctx.elapsed_ms(), 2)
            ctx.trace.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            ctx.trace.short_circuited_at = ctx.trace.short_circuited_at or "pipeline"
        return PipelineResult(
            **self._common(ctx),
            status=Status.BLOCKED,
            decision=Decision.BLOCK,
            safe_output="",
            action_taken="blocked",
            violations=[
                Violation(
                    category="moderation_error",
                    severity="critical",
                    confidence=0.99,
                    action="block",
                    policy_id="fail_closed",
                    stage="orchestrator",
                    detector="orchestrator",
                    reason=detail[:300],
                )
            ],
            categories=["moderation_error"],
            confidence=0.99,
            reason="moderation_error",
            blocked_reason="moderation_error",
            user_message=(
                "Moderation could not complete safely, so this request was not processed. "
                "Please try again - if it persists, contact support with the request id."
            ),
            fail_closed=True,
            degraded=True,
            error=detail[:400],
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def policy_bundle() -> Any:
        from app.pipeline.policy import get_policy_bundle

        return get_policy_bundle()

    def describe(self) -> dict[str, Any]:
        return {
            "stages": [{"name": s.name, "timeout_s": s.timeout_s, "fail_closed": s.fail_closed} for s in self.stages],
            "global_timeout_s": self.settings.pipeline_request_timeout_s,
            "policy_version": self.policy_bundle().version,
            "provider_chain": ">".join(_router_providers(self.settings)),
        }


# --------------------------------------------------------------------- utilities


def _remaining_stages(stages: Sequence[Stage], current: str) -> list[str]:
    names = [s.name for s in stages]
    return names[names.index(current) + 1 :] if current in names else []


def _merge_violations(ctx: PipelineContext) -> list[Violation]:
    merged: dict[tuple[str, str], Violation] = {}
    for result in (ctx.stage1, ctx.stage2):
        for violation in getattr(result, "violations", []) or []:
            key = (violation.category, violation.policy_id or "")
            existing = merged.get(key)
            if existing is None or violation.confidence > existing.confidence:
                merged[key] = violation
    if ctx.stage3 is not None:
        for violation in ctx.stage3.output_check_violations:
            merged[(violation.category, violation.policy_id or "")] = violation
    return sorted(merged.values(), key=lambda v: -v.confidence)[:32]


def _first_reason(result: Any) -> str | None:
    for reason in getattr(result, "reasons", []) or []:
        text = str(reason)
        if text.startswith("blocked: "):
            text = text[len("blocked: ") :]
        marker = text.split(" ", 1)[0].strip("()[],")
        if marker and marker.replace("_", "").replace("-", "").isalnum():
            return marker[:60]
        return text[:60] or None
    return None


def _aggregate_confidence(ctx: PipelineContext, *, blocking: bool) -> float:
    """Confidence that the *final* decision is right.

    Blocks are confident when several independent layers agreed; passes are confident
    when nothing at all was close to a threshold. Raw detector maxes would make a clean
    "hello" look uncertain (it would report 0.99 *because* it is safe), which is why the
    pass case is computed from distance-to-threshold instead.
    """
    thresholds = ModerationPipeline.policy_bundle().thresholds(ctx.mode.value)
    scores = ctx.stage1.scores.as_dict() if ctx.stage1 else {}
    peak = max(
        (
            scores.get("toxicity", 0.0) / max(thresholds.toxicity_block, 1e-6),
            scores.get("prompt_injection", 0.0) / max(thresholds.injection_block, 1e-6),
            scores.get("jailbreak", 0.0) / max(thresholds.jailbreak_block, 1e-6),
        ),
        default=0.0,
    )
    layer_agreement = sum(1 for r in (ctx.stage1, ctx.stage2, ctx.stage3) if r is not None and r.degraded)
    if blocking or (ctx.stage1 and ctx.stage1.decision is Decision.BLOCK):
        detectors = 1 + bool(ctx.stage1.judge_used) + bool((ctx.stage1.detector_detail or {}).get("classifier", {}).get("available"))
        return round(min(0.99, 0.6 + 0.08 * detectors + 0.25 * min(peak, 1.5)), 4)
    # pass / rewrite: further from any threshold -> higher confidence
    return round(max(0.35, min(0.98, 0.93 - 0.45 * max(0.0, min(peak, 1.0)) - 0.03 * layer_agreement)), 4)


def _router_providers(settings: Settings) -> list[str]:
    from app.pipeline.llm import get_router

    return get_router(settings).candidate_providers()


_PIPELINE: ModerationPipeline | None = None


def get_pipeline(settings: Settings | None = None) -> ModerationPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = ModerationPipeline(settings or get_settings())
    return _PIPELINE


def reset_pipeline() -> None:  # pragma: no cover - test helper
    global _PIPELINE
    _PIPELINE = None


async def run(text: str, **kwargs: Any) -> PipelineResult:  # pragma: no cover - convenience alias
    return await get_pipeline().run(text, **kwargs)
