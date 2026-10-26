"""Stage 1 - toxicity & prompt-injection gate (the "STOP HERE" stage).

Three inputs are fused:

* **Layer A.1 heuristics** (:mod:`app.pipeline.injection`) - structural injection
  signatures, lexicon hits, obfuscation and encoded-payload evidence.
* **Layer A.2 classifier** - ``protectai/deberta-v3-base-prompt-injection-v2`` when
  ``transformers`` is installed, otherwise reported as unavailable.
* **Layer B small-LLM judge** - strict JSON-only verdict, parsed with Pydantic, one
  reformat retry, then **fail closed**. The user text is quarantined inside a
  randomised delimiter and the prompt states - before and after the payload - that
  the block is data, which is what stops the judge itself from being injected.

Fusion is deliberately explainable rather than a black-box average:

    heuristic  : high precision, fires alone
    model side : classifier_weight * classifier + llm_weight * judge   (weighted mean)
    fused      : max(heuristic, model_side), boosted when both agree
    decision   : block when any fused score >= its per-mode threshold, or when a
                 ``stage1`` policy in policies.yaml fires with action=block

A block here short-circuits the chain: Stage 2/3 never run, so an unrewritten
harmful prompt can never reach the generation model.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.core.logging import get_logger
from app.core.text import detect_payloads
from app.pipeline.base import Decision, PipelineContext, Scorecard, Stage, Stage1Result, Violation
from app.pipeline.injection import ClassifierOutcome, InjectionHeuristics, get_classifier
from app.pipeline.llm import ChatMessage, LLMError, LLMNotConfigured, estimate_tokens
from app.pipeline.policy import RuleHit
from app.pipeline.prompts import build_judge_prompt

log = get_logger(__name__)

ALLOWED_CATEGORIES = {
    "hate",
    "harassment",
    "violence",
    "self_harm",
    "sexual",
    "weapons",
    "drugs",
    "pii_request",
    "fraud",
    "child_safety",
    "jailbreak",
    "prompt_injection",
    "obfuscation",
    "bullying",
    "toxic",
    "none",
}


class JudgeVerdict(BaseModel):
    """Exactly the JSON shape the judge is required to return."""

    model_config = ConfigDict(extra="forbid")

    toxicity: float = 0.0
    categories: list[str] = Field(default_factory=lambda: ["none"])
    prompt_injection: float = 0.0
    jailbreak: float = 0.0
    nsfw: float = 0.0
    reason: str = ""
    decision: Decision = Decision.PASS

    @field_validator("toxicity", "prompt_injection", "jailbreak", "nsfw", mode="before")
    @classmethod
    def _clamp(cls, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"score must be numeric, got {value!r}") from exc
        if number != number:  # NaN guard - a NaN would silently disable comparisons
            raise ValueError("score must not be NaN")
        return max(0.0, min(1.0, number))

    @field_validator("categories", mode="before")
    @classmethod
    def _clean_categories(cls, value: Any) -> list[str]:
        if value is None:
            return ["none"]
        if isinstance(value, str):
            value = [part.strip() for part in value.replace(",", " ").split()]
        out: list[str] = []
        for item in value:
            text = str(item).strip().lower().replace("-", "_").replace(" ", "_")
            if text and text not in out:
                out.append(text)
        return out or ["none"]

    @field_validator("decision", mode="before")
    @classmethod
    def _decision_alias(cls, value: Any) -> Any:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"allow", "safe", "ok", "clean", "pass"}:
                return "pass"
            if lowered in {"block", "deny", "reject", "unsafe", "flag"}:
                return "block"
        return value

    @property
    def unknown_categories(self) -> list[str]:
        return [c for c in self.categories if c not in ALLOWED_CATEGORIES]


class Stage1ToxicityStage(Stage):
    """Toxicity + prompt-injection check with configurable per-mode thresholds."""

    name = "stage1_toxicity"
    fail_closed = True

    def __init__(self, settings: Any = None, *, timeout_s: float | None = None, use_llm: bool | None = None, use_classifier: bool | None = None) -> None:
        from app.core.config import get_settings

        self.settings = settings or get_settings()
        self.timeout_s = timeout_s if timeout_s is not None else self.settings.stage1_timeout_s
        self.use_llm = self.settings.stage1_use_llm_judge if use_llm is None else use_llm
        self.use_classifier = self.settings.stage1_use_classifier if use_classifier is None else use_classifier
        self.heuristics = InjectionHeuristics()

    def _default_timeout(self) -> float:
        return self.settings.stage1_timeout_s

    # ---------------------------------------------------------------- main body
    async def _run(self, ctx: PipelineContext) -> Stage1Result:
        started = time.perf_counter()
        bundle = self.policy_bundle
        thresholds = bundle.thresholds(ctx.mode.value)
        payloads = detect_payloads(ctx.text, ctx.view)

        # ---- Layer A.1: deterministic heuristics ----------------------------
        heuristic_ran = True
        try:
            heuristic = self.heuristics.analyze(
                ctx.text,
                ctx.view,
                payloads=payloads,
                obfuscation_score=ctx.obfuscation_score,
            )
        except Exception as exc:  # the one condition that must block with no evidence
            heuristic_ran = False
            log.error("stage1 heuristics crashed", extra={"request_id": ctx.request_id, "error": str(exc)[:200]}, exc_info=True)
            return Stage1Result(
                stage=self.name,
                decision=Decision.BLOCK,
                categories=["moderation_error"],
                reasons=["Stage-1 detectors failed to run; request blocked fail-closed."],
                confidence=0.99,
                degraded=True,
                engine="fail-closed",
                error=f"{exc.__class__.__name__}: {exc}"[:300],
            )

        # ---- Layer A.2: transformer classifier ------------------------------
        classifier = ClassifierOutcome(available=False, reason="disabled_by_config")
        if self.use_classifier:
            clf = get_classifier(self.settings.stage1_classifier_model, device=self.settings.stage1_classifier_device)
            classifier = await clf.score(ctx.text, timeout_s=min(8.0, max(1.0, self.timeout_s / 2)))

        # ---- YAML rules, fed with the numeric evidence above ---------------
        scores_for_rules = {
            "toxicity": max(heuristic.toxicity_score, classifier.scores.get("toxicity", 0.0)),
            "prompt_injection": max(heuristic.injection_score, classifier.scores.get("prompt_injection", 0.0)),
            "jailbreak": max(heuristic.jailbreak_score, classifier.scores.get("prompt_injection", 0.0)),
            "nsfw": heuristic.nsfw_score,
        }
        hits: list[RuleHit] = bundle.evaluate(
            stage="stage1",
            mode=ctx.mode.value,
            text=ctx.text,
            view=ctx.view,
            scores=scores_for_rules,
            obfuscation_score=ctx.obfuscation_score,
            obfuscation_flags=ctx.obfuscation_flags,
            payloads=payloads,
            script_mix=ctx.script_mix,
        )
        rule_toxicity = max((h.score for h in hits if h.category in Stage1ToxicityStage.TOXIC_CATEGORIES), default=0.0)
        rule_injection = max((h.score for h in hits if h.category in Stage1ToxicityStage.INJECTION_CATEGORIES), default=0.0)
        blocking_rules = [h for h in hits if h.action == "block"]

        # ---- Layer B: small-LLM judge ---------------------------------------
        judge: JudgeVerdict | None = None
        judge_skipped: str | None = None
        llm_usage: dict[str, Any] | None = None
        judge_error: str | None = None
        if not self.use_llm:
            judge_skipped = "llm_judge_disabled_by_config"
        else:
            try:
                judge_raw, llm_response = await self._call_judge(ctx)
                judge = JudgeVerdict.model_validate(judge_raw)
                llm_usage = llm_response.as_dict()
            except LLMNotConfigured as exc:
                # No judge in this deployment: degrade explicitly, keep Layer A.
                judge_skipped = f"llm_judge_not_configured ({exc})"
            except (ValidationError, ValueError) as exc:
                judge_error = f"judge_output_invalid: {exc}"[:300]
            except LLMError as exc:
                judge_error = f"judge_call_failed: {exc}"[:300]
            except Exception as exc:  # pragma: no cover - unexpected
                judge_error = f"judge_crashed: {exc.__class__.__name__}: {exc}"[:300]

        # ---- fusion ---------------------------------------------------------
        w_llm = thresholds.llm_weight
        w_clf = thresholds.classifier_weight
        judge_toxicity = judge.toxicity if judge else 0.0
        judge_injection = max(judge.prompt_injection, judge.jailbreak) if judge else 0.0
        cls_toxicity = classifier.scores.get("toxicity", 0.0)
        cls_injection = classifier.scores.get("prompt_injection", 0.0)

        model_toxicity = w_llm * judge_toxicity + w_clf * cls_toxicity
        model_injection = w_llm * judge_injection + w_clf * cls_injection

        def fuse(rule_score: float, heuristic_score: float, model_score: float) -> float:
            deterministic = max(rule_score, heuristic_score)
            if deterministic >= 0.4 and model_score >= 0.4:  # agreement escalation
                return min(1.0, max(deterministic, model_score) + 0.15)
            return min(1.0, max(deterministic, model_score))

        toxicity = round(fuse(rule_toxicity, heuristic.toxicity_score, model_toxicity), 4)
        prompt_injection = round(fuse(rule_injection, heuristic.injection_score, model_injection), 4)
        jailbreak = round(min(1.0, max(prompt_injection, heuristic.jailbreak_score, w_llm * (judge.jailbreak if judge else 0.0))), 4)
        nsfw = round(max(heuristic.nsfw_score, judge.nsfw if judge else 0.0), 4)

        categories = sorted(set(heuristic.categories) | {h.category for h in hits} | (set(judge.categories) - {"none"} if judge else set()))
        categories = [c for c in categories if c and c != "none"]

        violations: list[Violation] = [self._violation_from_hit(hit, ctx) for hit in hits]
        if judge:
            violations.extend(self._violations_from_judge(judge))
        violations = self._dedupe(violations)

        # decision
        reasons: list[str] = []
        decision = Decision.PASS
        exceeded: list[str] = []
        if toxicity >= thresholds.toxicity_block:
            exceeded.append(f"toxicity {toxicity:.2f} >= {thresholds.toxicity_block:.2f}")
        if prompt_injection >= thresholds.injection_block:
            exceeded.append(f"prompt_injection {prompt_injection:.2f} >= {thresholds.injection_block:.2f}")
        if jailbreak >= thresholds.jailbreak_block:
            exceeded.append(f"jailbreak {jailbreak:.2f} >= {thresholds.jailbreak_block:.2f}")
        if judge and judge.decision is Decision.BLOCK:
            exceeded.append(f"judge verdict=block ({judge.reason[:120]})")
        if blocking_rules:
            exceeded.append("policy block: " + ", ".join(h.policy_id for h in blocking_rules[:3]))

        # A self-harm message asking for help is answered, never refused - even when the model is
        # confident, because it is confident about the *topic*. The exemption only removes the
        # score-driven line: any policy block (methods, incitement, a weapon, a minor) still
        # blocks, and so does injection evidence, which is the shape an attack actually takes.
        if exceeded and heuristic.crisis_help_seeking:
            kept = [line for line in exceeded if not line.startswith("toxicity ")]
            if blocking_rules and all((hit.category or "") == "self_harm" for hit in blocking_rules):
                self_harm_only = {h.policy_id for h in blocking_rules if (h.category or "") == "self_harm"}
                other_rules = [h for h in blocking_rules if h.policy_id not in self_harm_only]
                if not other_rules:
                    kept = [line for line in kept if not line.startswith("policy block:")]
            if len(kept) != len(exceeded):
                reasons.append(
                    "crisis help-seeking: model/policy toxicity alone does not block - "
                    "answered with resources and flagged for review"
                )
            exceeded = kept

        if thresholds.stage1_consensus == "both":
            # lenient mode: an LLM-only signal is not enough to stop the request
            deterministic_evidence = bool(blocking_rules) or max(heuristic.injection_score, heuristic.toxicity_score) >= 0.7
            if exceeded and not deterministic_evidence:
                reasons.append(f"lenient consensus: model-only signals downgraded to flag ({'; '.join(exceeded)})")
                exceeded = []

        # A judge that *errored* (not one that was simply not configured) must not be
        # read as "clean" while the fail-closed policy is on.
        if judge_error and self.settings.stage1_fail_closed_on_llm_error:
            exceeded.append(f"judge unavailable and fail-closed configured ({judge_error[:120]})")
        if not heuristic_ran:
            exceeded.append("no Stage-1 detector produced evidence for this input")

        if exceeded:
            decision = Decision.BLOCK
            reasons = [f"blocked: {line}" for line in exceeded] + reasons
        elif categories:
            decision = Decision.FLAG
            reasons.append("flagged for Stage-2 handling: " + ", ".join(categories[:4]))

        confidence = round(max(toxicity, prompt_injection, jailbreak, nsfw, rule_toxicity), 4)
        engine_bits = ["heuristics"]
        if classifier.available:
            engine_bits.append(f"classifier:{classifier.model.split('/')[-1]}")
        if judge is not None:
            engine_bits.append(f"judge:{(llm_usage or {}).get('provider', '?')}/{(llm_usage or {}).get('model', '?')}")
        elif judge_skipped:
            engine_bits.append("judge=skipped")
        elif judge_error:
            engine_bits.append("judge=failed")

        result = Stage1Result(
            stage=self.name,
            decision=decision,
            confidence=confidence,
            toxicity=toxicity,
            prompt_injection=prompt_injection,
            jailbreak=jailbreak,
            scores=Scorecard(
                toxicity=toxicity,
                prompt_injection=prompt_injection,
                jailbreak=jailbreak,
                nsfw=nsfw,
                obfuscation=ctx.obfuscation_score,
                sensitive_topic=max(heuristic.toxicity_score, nsfw),
            ),
            categories=categories,
            violations=violations,
            reasons=reasons or [f"stage1 clear (toxicity={toxicity:.2f}, injection={prompt_injection:.2f}, mode={ctx.mode.value})"],
            engine="+".join(engine_bits),
            detector_detail={"heuristics": heuristic.as_dict(), "classifier": classifier.as_dict()},
            judge=judge.model_dump(mode="json") if judge else None,
            judge_used=judge is not None,
            judge_skipped_reason=judge_skipped or judge_error,
            threshold=thresholds.toxicity_block,
            heuristics=[s.name for s in heuristic.signals],
            llm_model=(llm_usage or {}).get("model"),
            llm_usage=llm_usage,
            degraded=bool(judge_error) or (judge_skipped is not None and "not_configured" in str(judge_skipped)) or not classifier.available,
            error=judge_error if decision is Decision.BLOCK else None,
        )
        log.debug(
            "stage1 decision",
            extra={
                "request_id": ctx.request_id,
                "decision": decision.value,
                "toxicity": toxicity,
                "injection": prompt_injection,
                "categories": categories,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return result

    # ------------------------------------------------------------------- judge
    async def _call_judge(self, ctx: PipelineContext) -> tuple[dict[str, Any], Any]:
        system, user = build_judge_prompt(ctx.text, request_id=ctx.request_id)
        router = self._router()
        try:
            data, response = await router.chat_json(
                [ChatMessage("system", system), ChatMessage("user", user)],
                tier="small",
                timeout=min(self.settings.llm_request_timeout_s, max(3.0, self.timeout_s - 2)),
                retries_on_parse_error=1,
            )
        except LLMError as exc:
            if "all LLM providers failed" in str(exc):
                provider = self.settings.resolved_provider()
                if provider == "local":
                    raise LLMNotConfigured("no LLM provider configured (llm_provider=local)") from exc
            raise
        if data.get("available") is False and data.get("engine") == "local-deterministic":
            raise LLMNotConfigured(str(data.get("note", "local engine cannot judge")))
        if data.get("engine") == "local-deterministic":
            raise LLMNotConfigured("judge ran on the offline engine, which does not produce verdicts")
        return data, response

    def _router(self) -> Any:
        from app.pipeline.llm import get_router

        return get_router(self.settings)

    # ------------------------------------------------------------- conversions
    TOXIC_CATEGORIES = {"hate", "harassment", "violence", "self_harm", "sexual", "weapons", "drugs", "child_safety", "bullying", "toxic", "fraud", "gore"}
    INJECTION_CATEGORIES = {"prompt_injection", "jailbreak", "obfuscation", "system_prompt_leak", "pii_request"}

    @staticmethod
    def _dedupe(violations: list[Violation]) -> list[Violation]:
        seen: dict[tuple[str, str, int | None], Violation] = {}
        for violation in violations:
            key = (violation.category, violation.policy_id or "", violation.span[0] if violation.span else None)
            existing = seen.get(key)
            if existing is None or violation.confidence > existing.confidence:
                seen[key] = violation
        return sorted(seen.values(), key=lambda v: -v.confidence)[:24]

    @staticmethod
    def _violation_from_hit(hit: RuleHit, ctx: PipelineContext) -> Violation:
        return Violation(
            category=hit.category,
            severity=hit.severity,  # type: ignore[arg-type]
            confidence=hit.score,
            action=hit.action,
            policy_id=hit.policy_id,
            stage="stage1",
            detector=hit.detector,
            reason=hit.reason[:400],
            excerpt=(hit.matched[0][:120] if hit.matched else None),
            span=tuple(hit.spans[0]) if hit.spans else None,
        )

    @staticmethod
    def _violations_from_judge(judge: JudgeVerdict) -> list[Violation]:
        score_by_category = {
            "hate": judge.toxicity,
            "harassment": judge.toxicity,
            "violence": judge.toxicity,
            "self_harm": judge.toxicity,
            "sexual": max(judge.nsfw, judge.toxicity),
            "weapons": judge.toxicity,
            "drugs": judge.toxicity,
            "child_safety": max(0.95, judge.toxicity),
            "fraud": judge.toxicity,
            "pii_request": max(judge.toxicity, judge.prompt_injection),
            "prompt_injection": judge.prompt_injection,
            "jailbreak": judge.jailbreak,
            "toxic": judge.toxicity,
            "bullying": judge.toxicity,
        }
        violations: list[Violation] = []
        for category in judge.categories:
            if category == "none":
                continue
            severity = "critical" if category in {"child_safety", "self_harm", "hate"} else "high" if category in {"violence", "sexual", "prompt_injection", "jailbreak"} else "medium"
            violations.append(
                Violation(
                    category=category,
                    severity=severity,  # type: ignore[arg-type]
                    confidence=score_by_category.get(category, judge.toxicity),
                    action="block" if judge.decision is Decision.BLOCK else "flag",
                    policy_id=f"judge:{category}",
                    stage="stage1",
                    detector="llm_judge",
                    reason=judge.reason[:300] or "small-LLM judge flagged this category",
                    excerpt=None,
                )
            )
        if judge.unknown_categories:
            log.warning("judge reported unknown categories", extra={"unknown": judge.unknown_categories})
        return violations

    def _fail_closed(self, ctx: PipelineContext, error: str, *, timeout: bool = False) -> Stage1Result:
        reason = f"stage1_{'timeout' if timeout else 'error'}"
        return Stage1Result(
            stage=self.name,
            decision=Decision.BLOCK,
            reasons=[f"Stage 1 could not complete safely ({error}). Request blocked fail-closed."],
            confidence=0.99,
            categories=["moderation_error"],
            engine="fail-closed",
            degraded=True,
            error=f"{reason}: {error}"[:400],
            scores=Scorecard(),
        )

    async def execute(self, ctx: PipelineContext) -> Stage1Result:
        result = await super().execute(ctx)
        if isinstance(result, Stage1Result):
            return result
        # base class produced a generic StageResult (timeout / crash): promote it
        return self._fail_closed(ctx, str(result.error or "unknown"), timeout="timeout" in (result.error or ""))

    @staticmethod
    def prompt_token_estimate(ctx: PipelineContext) -> int:
        _system, user = build_judge_prompt(ctx.text, request_id=ctx.request_id)
        return estimate_tokens(user)
