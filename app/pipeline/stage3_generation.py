"""Stage 3 - safe final output generation + output-side guardrails.

Two operating modes, chosen by ``GUARD_STAGE3_MODE``:

``passthrough`` (default)
    The **sanitized** Stage-2 text is the safe output. This is the right default for a
    moderation API: consumers get the cleaned content, we never invent an answer, and
    the whole chain works with no external API at all.
``assistant``
    A main LLM (Gemini / GPT-4o / Claude / Ollama) answers the sanitized request under
    a hard system prompt, then the answer is *re-checked* before it leaves.

Either way the output passes an independent check (policy stage ``stage3_output`` +
toxicity heuristics + PII-leak scan). A failing check never returns the model's text:
we substitute a canned safe message and mark the trace, because "the moderator said the
answer is fine" is only useful if something verifies it.

``return_reasoning`` produces a short, audience-safe explanation (written by a small
model when available, otherwise composed from the policy hits). It is deliberately NOT
chain-of-thought: raw reasoning would leak thresholds and the judge prompt.
"""

from __future__ import annotations

import re
import time
from typing import Any

from app.core.logging import get_logger
from app.pipeline.base import Decision, PipelineContext, Scorecard, Stage, Stage3Result, Violation
from app.pipeline.injection import InjectionHeuristics
from app.pipeline.llm import ChatMessage, LLMError, LLMNotConfigured
from app.pipeline.pii import PiiDetector
from app.pipeline.prompts import (
    JAILBREAK_COMPLIANCE_MARKERS,
    build_generation_prompt,
    build_reasoning_prompt,
)

log = get_logger(__name__)

SAFE_FALLBACKS = {
    "default": (
        "I can't provide that content. I can help with a safer version of the request - "
        "ask me for a summary, an explanation of the risks, or a general overview."
    ),
    "self_harm": (
        "I'm sorry you're dealing with this. I can't help with harming yourself, but you "
        "deserve support right now: in Bangladesh call KaanchBCP 106 (24/7, free) or "
        "emergency services on 999. If you can, tell someone you trust today."
    ),
    "child_safety": "I can't help with anything involving the exploitation of children. If you know a child is at risk, contact Bangabandhu Children's Helpline 10980.",
    "violence": "I can't help with harming people or property. I can help with conflict de-escalation, reporting a threat, or safety planning instead.",
    "pii": "I can't repeat personal contact details. I can help you draft a message to that person using a placeholder for their contact info.",
}
CRITICAL_LEAK_CATEGORIES = {"pii_leak", "system_prompt_leak", "unsafe_advice", "child_safety"}


class Stage3GenerationStage(Stage):
    """Generate the final safe answer and defend it with an output check."""

    name = "stage3_generation"
    fail_closed = True

    def __init__(self, settings: Any = None, *, timeout_s: float | None = None, mode: str | None = None) -> None:
        from app.core.config import get_settings

        self.settings = settings or get_settings()
        self.timeout_s = timeout_s if timeout_s is not None else self.settings.stage3_timeout_s
        self.mode = mode or self.settings.stage3_mode
        self._heuristics = InjectionHeuristics()
        self._pii = PiiDetector(use_presidio=False)  # output scan must stay fast + offline

    def _default_timeout(self) -> float:
        return self.settings.stage3_timeout_s

    # --------------------------------------------------------------------- main
    async def _run(self, ctx: PipelineContext) -> Stage3Result:
        started = time.perf_counter()
        sanitized = ctx.text_for_next_stage
        disclaimers = list(ctx.stage2.disclaimers) if ctx.stage2 else []

        usage: dict[str, Any] = {}
        provider = "none"
        model = "none"
        attempts = 1
        if self.mode == "assistant":
            output, usage, provider, model, attempts = await self._generate(ctx, sanitized)
            if not output.strip():
                output = sanitized
        else:
            output = sanitized
            usage = {"prompt_tokens": 0, "completion_tokens": _tokens(output), "estimated_cost_usd": 0.0, "provider": "none", "model": "passthrough"}
            provider, model = "passthrough", "policy-sanitized-text"

        if disclaimers:
            output = self._append_disclaimers(output, disclaimers)

        # ------------------------------------------------- output-side guardrails
        check_violations: list[Violation] = []
        check_passed = True
        if self.settings.stage3_output_check:
            check_passed, check_violations = self._output_check(ctx, output)
        if not check_passed:
            categories = {v.category for v in check_violations}
            fallback_key = next((c for c in ("child_safety", "self_harm", "violence", "pii") if c in categories), "default")
            log.warning(
                "stage3 output check failed; substituted safe response",
                extra={"request_id": ctx.request_id, "categories": sorted(categories), "provider": provider},
            )
            output = SAFE_FALLBACKS[fallback_key]
            if fallback_key != "default" and disclaimers:
                output = self._append_disclaimers(output, disclaimers[:1])

        reasoning: str | None = None
        if ctx.return_reasoning:
            reasoning = await self._explain(ctx, output)

        confidence = self._confidence(ctx, check_passed)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return Stage3Result(
            stage=self.name,
            decision=Decision.PASS if check_passed else Decision.BLOCK,
            latency_ms=latency_ms,
            confidence=confidence,
            output=output,
            reasoning=reasoning,
            output_check_passed=check_passed,
            output_check_violations=check_violations,
            provider=provider,
            model=model,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            estimated_cost_usd=float(usage.get("estimated_cost_usd", 0.0) or 0.0),
            attempts=attempts,
            mode=self.mode,
            scores=Scorecard(
                toxicity=ctx.stage1.toxicity if ctx.stage1 else 0.0,
                prompt_injection=ctx.stage1.prompt_injection if ctx.stage1 else 0.0,
                jailbreak=ctx.stage1.jailbreak if ctx.stage1 else 0.0,
                pii_risk=self._pii_risk(output),
            ),
            categories=[v.category for v in check_violations] or (["sensitive_topic"] if (ctx.stage1 and ctx.stage1.decision is Decision.FLAG) else []),
            reasons=(
                ["output passed the response-side policy check"]
                if check_passed
                else [f"output check blocked: {v.reason[:120]}" for v in check_violations[:3]]
            ),
            engine=f"{self.mode}:{provider}/{model}",
            output_sanitized=bool(disclaimers) or not check_passed,
        )

    # ---------------------------------------------------------------- generation
    async def _generate(self, ctx: PipelineContext, sanitized: str) -> tuple[str, dict[str, Any], str, str, int]:
        from app.pipeline.llm import get_router

        system, user = build_generation_prompt(sanitized, request_id=ctx.request_id)
        try:
            response = await get_router(self.settings).chat(
                [ChatMessage("system", system), ChatMessage("user", user)],
                tier="main",
                timeout=min(self.timeout_s - 1.0, self.settings.llm_request_timeout_s * 1.5),
            )
        except (LLMNotConfigured, LLMError) as exc:
            log.warning("stage3 generation unavailable; returning sanitized text", extra={"request_id": ctx.request_id, "reason": str(exc)[:200]})
            notice = ""
            if isinstance(exc, LLMNotConfigured):
                notice = "\n\n[Generation model not configured - returning the policy-sanitized text as-is.]"
            return sanitized + notice, {"prompt_tokens": 0, "completion_tokens": 0, "estimated_cost_usd": 0.0}, "unavailable", "none", 1
        return (
            response.text.strip(),
            {
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "estimated_cost_usd": response.estimated_cost_usd,
                "provider": response.provider,
                "model": response.model,
                "llm_latency_ms": round(response.latency_ms, 2),
                "attempts": response.attempts,
            },
            response.provider,
            response.model,
            response.attempts,
        )

    # -------------------------------------------------------------- output check
    def _output_check(self, ctx: PipelineContext, output: str) -> tuple[bool, list[Violation]]:
        """Run the stage3_output policies plus direct leak detectors on the answer."""
        from app.core.text import build_detection_view, detect_payloads

        view = build_detection_view(output)
        payloads = detect_payloads(output, view)
        heuristic = self._heuristics.analyze(output, view, payloads=payloads, obfuscation_score=0.0)
        entities = [span.as_dict() for span in self._pii.detect(output).spans]
        hits = self.policy_bundle.evaluate(
            stage="stage3_output",
            mode=ctx.mode.value,
            text=output,
            view=view,
            scores={
                "toxicity": heuristic.toxicity_score,
                "nsfw": heuristic.nsfw_score,
                "prompt_injection": heuristic.injection_score,
                "jailbreak": heuristic.jailbreak_score,
            },
            llm_categories={c: 0.8 for c in heuristic.categories},
            obfuscation_score=0.0,
            obfuscation_flags=[],
            payloads=payloads,
            script_mix=ctx.script_mix,
            entities=entities,
        )
        violations = [
            Violation(
                category=hit.category,
                severity=hit.severity,  # type: ignore[arg-type]
                confidence=hit.score,
                action=hit.action,
                policy_id=hit.policy_id,
                stage="stage3_output",
                detector=hit.detector,
                reason=hit.reason[:300],
                excerpt=(hit.matched[0][:100] if hit.matched else None),
            )
            for hit in hits
        ]

        if self.mode != "assistant":
            # Passthrough mode: the "output" *is* the caller's text, already judged upstream -
            # possibly and deliberately allowed through an exemption (protective context,
            # reported harassment, help-seeking about self-harm). Re-classifying it here would
            # quietly undo those exemptions, which is how a chain ends up refusing the very
            # questions its own policy file says to answer. So only the *leakage* detectors run:
            # did the chain expose PII, the system prompt, or actionable harm instructions.
            # In assistant mode the text is new, so every stage3_output policy applies.
            leakage_policies = {"output_pii_leak", "output_system_prompt_leak", "output_unsafe_advice"}
            retained = [v for v in violations if v.policy_id in leakage_policies]
            if len(retained) != len(violations):
                log.debug(
                    "output check limited to leakage detectors in passthrough mode",
                    extra={
                        "request_id": ctx.request_id,
                        "skipped": sorted({v.policy_id or "" for v in violations if v.policy_id not in leakage_policies}),
                    },
                )
            violations = retained

        # Direct detectors that do not need a policy entry (cheap, high precision).
        lowered = output.lower()
        for marker in JAILBREAK_COMPLIANCE_MARKERS:
            if marker in lowered:
                violations.append(
                    Violation(
                        category="system_prompt_leak",
                        severity="critical",
                        confidence=0.9,
                        action="block",
                        policy_id="output_jailbreak_compliance",
                        stage="stage3_output",
                        detector="marker",
                        reason=f"output contains compliance language {marker!r}",
                    )
                )
                break
        if re.search(r"(?:```|~~~)?\s*(?:SYSTEM|GUARDRAIL)_?(?:PROMPT|MESSAGE)\s*[:=]", output, re.IGNORECASE):
            violations.append(
                Violation(
                    category="system_prompt_leak",
                    severity="critical",
                    confidence=0.95,
                    action="block",
                    policy_id="output_prompt_dump",
                    stage="stage3_output",
                    detector="marker",
                    reason="output appears to quote internal prompt configuration",
                )
            )
        hard_pii = [e for e in entities if e["entity_type"] in {"EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN", "NID_NUMBER"}]
        for entity in hard_pii[:3]:
            violations.append(
                Violation(
                    category="pii_leak",
                    severity="critical",
                    confidence=float(entity.get("score", 0.9)),
                    action="block",
                    policy_id="output_pii_leak",
                    stage="stage3_output",
                    detector=entity.get("source", "regex"),
                    reason=f"generated output contained {entity['entity_type']}",
                )
            )

        blocking = [v for v in violations if v.action == "block" or v.category in CRITICAL_LEAK_CATEGORIES]
        if not blocking:
            return True, violations
        return False, blocking

    @staticmethod
    def _pii_risk(text: str) -> float:
        spans = PiiDetector(use_presidio=False).detect(text).spans if text else []
        return 0.0 if not spans else min(1.0, 0.5 + 0.15 * len(spans))

    @staticmethod
    def _append_disclaimers(output: str, disclaimers: list[str]) -> str:
        body = output.rstrip()
        for disclaimer in disclaimers:
            text = disclaimer.strip()
            if text and text not in body:
                body = f"{body}\n\n---\n{text}"
        return body

    @staticmethod
    def _confidence(ctx: PipelineContext, check_passed: bool) -> float:
        stage1_conf = ctx.stage1.confidence if ctx.stage1 else 0.0
        stage2_conf = ctx.stage2.confidence if ctx.stage2 else 0.0
        base = max(stage1_conf, stage2_conf)
        # Confidence in an *allowed* response grows when nothing was flagged anywhere.
        return round(min(0.99, 0.55 + 0.45 * (1 - base) - (0.0 if check_passed else 0.3)), 4)

    # ------------------------------------------------------------------ reasoning
    async def _explain(self, ctx: PipelineContext, output: str) -> str:
        status = "allowed"
        if ctx.stage2 and ctx.stage2.status != "clean":
            status = ctx.stage2.status
        categories = sorted({v.category for v in (ctx.stage1.violations if ctx.stage1 else [])} | {v.category for v in (ctx.stage2.violations if ctx.stage2 else [])})
        deterministic = self._deterministic_explanation(status, categories, ctx)
        from app.pipeline.llm import get_router

        system, user = build_reasoning_prompt(status=status, categories=categories, text=ctx.text, request_id=ctx.request_id)
        try:
            response = await get_router(self.settings).chat(
                [ChatMessage("system", system), ChatMessage("user", user)],
                tier="small",
                timeout=min(10.0, self.settings.llm_request_timeout_s),
                max_tokens=160,
            )
        except (LLMError, LLMNotConfigured) as exc:
            log.debug("reasoning generation skipped", extra={"reason": str(exc)[:120]})
            return deterministic
        text = response.text.strip()
        if not text or len(text) > 600 or re.search(r"system prompt|policies\.yaml|threshold", text, re.IGNORECASE):
            return deterministic
        return text

    @staticmethod
    def _deterministic_explanation(status: str, categories: list[str], ctx: PipelineContext) -> str:
        if not categories:
            return "No policy was triggered, so the request passed through unchanged apart from text normalisation."
        joined = ", ".join(categories[:4])
        if status == "blocked":
            return f"Blocked because the input matched these policies: {joined}. The content was not forwarded to the answer model."
        if status == "rewritten":
            return f"Parts of the message matched {joined}; those fragments were masked or reworded so the request could still be answered."
        return f"Flagged for {joined} but kept within policy limits, so the response was allowed with the notes attached."

    # ------------------------------------------------------------------- failures
    def on_timeout(self, ctx: PipelineContext, exc: Exception) -> Stage3Result:
        return self._fail_closed(f"timeout after {self.timeout_s:.1f}s", "timeout")

    def on_error(self, ctx: PipelineContext, exc: Exception) -> Stage3Result:
        log.error("stage3 failed", extra={"request_id": ctx.request_id, "error": f"{exc.__class__.__name__}: {exc}"}, exc_info=exc)
        return self._fail_closed(f"{exc.__class__.__name__}: {exc}"[:200], "error")

    def _fail_closed(self, detail: str, kind: str) -> Stage3Result:
        return Stage3Result(
            stage=self.name,
            decision=Decision.BLOCK,
            output=SAFE_FALLBACKS["default"],
            output_check_passed=False,
            confidence=0.99,
            categories=["moderation_error"],
            reasons=[f"Stage 3 {kind}; a safe fallback response was returned instead of the model output ({detail})"],
            engine=f"fail-closed:{kind}",
            degraded=True,
            error=f"stage3_{kind}: {detail}"[:400],
            mode=self.mode,
        )


def _tokens(text: str) -> int:
    from app.pipeline.llm import estimate_tokens

    return estimate_tokens(text)
