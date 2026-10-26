"""Stage 2 - content-policy validation, PII masking and targeted rewriting.

Pipeline for one request::

    entities = PII scan (Presidio + BD-aware regex)  on the sanitized text
    hits     = policy bundle (stage2), fed with entities + Stage-1 verdict
    block    -> any hit whose action is `block` (short-circuits the chain)
    mask     -> deterministic placeholder substitution (spans from the PII scan)
    rewrite  -> (1) YAML `rewrite_substitutions`, then, if unsafe fragments remain,
                (2) the LLM rewriter - asked to touch *only* the flagged fragments
    verify   -> re-run the scan on the result; still-unsafe output is blocked
    disclaimer -> appended once, deduplicated

Deterministic-first matters: masking and YAML substitutions run with no model at all,
so privacy guarantees do not depend on an API being reachable. The LLM only ever
receives an already-PII-masked string, and the output verification pass is what keeps
a chatty rewriter from smuggling the content back in.

Guardrails.ai / LangChain are supported as an *additional* validator layer
(``GUARD_USE_GUARDRAILS=true``); they are optional imports so the service runs without
them, and their findings are merged into the same hit list.
"""

from __future__ import annotations

import difflib
import re
import time
from typing import Any

from app.core.logging import get_logger
from app.core.text import split_sentences
from app.pipeline.base import (
    Decision,
    PipelineContext,
    Scorecard,
    Stage,
    Stage2Action,
    Stage2Result,
    Violation,
)
from app.pipeline.llm import ChatMessage, LLMError, LLMNotConfigured
from app.pipeline.pii import PiiDetector, apply_masks
from app.pipeline.policy import RuleHit
from app.pipeline.prompts import build_rewrite_prompt

log = get_logger(__name__)

BRACKET_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_]{2,24}\]")


class Stage2PolicyStage(Stage):
    """Policy validation + transformation of the (already sanitized) text."""

    name = "stage2_policy"
    fail_closed = True

    def __init__(
        self,
        settings: Any = None,
        *,
        timeout_s: float | None = None,
        pii_detector: PiiDetector | None = None,
        use_llm_rewrite: bool | None = None,
    ) -> None:
        from app.core.config import get_settings

        self.settings = settings or get_settings()
        self.timeout_s = timeout_s if timeout_s is not None else self.settings.stage2_timeout_s
        self.pii = pii_detector or PiiDetector(use_presidio=self.settings.stage2_use_presidio)
        self.use_llm_rewrite = self.settings.stage2_allow_llm_rewrite if use_llm_rewrite is None else use_llm_rewrite

    def _default_timeout(self) -> float:
        return self.settings.stage2_timeout_s

    # ------------------------------------------------------------------- main
    async def _run(self, ctx: PipelineContext) -> Stage2Result:
        started = time.perf_counter()
        bundle = self.policy_bundle
        mode = ctx.mode.value
        text = ctx.text
        view = ctx.view

        # ---- 1. PII scan -----------------------------------------------------
        report = self.pii.detect(text)
        entities = [span.as_dict() | {"start": span.start, "end": span.end, "text": span.text} for span in report.spans]

        # ---- 2. policy evaluation -------------------------------------------
        hits = bundle.evaluate(
            stage="stage2",
            mode=mode,
            text=text,
            view=view,
            scores={
                "toxicity": ctx.stage1.toxicity if ctx.stage1 else 0.0,
                "prompt_injection": ctx.stage1.prompt_injection if ctx.stage1 else 0.0,
                "jailbreak": ctx.stage1.jailbreak if ctx.stage1 else 0.0,
                "nsfw": ctx.stage1.scores.nsfw if ctx.stage1 else 0.0,
            },
            llm_categories={c: (ctx.stage1.confidence if ctx.stage1 else 0.0) for c in (ctx.stage1.categories if ctx.stage1 else [])},
            obfuscation_score=ctx.obfuscation_score,
            obfuscation_flags=ctx.obfuscation_flags,
            script_mix=ctx.script_mix,
            entities=entities,
            stage1_result=ctx.stage1,
        )
        guardrail_hits = self._guardrails_hits(text)
        hits = hits + [h for h in guardrail_hits if h.policy_id not in {x.policy_id for x in hits}]

        blocking = [h for h in hits if h.action == "block"]
        violations = [self._violation(h) for h in hits]

        # ---- 3. block --------------------------------------------------------
        if blocking:
            severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
            blocking.sort(key=lambda h: severity_rank.get(h.severity, 9))
            top = blocking[0]
            log.info(
                "stage2 blocked request",
                extra={"request_id": ctx.request_id, "policies": [h.policy_id for h in blocking][:6], "category": top.category},
            )
            return Stage2Result(
                stage=self.name,
                status="blocked",
                decision=Decision.BLOCK,
                text="",
                original_text=text,
                violations=violations,
                actions=[],
                entities=entities,
                confidence=round(max(h.score for h in blocking), 4),
                categories=sorted({h.category for h in blocking}),
                reasons=[f"policy {h.policy_id} blocked this request ({h.reason[:160]})" for h in blocking[:4]],
                engine=self._engine_label(report, "blocked"),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        # ---- 4. mask (PII) ---------------------------------------------------
        placeholders = self._placeholders(bundle)
        masked_text, mask_actions = apply_masks(text, report.spans, placeholders)
        actions = [Stage2Action(policy_id=a["entity_type"].lower(), rule_name="pii_masking", action="mask", **self._action_kwargs(a)) for a in mask_actions]
        for span in report.spans:
            if span.entity_type == "PERSON":
                continue  # recorded, never masked on its own - see pii_personal_name
        changed = bool(mask_actions)

        # ---- 5. rewrite ------------------------------------------------------
        disclaimers: list[str] = []
        for hit in [h for h in hits if h.action == "disclaimer"]:
            policy = bundle.get(hit.policy_id)
            if policy and policy.disclaimer and policy.disclaimer not in disclaimers:
                disclaimers.append(policy.disclaimer.strip())

        rule_hits_requiring_rewrite = [h for h in hits if h.action == "rewrite"]
        rewritten = masked_text
        rewrite_engine = "none"
        if rule_hits_requiring_rewrite:
            rewritten, sub_actions = self._apply_substitutions(bundle, rule_hits_requiring_rewrite, masked_text)
            actions.extend(sub_actions)
            if sub_actions:
                rewrite_engine = "yaml-substitutions"
            # Re-run the *rules* (not the stale pre-substitution spans) to find what is
            # still unsafe: substitutions change offsets and can resolve a hit entirely.
            residual = [
                h
                for h in bundle.evaluate(
                    stage="stage2",
                    mode=mode,
                    text=rewritten,
                    view=self._view_of(rewritten),
                    scores={"toxicity": ctx.stage1.toxicity if ctx.stage1 else 0.0},
                    obfuscation_score=ctx.obfuscation_score,
                    obfuscation_flags=ctx.obfuscation_flags,
                    script_mix=ctx.script_mix,
                    entities=[span.as_dict() for span in self.pii.detect(rewritten).spans],
                    stage1_result=ctx.stage1,
                )
                if h.action == "rewrite"
            ]
            if residual:
                llm_out = await self._llm_rewrite(ctx, residual, rewritten) if self.use_llm_rewrite else None
                if llm_out is not None:
                    candidate, engine = llm_out
                    # Rewriter output is untrusted: it must survive a re-scan.
                    verified = self._verify(ctx, candidate, bundle, mode)
                    if verified:
                        rewritten = candidate
                        rewrite_engine = engine if rewrite_engine == "none" else f"{rewrite_engine}+{engine}"
                        actions.append(
                            Stage2Action(
                                policy_id=residual[0].policy_id,
                                rule_name="llm_rewrite",
                                action="rewrite",
                                original=masked_text,
                                replacement=candidate,
                                confidence=round(sum(h.score for h in residual) / len(residual), 4),
                                reason="LLM rewrote the flagged fragments; verified by re-scan",
                                detector="llm_judge",
                            )
                        )
                    else:
                        llm_out = None
                if llm_out is None:
                    # No usable rewrite (offline, refused, or failed verification):
                    # delete the flagged sentences. Losing a sentence is always safer
                    # than shipping the harmful one, and we say so in the trace.
                    dropped, dropped_ranges = self._redact_or_drop(rewritten, residual)
                    if dropped != rewritten:
                        rewritten = dropped
                        rewrite_engine = "redact" if any(mode_kind == "redact" for _s, _e, mode_kind in dropped_ranges) else "sentence-drop"
                        actions.append(
                            Stage2Action(
                                policy_id=residual[0].policy_id,
                                rule_name="unsafe_fragment_removed",
                                action="blocked_span_removed",
                                original=masked_text,
                                replacement=rewritten,
                                confidence=0.9,
                                reason=(
                                    "no safe rewrite available (LLM off/refused/failed verification); "
                                    "flagged sentences removed"
                                ),
                                detector="policy",
                            )
                        )

        changed = changed or rewritten != masked_text

        # ---- 6. final verification ------------------------------------------
        final_hits = bundle.evaluate(
            stage="stage2",
            mode=mode,
            text=rewritten,
            view=self._view_of(rewritten),
            scores={"toxicity": ctx.stage1.toxicity if ctx.stage1 else 0.0},
            obfuscation_score=ctx.obfuscation_score,
            obfuscation_flags=[],
            script_mix=ctx.script_mix,
            entities=[s.as_dict() for s in self.pii.detect(rewritten).spans],
            stage1_result=ctx.stage1,
        )
        still_blocking = [h for h in final_hits if h.action == "block"]
        if still_blocking:
            return Stage2Result(
                stage=self.name,
                status="blocked",
                decision=Decision.BLOCK,
                text="",
                original_text=text,
                violations=violations + [self._violation(h) for h in still_blocking],
                actions=actions,
                entities=entities,
                confidence=round(max(h.score for h in still_blocking), 4),
                categories=sorted({h.category for h in still_blocking}),
                reasons=[f"content still violated {h.policy_id} after rewriting; blocked" for h in still_blocking[:3]],
                engine=self._engine_label(report, rewrite_engine),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        if not rewritten.strip():
            rewritten = "[no content remained after policy filtering]"

        diff = self._diff(text, rewritten)
        # `status` describes what happened to the caller's *text*. A disclaimer is an appendix,
        # so counting it as a rewrite would make every crisis-resources response claim that we
        # edited the user's message - which is exactly the signal clients are told to surface
        # as a diff. Annotation is reported separately via `disclaimers`/`action_taken`.
        status: Any = "rewritten" if changed else "clean"
        categories = sorted({h.category for h in hits})
        result = Stage2Result(
            stage=self.name,
            status=status,  # type: ignore[arg-type]
            # Any category escalates to a human queue even when nothing was edited: a
            # self-harm disclosure that we answered with a helpline still needs review.
            decision=Decision.FLAG if categories else Decision.PASS,
            text=rewritten,
            original_text=text,
            actions=actions,
            diff=diff,
            entities=entities,
            disclaimers=disclaimers,
            violations=violations,
            confidence=round(max((h.score for h in hits), default=0.0), 4),
            categories=categories,
            scores=Scorecard(
                pii_risk=report.risk,
                legal_risk=max((h.score for h in hits if h.category in {"legal", "copyright", "fraud"}), default=0.0),
                nsfw=max((h.score for h in hits if h.category == "nsfw"), default=0.0),
                toxicity=ctx.stage1.toxicity if ctx.stage1 else 0.0,
                prompt_injection=ctx.stage1.prompt_injection if ctx.stage1 else 0.0,
                jailbreak=ctx.stage1.jailbreak if ctx.stage1 else 0.0,
                obfuscation=ctx.obfuscation_score,
                sensitive_topic=max((h.score for h in hits), default=0.0),
            ),
            reasons=self._reasons(hits, actions, status),
            engine=self._engine_label(report, rewrite_engine),
            rewrite_engine=rewrite_engine,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return result

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _action_kwargs(action_row: dict[str, Any]) -> dict[str, Any]:
        return {
            "original": action_row["original"],
            "replacement": action_row["replacement"],
            "start": action_row["start"],
            "end": action_row["end"],
            "confidence": action_row["confidence"],
            "reason": f"{action_row['entity_type']} masked for privacy",
            "detector": action_row["detector"],
        }

    def _placeholders(self, bundle: Any) -> dict[str, str]:
        mapping = bundle.mask_entities()
        # Anything the YAML does not mention still gets a sane placeholder.
        fallback = {
            "URL": "[URL]",
            "HANDLE": "[HANDLE]",
            "DATE_OF_BIRTH": "[DATE_OF_BIRTH]",
            "TAX_NUMBER": "[TAX_ID]",
            "LOCATION": "[LOCATION]",
            "PERSON": "[NAME]",
        }
        for key, value in fallback.items():
            mapping.setdefault(key, value)
        return mapping

    @staticmethod
    def _view_of(text: str) -> str:
        from app.core.text import build_detection_view

        return build_detection_view(text)

    def _engine_label(self, report: Any, rewrite_engine: str) -> str:
        bits = ["presidio+regex" if report.presidio_used else "regex"]
        if report.error:
            bits.append("pii_degraded")
        if rewrite_engine and rewrite_engine != "none":
            bits.append(f"rewrite:{rewrite_engine}")
        if self._guardrails_available():
            bits.append("guardrails")
        return "+".join(bits)

    def _apply_substitutions(self, bundle: Any, hits: list[RuleHit], text: str) -> tuple[str, list[Stage2Action]]:
        """YAML-defined deterministic substitutions; no model, fully predictable."""
        actions: list[Stage2Action] = []
        out = text
        for hit in hits:
            policy = bundle.get(hit.policy_id)
            if not policy or not policy.rewrite_substitutions:
                continue
            for rule in policy.rewrite_substitutions:
                pattern = re.compile(rule.pattern, re.IGNORECASE)
                # Replacements are literal text (no backreferences): a policy file must
                # not be able to inject group references into the output.
                replacement = rule.replacement

                def _sub(match: re.Match[str], replacement=replacement, hit=hit, policy=policy) -> str:
                    actions.append(
                        Stage2Action(
                            policy_id=policy.id,
                            rule_name=policy.name,
                            action="rewrite",
                            original=match.group(0),
                            replacement=replacement,
                            start=match.start(),
                            end=match.end(),
                            confidence=round(hit.score, 4),
                            reason=rule.description or f"{policy.category} neutralised by substitution",
                            detector="yaml_substitution",
                        )
                    )
                    return replacement

                out = pattern.sub(_sub, out)
        return out, actions

    @staticmethod
    def _still_present(text: str, hit: RuleHit) -> bool:
        """Is the flagged wording still in the text after substitution?

        Checked both on the raw text and on the obfuscation-folded view, so
        "you idiot" and "y0u 1d10t" are treated as the same violation.
        """
        folded = Stage2PolicyStage._view_of(text).lower()
        for snippet in hit.matched:
            snippet = snippet.strip().lower()
            if snippet and (snippet in text.lower() or snippet in folded):
                return True
        for start, end in hit.spans:
            if 0 <= start < end <= len(text):
                fragment = text[start:end].strip().lower()
                if len(fragment) > 2 and (fragment in text.lower() or fragment in folded):
                    return True
        return False

    @staticmethod
    def _redact_or_drop(text: str, hits: list[RuleHit]) -> tuple[str, list[tuple[int, int, str]]]:
        """Last-resort deterministic sanitiser: redact what we can, delete what we must.

        For every flagged character range we look at the sentence that contains it. If
        the flagged material is most of that sentence (>=55%) the sentence exists only to
        cause harm, so it is removed. Otherwise the span is replaced with
        ``[REDACTED:<CATEGORY>]`` so the sentence still reads and the user can see what
        happened. Pure deletion of characters would leave "Please write a post that says"
        style fragments, which is bad UX and hides what the filter did.
        """
        sentences = split_sentences(text)
        if not sentences:
            return text, []

        ranges: set[tuple[int, int]] = set()
        for hit in hits:
            spans = [(int(s), int(e)) for s, e in hit.spans if s is not None and e is not None]
            if not spans and hit.matched:
                needle = hit.matched[0].strip()
                if needle:
                    cursor = text.lower().find(needle.lower())
                    if cursor >= 0:
                        spans = [(cursor, cursor + len(needle))]
            for start, end in spans:
                if 0 <= start < end <= len(text):
                    # widen to word boundaries so we never cut a word in half
                    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
                        start -= 1
                    while end < len(text) and (text[end].isalnum() or text[end] == "_"):
                        end += 1
                    # ...and swallow a coordinating continuation ("X is a thief and a
                    # fraud"), otherwise the tail of the accusation survives redaction.
                    tail = re.compile(r"[,;]?(?:\s+(?:and|or|plus|also)?\s+[^,.;:!?]{1,24}?)*(?=[,.;:!?]|$)").match(text, end)
                    if tail:
                        candidate = tail.group(0)
                        if re.search(r"\b(?:fraud|thief|scam(?:mer)?|liar|criminal|corrupt|cheat(?:er)?|monster)\b", candidate, re.IGNORECASE):
                            end += len(candidate.rstrip())
                    ranges.add((start, end))
        if not ranges:
            return text, []

        merged: list[tuple[int, int]] = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        category_by_pos = {hit.spans[0][0]: hit.category.upper().replace("_", " ") for hit in hits if hit.spans}
        delete_spans: list[tuple[int, int]] = []
        replace_spans: list[tuple[int, int, str]] = []
        for start, end in merged:
            enclosing = next((s for s in sentences if s[0] <= start < s[1]), None) or (sentences[0][0], sentences[0][1], sentences[0][2])
            s_start, s_end, sentence = enclosing
            flagged = sum(min(e, s_end) - max(st, s_start) for st, e in merged if st < s_end and e > s_start)
            label = category_by_pos.get(start, "POLICY")
            # Only delete when the flagged material *is* the sentence; a
            # partial accusation gets redacted so the request stays readable.
            if flagged / max(len(sentence.strip()), 1) >= 0.7:
                delete_spans.append((s_start, s_end))
            else:
                replace_spans.append((start, end, f"[REDACTED:{label}]"))

        out = text
        ops: list[tuple[int, int, str]] = [(s, e, "") for s, e in delete_spans] + replace_spans
        applied: list[tuple[int, int, str]] = []
        for start, end, replacement in sorted(ops, key=lambda o: -o[0]):
            out = out[:start] + replacement + out[end:]
            applied.append((start, end, "delete" if replacement == "" else "redact"))
        out = re.sub(r"\s{2,}", " ", out)
        out = re.sub(r"\s+([,.;:!?])", r"\1", out)
        out = re.sub(r"^[,.;:!?\s]+|[,.;:!?\s]+$", "", out)
        out = re.sub(r"\(\s*\)", " ", out).strip()
        if not out:
            out = "[no content remained after policy filtering]"
        return out, sorted(applied)

    async def _llm_rewrite(self, ctx: PipelineContext, hits: list[RuleHit], text: str) -> tuple[str, str] | None:
        from app.pipeline.llm import get_router

        bundle = self.policy_bundle
        flags = []
        for hit in hits[:8]:
            policy = bundle.get(hit.policy_id)
            flags.append(
                {
                    "policy_id": hit.policy_id,
                    "snippet": (hit.matched[0] if hit.matched else "")[:200],
                    "hint": (policy.rewrite_hint if policy else None) or "de-escalate the harmful wording",
                }
            )
        system, user = build_rewrite_prompt(text, flags, request_id=ctx.request_id)
        try:
            data, response = await get_router(self.settings).chat_json(
                [ChatMessage("system", system), ChatMessage("user", user)],
                tier="small",
                timeout=min(self.timeout_s - 2, self.settings.llm_request_timeout_s),
            )
        except (LLMError, LLMNotConfigured) as exc:
            log.info("llm rewrite unavailable, keeping deterministic result", extra={"request_id": ctx.request_id, "reason": str(exc)[:160]})
            return None
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("llm rewrite crashed", extra={"request_id": ctx.request_id, "error": str(exc)[:160]})
            return None
        candidate = data.get("rewritten_text")
        if not isinstance(candidate, str) or not candidate.strip():
            return None
        if data.get("available") is False:
            return None
        return candidate.strip(), f"llm:{response.provider}"

    def _verify(self, ctx: PipelineContext, candidate: str, bundle: Any, mode: str) -> bool:
        """Rewriter output is treated as untrusted input and re-scanned."""
        if not candidate.strip() or len(candidate) > max(len(ctx.text) * 3, 200):
            return False
        if len(candidate) < max(1, int(len(ctx.text) * 0.15)):
            return False  # the rewriter deleted the user's message - not a rewrite
        report = self.pii.detect(candidate)
        view = self._view_of(candidate)
        hits = bundle.evaluate(
            stage="stage2",
            mode=mode,
            text=candidate,
            view=view,
            scores={"toxicity": max((ctx.stage1.toxicity if ctx.stage1 else 0.0) * 0.5, 0.0)},
            obfuscation_score=0.0,
            obfuscation_flags=[],
            script_mix=ctx.script_mix,
            entities=[s.as_dict() for s in report.spans],
        )
        hard = [h for h in hits if h.action == "block" or h.category in {"hate", "violence", "child_safety", "self_harm", "sexual"}]
        if hard and any(h.score >= 0.5 for h in hard):
            return False
        # PII must not reappear through the rewriter
        leaked = [s for s in report.spans if s.entity_type in {"EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "NID_NUMBER", "US_SSN"}]
        if leaked:
            log.info("rewrite leaked PII; rejected", extra={"request_id": ctx.request_id, "types": [s.entity_type for s in leaked][:5]})
            return False
        injection_view = view.lower()
        if re.search(r"ignore (?:all )?previous instructions|reveal .*system prompt", injection_view):
            return False
        return True

    def _guardrails_available(self) -> bool:
        if not getattr(self.settings, "use_guardrails", False):
            return False
        try:  # pragma: no cover - optional dependency
            import guardrails  # noqa: F401  # type: ignore[import-not-found]

            return True
        except ImportError:
            return False

    def _guardrails_hits(self, text: str) -> list[RuleHit]:
        """Optional Guardrails.ai validators, merged into the same hit list."""
        if not self._guardrails_available():
            return []
        hits: list[RuleHit] = []
        try:  # pragma: no cover - exercised only with the extra installed
            from guardrails.validators import Provenity, ToxicLanguage  # type: ignore[import-not-found]

            for validator_name, validator in (("toxic_language", ToxicLanguage()), ("provenity", Provenity())):
                try:
                    passed, metadata = validator.validate(text, {})
                except Exception:
                    continue
                if not passed:
                    fail_state = getattr(metadata, "fail_state", None) or {}
                    hits.append(
                        RuleHit(
                            policy_id=f"guardrails:{validator_name}",
                            category="toxic" if "toxic" in validator_name else "provenance",
                            severity="medium",
                            action="flag",
                            score=float(fail_state.get("score", 0.7) or 0.7),
                            reason=f"guardrails.{validator_name} flagged the text",
                            detector="guardrails",
                        )
                    )
        except Exception as exc:  # pragma: no cover
            log.debug("guardrails validators unavailable", extra={"error": str(exc)[:120]})
        return hits

    @staticmethod
    def _violation(hit: RuleHit) -> Violation:
        return Violation(
            category=hit.category,
            severity=hit.severity,  # type: ignore[arg-type]
            confidence=hit.score,
            action=hit.action,
            policy_id=hit.policy_id,
            stage="stage2",
            detector=hit.detector,
            reason=hit.reason[:400],
            excerpt=(hit.matched[0][:120] if hit.matched else None),
            span=tuple(hit.spans[0]) if hit.spans else None,
        )

    @staticmethod
    def _reasons(hits: list[RuleHit], actions: list[Stage2Action], status: str) -> list[str]:
        out = [f"policy {h.policy_id}: {h.reason[:140]}" for h in hits[:6]]
        if actions:
            kinds: dict[str, int] = {}
            for action in actions:
                kinds[action.action] = kinds.get(action.action, 0) + 1
            out.append("transformations: " + ", ".join(f"{k}x{v}" for k, v in sorted(kinds.items())))
        if status == "clean" and not out:
            out.append("no content policy violations")
        return out

    @staticmethod
    def _diff(before: str, after: str) -> list[dict[str, Any]]:
        """Word-level diff so the UI can show exactly what changed and why."""
        if before == after:
            return []
        before_tokens = re.split(r"(\s+)", before)
        after_tokens = re.split(r"(\s+)", after)
        entries: list[dict[str, Any]] = []
        matcher = difflib.SequenceMatcher(a=before_tokens, b=after_tokens, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            removed = "".join(before_tokens[i1:i2]).strip()
            added = "".join(after_tokens[j1:j2]).strip()
            if not removed and not added:
                continue
            entries.append(
                {
                    "op": tag,  # replace | delete | insert
                    "removed": removed[:240],
                    "added": added[:240],
                    "before_index": i1,
                    "after_index": j1,
                }
            )
        return entries[:60]

    # ------------------------------------------------------------ failure paths
    def on_timeout(self, ctx: PipelineContext, exc: Exception) -> Stage2Result:
        return self._fail_closed(ctx, f"timeout after {self.timeout_s:.1f}s", "timeout")

    def on_error(self, ctx: PipelineContext, exc: Exception) -> Stage2Result:
        log.error("stage2 failed", extra={"request_id": ctx.request_id, "error": f"{exc.__class__.__name__}: {exc}"}, exc_info=exc)
        return self._fail_closed(ctx, f"{exc.__class__.__name__}: {exc}"[:200], "error")

    def _fail_closed(self, ctx: PipelineContext, detail: str, kind: str) -> Stage2Result:
        return Stage2Result(
            stage=self.name,
            status="blocked",
            decision=Decision.BLOCK,
            text="",
            original_text=ctx.text,
            confidence=0.99,
            categories=["moderation_error"],
            reasons=[f"Stage 2 {kind}; request blocked fail-closed ({detail})"],
            engine=f"fail-closed:{kind}",
            degraded=True,
            error=f"stage2_{kind}: {detail}"[:400],
        )
