"""Output formatter - the stable ``v1`` API envelope.

This is the only place the wire format is defined. Two rules are enforced here rather
than trusted to callers:

* **A blocked request never echoes the input.** For ``status="blocked"`` the payload
  carries a friendly ``user_message`` and category labels only; the offending text is
  not repeated, because the response is often rendered straight into a UI or pasted
  into another model's context.
* **Unknown/unsafe output degrades to a fixed message**, not to the raw model text
  (``safe_output`` is empty for blocks, so a client that ignores ``status`` still
  cannot display the harmful content).

``schema_version`` ("v1") is part of every response so a future breaking change can be
introduced as ``v2`` with clients pinning the version they understand.
"""

from __future__ import annotations

from typing import Any

from app.pipeline.base import PipelineResult, Status

SCHEMA_VERSION = "v1"
POLICY_SPEC_URL = "https://example.com/guardrail-moderation/docs/policies/v1"

#: status -> (http_status, headline)
STATUS_META: dict[str, tuple[int, str]] = {
    "allowed": (200, "Content accepted"),
    "rewritten": (200, "Content accepted after safety rewriting"),
    "blocked": (200, "Content blocked by moderation policy"),
}

BLOCKED_MESSAGES: dict[str, str] = {
    "moderation_error": (
        "We could not check this message safely, so it was not sent. Please try again; "
        "if it keeps happening, contact support with the request id."
    ),
    "prompt_injection": (
        "That message asked us to change how we operate (for example to reveal our "
        "instructions). We can't do that, but we're happy to keep helping with your task."
    ),
    "jailbreak": (
        "We can't switch into an unfiltered mode. Ask us for the safe version of what "
        "you need and we'll do our best."
    ),
    "hate": (
        "This message was blocked because it contains hate speech. We can help you "
        "rephrase it into something that doesn't attack a group of people."
    ),
    "harassment": (
        "This message was blocked because it targets a person with abuse. Tell us what "
        "you want to achieve and we'll help you say it constructively."
    ),
    "self_harm": (
        "It sounds like you're carrying something heavy. We can't help with self-harm, "
        "but please talk to someone now - in Bangladesh: KaanchBCP 106 (24/7, free) or "
        "999. If you're elsewhere, contact your local crisis line."
    ),
    "violence": (
        "We can't help with plans to harm people or property. If someone is in danger, "
        "contact your local emergency number."
    ),
    "child_safety": (
        "This request was blocked. If a child may be at risk, contact Bangabandhu "
        "Children's Helpline 10980 or your local law enforcement."
    ),
    "sexual": "This message was blocked because it contains explicit sexual content.",
    "weapons": "We can't help with weapons. We can help with safety or reporting concerns instead.",
    "drugs": "We can't help with buying or selling controlled drugs. We can share harm-reduction or treatment information.",
    "fraud": "This message was blocked because it looks like fraud or scam infrastructure.",
    "doxxing": "This message was blocked because it exposes several personal identifiers together.",
    "pii": "This message was blocked because it exposes personal data about someone. We can help you write the message with the details redacted.",
    "legal": "This message was blocked because it makes an unverified accusation about a named person.",
    "copyright": "We can't reproduce protected material. We can summarise or discuss it instead.",
    "dangerous_goods": "This message was blocked because it involves illegal commerce.",
}
DEFAULT_BLOCKED_MESSAGE = (
    "This message was blocked by the content policy. You can rephrase it and try again, "
    "or ask us for help with a safer version of your request."
)


def friendly_block_message(categories: list[str], reason: str | None) -> str:
    """Pick the most specific guidance we have for this block."""
    if reason and reason in BLOCKED_MESSAGES:
        return BLOCKED_MESSAGES[reason]
    for category in categories or []:
        if category in BLOCKED_MESSAGES:
            return BLOCKED_MESSAGES[category]
    return DEFAULT_BLOCKED_MESSAGE


def _summary(result: PipelineResult) -> dict[str, Any]:
    reasons: list[str] = []
    for violation in result.violations[:8]:
        detail = violation.reason or violation.category
        reasons.append(f"[{violation.stage}] {violation.policy_id or violation.category}: {detail}"[:280])
    if not reasons:
        reasons.append(f"pipeline decision: {result.reason or 'no violations detected'}")
    return {
        "categories": sorted({v.category for v in result.violations} | set(result.categories)),
        "scores": {k: round(float(v), 4) for k, v in (result.scores or {}).items()},
        "reasons": reasons,
        "violation_count": len(result.violations),
        "top_severity": _top_severity(result),
        "blocked_by": (result.trace.short_circuited_at if result.trace else None),
    }


def _top_severity(result: PipelineResult) -> str | None:
    rank = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    severities = [v.severity for v in result.violations]
    return max(severities, key=lambda s: rank.get(s, 0)) if severities else None


def _policy_detail(result: PipelineResult) -> list[dict[str, Any]]:
    """Per-violation detail for auditors; never includes the harmful excerpt on blocks."""
    out: list[dict[str, Any]] = []
    for violation in result.violations[:12]:
        entry: dict[str, Any] = {
            "category": violation.category,
            "severity": violation.severity,
            "action": violation.action,
            "policy_id": violation.policy_id,
            "stage": violation.stage,
            "detector": violation.detector,
            "confidence": round(violation.confidence, 4),
            "reason": violation.reason[:300],
        }
        if violation.excerpt and result.status is not Status.BLOCKED:
            entry["excerpt"] = violation.excerpt[:160]
        out.append(entry)
    return out


def _transformations(result: PipelineResult) -> list[dict[str, Any]]:
    return list(result.transformations or [])


def format_response(
    result: PipelineResult,
    *,
    include_trace: bool = False,
    postprocess_report: dict[str, Any] | None = None,
    request_url: str | None = None,
) -> dict[str, Any]:
    """Build the ``v1`` response envelope."""
    status = result.status.value
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "request_id": result.request_id,
        "status": status,
        "safe_output": "" if status == "blocked" else (result.safe_output or ""),
        "moderation_summary": _summary(result),
        "action_taken": result.action_taken,
        "confidence": round(float(result.confidence), 4),
        "latency_ms": round(float(result.latency_ms), 2),
        "created_at": result.created_at,
        "policy": {
            "version": result.policy_version,
            "mode": result.mode,
            "spec_url": POLICY_SPEC_URL,
        },
        "models": result.model_versions,
        "usage": {
            **(result.token_usage or {}),
            "estimated_cost_usd": result.estimated_cost_usd,
        },
        "flags": {
            "degraded": result.degraded,
            "fail_closed": result.fail_closed,
            "reason": result.reason,
        },
    }
    if result.disclaimers:
        payload["disclaimers"] = result.disclaimers
    if status == "blocked":
        payload["user_message"] = result.user_message or friendly_block_message(payload["moderation_summary"]["categories"], result.blocked_reason)
        payload["blocked"] = {
            "reason": result.blocked_reason or "policy_violation",
            "categories": payload["moderation_summary"]["categories"],
            # "retry_allowed" answers exactly one client question: could resending *this*
            # request plausibly succeed? A content block is deterministic, so no. A fail-closed
            # block is an accident inside our process, so yes - and the caller should back off
            # briefly rather than give up on the user's text.
            "retry_allowed": result.blocked_reason == "moderation_error",
            "detail": _policy_detail(result),
        }
        payload["safe_output"] = ""
        payload["echo"] = None  # explicit: we never echo blocked content
    else:
        payload["moderation_summary"]["transformations"] = _transformations(result)
    if include_trace and result.trace is not None:
        payload["pipeline_trace"] = result.trace.as_dict()
    if postprocess_report is not None:
        payload["storage"] = postprocess_report
    if request_url:
        payload["_links"] = {
            "self": request_url,
            "log": f"/logs/{result.request_id}",
            "explain": f"/explain/{result.request_id}",
        }
    return payload


def openapi_examples() -> dict[str, Any]:
    """Response examples for each status, wired into the OpenAPI docs."""
    return {
        "allowed": {
            "summary": "Clean input, passed through unchanged",
            "value": format_response(_example_result(Status.ALLOWED, "none", "no_violations")),
        },
        "rewritten": {
            "summary": "PII masked and a toxic fragment reworded",
            "value": format_response(_example_result(Status.REWRITTEN, "masked_pii_and_rewrote", "policy_transformations_applied")),
        },
        "blocked": {
            "summary": "Prompt injection blocked at Stage 1 (no content echoed)",
            "value": format_response(_example_result(Status.BLOCKED, "blocked", "prompt_injection")),
        },
        "blocked_fail_closed": {
            "summary": "Moderation could not run - fail closed",
            "value": format_response(_example_result(Status.BLOCKED, "blocked", "moderation_error", fail_closed=True)),
        },
    }


def _example_result(status: Status, action: str, reason: str, *, fail_closed: bool = False) -> PipelineResult:
    from app.pipeline.base import Decision, Violation

    blocked = status is Status.BLOCKED
    return PipelineResult(
        request_id="0f2c1f0a-2c34-4c9a-9d31-demo",
        status=status,
        decision=Decision.BLOCK if blocked else Decision.PASS,
        safe_output="" if blocked else "Please send me the revised invoice for March.",
        original_text="" if blocked else "Email me at a@b.com, you idiot, and send the revised invoice.",
        action_taken=action,  # type: ignore[arg-type]
        confidence=0.96 if not blocked else 0.93,
        latency_ms=84.0,
        mode="balanced",
        policy_version="1.0.0+ab12cd34",
        model_versions={"stage1_judge": "gemini-2.0-flash-lite", "stage3_model": "none", "policy_version": "1.0.0+ab12cd34"},
        reason=reason,
        blocked_reason=reason if blocked else None,
        user_message=friendly_block_message(["prompt_injection"], reason) if blocked else None,
        fail_closed=fail_closed,
        categories=["prompt_injection"] if blocked else [],
        violations=[]
        if not blocked
        else [
            Violation(
                category="prompt_injection",
                severity="critical",
                confidence=0.93,
                action="block",
                policy_id="prompt_injection_classic",
                stage="stage1",
                detector="regex+llm_judge",
                reason="pattern match: ignore previous instructions",
            )
        ],
        scores={"toxicity": 0.0, "prompt_injection": 0.93 if blocked else 0.02},
        token_usage={"prompt_tokens": 210, "completion_tokens": 0, "provider": "gemini", "model": "gemini-2.0-flash-lite"},
    )
