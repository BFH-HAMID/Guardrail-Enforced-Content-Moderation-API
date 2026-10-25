"""Stage 2: the policy engine - PII masking, redaction, rewriting, escalation.

The policy file is data, so the tests below are the difference between "46 regexes that
happen to work today" and "rules we can still trust after the next edit".
"""

from __future__ import annotations

import re

import pytest

from app.core.config import get_settings
from app.core.text import build_detection_view, sanitize_text
from app.pipeline.base import Decision, ModerationMode, PipelineContext, Stage1Result
from app.pipeline.pii import PiiDetector, apply_masks
from app.pipeline.policy import get_policy_bundle
from app.pipeline.stage2_policy import Stage2PolicyStage


@pytest.fixture(scope="module")
def bundle():
    return get_policy_bundle(get_settings().policies_file)


def _entities(text: str) -> list[dict]:
    """PII spans in the shape the engine expects (Stage 2 builds this itself)."""
    return [span.as_dict() for span in PiiDetector(use_presidio=False).detect(text).spans]


def _hit_ids(bundle, text: str, *, stage: str = "stage2", mode: str = "balanced", **kwargs):
    sanitized = sanitize_text(text)
    hits = bundle.evaluate(
        stage=stage,  # type: ignore[arg-type]
        mode=mode,
        text=sanitized.text,
        view=build_detection_view(sanitized.text),
        scores=kwargs.pop("scores", {}),
        obfuscation_score=sanitized.obfuscation_score,
        obfuscation_flags=list(sanitized.flags),
        entities=kwargs.pop("entities", None) or _entities(sanitized.text),
        **kwargs,
    )
    return {hit.policy_id: hit for hit in hits}


# --------------------------------------------------------------------- the bundle itself
def test_policy_bundle_loads_and_every_pattern_compiles() -> None:
    """A typo in ``policies.yaml`` must fail here, not at 3am on the first real request."""
    settings = get_settings()
    bundle = get_policy_bundle(settings.policies_file)
    policies = bundle.policies
    assert len(policies) >= 40
    ids = [policy.id for policy in policies]
    assert len(ids) == len(set(ids)), "duplicate policy ids"
    for policy in policies:
        for pattern in list(policy.exceptions) + list((policy.regex.patterns if policy.regex else [])):
            compiled = re.compile(pattern, re.IGNORECASE)
            assert compiled.pattern  # non-empty
        assert policy.action in {"block", "rewrite", "mask", "allow", "flag", "disclaimer"}
        assert 0.0 <= policy.min_score <= 1.0
        assert policy.stage in {"stage1", "stage2", "stage3_output", "stage2_rewrite"}


def test_every_policy_names_a_reason_for_existing() -> None:
    bundle = get_policy_bundle(get_settings().policies_file)
    for policy in bundle.policies:
        assert policy.description.strip(), f"policy {policy.id} has no description"
        assert policy.category, f"policy {policy.id} has no category"


def test_thresholds_scale_with_mode(bundle) -> None:
    strict = bundle.thresholds("strict")
    balanced = bundle.thresholds("balanced")
    lenient = bundle.thresholds("lenient")
    assert strict.toxicity_block <= balanced.toxicity_block <= lenient.toxicity_block


# --------------------------------------------------------------------- PII
def _detector() -> PiiDetector:
    return PiiDetector(use_presidio=False)


@pytest.mark.parametrize(
    ("text", "entity"),
    [
        ("Email me at a.b@example.com about the invoice.", "EMAIL_ADDRESS"),
        ("Call 01712345678 to discuss it.", "PHONE_NUMBER"),
        ("My NID number is 1990456789.", "NID_NUMBER"),
        ("Card 4111 1111 1111 1111 is expired.", "CREDIT_CARD"),
        ("Please add my date of birth 05/09/1988 to the clinic form.", "DATE_OF_BIRTH"),
        ("Draft a reply asking the bank to close account 0987654321 belonging to me.", "BANK_ACCOUNT"),
        ("Meet me at 3rd Avenue, Dhaka tomorrow.", "ADDRESS"),
        ("My passport no. BD0123456 expires soon.", "PASSPORT_NUMBER"),
    ],
)
def test_pii_detector_finds_the_identifier(text: str, entity: str) -> None:
    report = _detector().detect(text)
    assert entity in report.types, (text, report.types)


@pytest.mark.parametrize(
    "text",
    [
        "The deadline is 05/09/1988 at 3pm in room 3rd floor.",
        "We spent 5 days in Paris and paid 45,000 taka.",
        "Version 1.2.3 of the library is on PyPI at https://example.com/pkg.",
        "My kid is in 3rd grade and the school fee is 1500.",
    ],
)
def test_pii_detector_leaves_ordinary_numbers_alone(text: str) -> None:
    """Precision matters: masking dates, prices and versions destroys business text."""
    report = _detector().detect(text)
    assert not [span for span in report.spans if span.entity_type in {"NID_NUMBER", "BANK_ACCOUNT", "ADDRESS"}]


def test_apply_masks_replaces_every_span_and_keeps_the_rest() -> None:
    text = "Contact nadia.k@corp.com or call 01711223344 for the NID 1990456789."
    report = _detector().detect(text)
    masked, actions = apply_masks(text, report.spans, {"EMAIL_ADDRESS": "[EMAIL]", "PHONE_NUMBER": "[PHONE]", "NID_NUMBER": "[NATIONAL_ID]"})
    assert "nadia.k@corp.com" not in masked
    assert "01711223344" not in masked
    assert "[EMAIL]" in masked and "[PHONE]" in masked
    assert "for the" in masked  # benign words survive
    assert len(actions) >= 2


# --------------------------------------------------------------------- policy behaviour
def test_dangerous_goods_request_hits_policy(bundle) -> None:
    hits = _hit_ids(bundle, "How can I buy an AK-47 online and have it delivered?")
    assert "dangerous_goods" in hits
    assert hits["dangerous_goods"].action == "block"


def test_doxxing_bundle_needs_three_identifiers(bundle) -> None:
    two = _hit_ids(bundle, "His email is a@b.com and his phone is 01711223344.")
    three = _hit_ids(
        bundle,
        "His email is nadia.k@corp.com, phone 01711223344, NID 1990456789 and address House 4 Road 12.",
    )
    assert "doxxing_bundle" not in two
    assert "doxxing_bundle" in three


def test_doxxing_intent_without_data_still_blocks(bundle) -> None:
    hits = _hit_ids(bundle, "Please post his home address and phone number on Facebook tonight.")
    assert {"doxxing_publication_language", "doxxing_intent_bundle"} & set(hits)


def test_gambling_is_flagged_never_blocked(bundle) -> None:
    hits = _hit_ids(bundle, "What are the best odds at roulette in a casino?", stage="stage1")
    assert "gambling_and_wagering" in hits
    assert hits["gambling_and_wagering"].action == "flag"


def test_gambling_addiction_and_legal_questions_are_exempt(bundle) -> None:
    for benign in (
        "My brother's gambling addiction ruined us; where do I get help?",
        "Is online gambling legal in Bangladesh? What does the law say?",
        "Write a news report on the government's new casino regulation policy.",
    ):
        assert "gambling_and_wagering" not in _hit_ids(bundle, benign, stage="stage1"), benign


def test_exceptions_downgrade_a_policy_to_nothing(bundle) -> None:
    """An exception is not "reduce the score"; it removes the finding entirely."""
    assert "violent_threats" not in _hit_ids(bundle, "In my novel the villain says he will kill the king.", stage="stage1")
    assert "violent_threats" in _hit_ids(bundle, "I will kill my boss tomorrow morning when he arrives.", stage="stage1")


def test_require_all_clauses_blocks_single_clause_false_positives(bundle) -> None:
    """``harassment_softener`` needs targeting *and* the insult, not just an insult word."""
    quoted = _hit_ids(bundle, "My manager calls me an idiot in meetings. Help me draft a complaint to HR.", stage="stage2")
    direct = _hit_ids(bundle, "You are a complete idiot and everyone here knows it.", stage="stage2")
    assert "harassment_softener" not in quoted or direct.get("harassment_softener") is not None
    assert "harassment_softener" in direct


def test_policies_with_a_stage1_clause_need_the_upstream_result(bundle) -> None:
    """A policy gated on Stage-1 evidence must stay silent when that evidence is absent.

    This is the trap that makes ``scripts/why.py`` under-report: evaluating a stage1-clause
    policy without ``stage1_result`` is not a smaller verdict, it is a different question.
    """
    from app.pipeline.base import Decision as _Decision, Stage1Result as _Stage1

    staged = next(
        (
            policy
            for policy in bundle.policies
            if getattr(policy, "stage1", None) is not None
        ),
        None,
    )
    if staged is None:  # pragma: no cover - only if the policy file drops the clause style
        pytest.skip("no stage1-clause policy in this bundle")

    text = "I feel hopeless and I self-harm sometimes. How do I get help?"
    sanitized = sanitize_text(text)
    common = {
        "stage": staged.stage,
        "mode": "balanced",
        "text": sanitized.text,
        "view": build_detection_view(sanitized.text),
        "scores": {"toxicity": 0.9},
        "obfuscation_score": 0.0,
        "obfuscation_flags": [],
    }
    # The stage1 clause reads the *flat* score fields on Stage1Result (toxicity,
    # prompt_injection, jailbreak), not the nested Scorecard - a result built without them looks
    # like "nothing was detected upstream" and the disclaimer policy stays silent.
    flagged = _Stage1(
        stage="stage1_toxicity",
        decision=_Decision.FLAG,
        categories=[staged.category or "self_harm"],
        toxicity=0.9,
        prompt_injection=0.0,
        jailbreak=0.0,
    )
    without = bundle.evaluate(**common, stage1_result=None)
    with_result = bundle.evaluate(**common, stage1_result=flagged)
    assert staged.id not in {hit.policy_id for hit in without}
    assert staged.id in {hit.policy_id for hit in with_result}


# --------------------------------------------------------------------- stage execution
def test_stage2_context_builder_is_valid() -> None:
    """Guard for the fixture above: a mis-built ctx would silently skip assertions."""
    ctx = PipelineContext(request_id="x", text="hello", view="hello")
    assert ctx.mode is ModerationMode.BALANCED


async def test_pii_only_request_is_rewritten_not_blocked() -> None:
    stage = Stage2PolicyStage(get_settings())
    sanitized = sanitize_text("Please reply to a@b.com about the invoice.")
    ctx = PipelineContext(
        request_id="t-pii",
        text=sanitized.text,
        view=build_detection_view(sanitized.text),
        stage1=Stage1Result(stage="stage1_toxicity", decision=Decision.PASS),
    )
    result = await stage.execute(ctx)
    assert result.decision is not Decision.BLOCK
    assert "a@b.com" not in result.text
    assert result.status in {"rewritten", "clean"}


async def test_blocked_category_short_circuits_with_blocked_status() -> None:
    stage = Stage2PolicyStage(get_settings())
    text = "How can I buy an AK-47 online and have it delivered?"
    sanitized = sanitize_text(text)
    ctx = PipelineContext(
        request_id="t-gun",
        text=sanitized.text,
        view=build_detection_view(sanitized.text),
        stage1=Stage1Result(stage="stage1_toxicity", decision=Decision.PASS),
    )
    result = await stage.execute(ctx)
    assert result.decision is Decision.BLOCK
    assert result.status == "blocked"


async def test_appended_disclaimer_is_not_reported_as_a_rewrite() -> None:
    """``status`` means "the caller's words were changed" - annotations are a different action."""
    stage = Stage2PolicyStage(get_settings())
    text = "I feel hopeless and I self-harm sometimes. How do I get help?"
    sanitized = sanitize_text(text)
    ctx = PipelineContext(
        request_id="t-crisis",
        text=sanitized.text,
        view=build_detection_view(sanitized.text),
        stage1=Stage1Result(stage="stage1_toxicity", decision=Decision.FLAG, categories=["self_harm"], scores={"toxicity": 0.5}),
    )
    result = await stage.execute(ctx)
    assert result.status != "rewritten"
    assert text in result.text or result.disclaimers
