"""Structural invariants of `config/policies.yaml`.

These are not "does the rule detect X" tests (test_stage2.py covers that); they guard the traps that
made a rule *silently* stop working in this project's history:

* a `\\b` written next to Bangla characters never matches (vowel signs are `\\W`), so a rule looks
  correct in review and does nothing in production;
* a `\\uXXXX` escape loads fine in YAML and then matches text the sanitizer has already NFKC'd away;
* a `requires_policy_hits` pointing at a renamed policy disables a dependent rule with no error.

All three passed CI before they were caught. This file is why they should not come back.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "config" / "policies.yaml"

# anything ASCII-letter/digit on both sides is fine; the trap is a boundary *touching* non-ASCII
_BOUNDARY_TOUCHING_NON_ASCII = re.compile(r"\((?:\?<!)?\\b(?=[^\x00-\x7f])|(?<=[^\x00-\x7f])\\b\)")


@pytest.fixture(scope="module")
def doc() -> dict:
    return yaml.safe_load(POLICIES.read_text(encoding="utf-8"))


def _patterns(policy: dict) -> list[str]:
    return list((policy.get("regex") or {}).get("patterns") or [])


def test_top_level_shape(doc: dict) -> None:
    assert set(doc) == {"version", "name", "modes", "policies"}, "top-level keys of the policy file changed shape"
    assert set(doc["modes"]) == {"strict", "balanced", "lenient"}
    for mode, preset in doc["modes"].items():
        for key in ("toxicity_block", "injection_block", "jailbreak_block", "sensitive_topic_flag", "stage1_consensus"):
            assert key in preset, f"{mode} preset missing {key}"
        assert preset["stage1_consensus"] in {"any", "both"}
        assert 0.0 < preset["toxicity_block"] <= 1.0


def test_ids_are_unique_and_stable(doc: dict) -> None:
    ids = [policy["id"] for policy in doc["policies"]]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[a-z0-9_]{3,48}", i) for i in ids), "ids appear in client responses; keep them snake_case"
    # renaming an id breaks cached clients and corpus rows; the corpus is the canary
    corpus = (ROOT / "tests" / "data" / "moderation_testset.jsonl").read_text(encoding="utf-8")
    assert len(ids) >= 40
    assert "self_harm" in corpus


def test_every_pattern_compiles_and_none_use_unicode_escapes(doc: dict) -> None:
    offenders = []
    for policy in doc["policies"]:
        for pattern in _patterns(policy) + list(policy.get("exceptions") or []):
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                offenders.append(f"{policy['id']}: {exc}")
            if "\\u" in pattern or re.search(r"\\U[0-9a-fA-F]", pattern):
                # A literal-script character survives NFKC; an escape may not match what the
                # sanitizer produced, and it is unreadable in review either way.
                offenders.append(f"{policy['id']}: uses a \\uXXXX escape instead of literal text")
    assert not offenders, "\n".join(offenders)


def test_no_word_boundary_touching_non_ascii(doc: dict) -> None:
    """`\\b` beside Bangla/Devanagari is a silent no-op; use indic_safe_pattern instead."""
    offenders = []
    for policy in doc["policies"]:
        for pattern in _patterns(policy) + list(policy.get("exceptions") or []):
            if _BOUNDARY_TOUCHING_NON_ASCII.search(pattern):
                offenders.append(f"{policy['id']}: {pattern[:70]}")
    assert not offenders, "word boundary adjacent to non-ASCII text (will never match): \n  " + "\n  ".join(offenders)


def test_requires_policy_hits_reference_existing_policies(doc: dict) -> None:
    ids = {policy["id"] for policy in doc["policies"]}
    dangling = [(p["id"], dep) for p in doc["policies"] for dep in (p.get("requires_policy_hits") or []) if dep not in ids]
    assert not dangling, f"policies depending on a non-existent rule: {dangling}"


def test_stage3_output_policies_are_blocking(doc: dict) -> None:
    """A stage-3 guard that only *flags* is decoration: by then the answer is already written."""
    wrong = [
        p["id"]
        for p in doc["policies"]
        if p["stage"] == "stage3_output" and p["action"] not in {"block", "redact", "mask"}
    ]
    assert not wrong, wrong


def test_mask_and_rewrite_policies_have_something_to_act_on(doc: dict) -> None:
    for policy in doc["policies"]:
        action = policy["action"]
        if action == "mask":
            assert policy.get("pii_entities") or policy.get("entities") or _patterns(policy), policy["id"]
        if action in {"rewrite", "redact"}:
            sources = ("regex", "lexicon", "classifier", "llm_categories", "obfuscation", "payloads", "script", "entities", "stage1")
            has_something = bool(policy.get("rewrite_substitutions")) or any(policy.get(name) for name in sources)
            assert has_something, f"{policy['id']} rewrites but no clause could ever match"
        if action == "disclaimer":
            assert policy.get("disclaimer"), f"{policy['id']} has action=disclaimer with no disclaimer text"


def test_disclaimers_are_actionable_not_generic(doc: dict) -> None:
    """The self-harm and medical disclaimers must carry an actual route to help."""
    for policy in doc["policies"]:
        text = (policy.get("disclaimer") or "").lower()
        if not text:
            continue
        if "self-harm" in text or "suicide" in text or "harming yourself" in text:
            assert any(token in text for token in ("106", "999", "hotline", "call")), f"{policy['id']}: crisis text without a number"


def test_actions_are_consistent_with_severity(doc: dict) -> None:
    """Nothing `critical` may be a `flag`, and nothing `info` may block."""
    for policy in doc["policies"]:
        severity, action = policy.get("severity"), policy.get("action")
        if severity == "critical":
            assert action in {"block", "mask"}, f"{policy['id']} is critical but action={action}"
        if severity == "info":
            assert action in {"flag", "disclaimer", "allow", "rewrite"}, f"{policy['id']} is info but action={action}"


def test_pii_policies_cover_the_bangladesh_formats(doc: dict) -> None:
    """The point of this deployment is BD identifier formats; if a rule disappears, notice."""
    ids = {policy["id"] for policy in doc["policies"]}
    assert {"pii_email", "pii_phone", "pii_national_id", "pii_payment_card", "pii_home_address", "pii_date_of_birth"} <= ids
    by_id = {policy["id"]: policy for policy in doc["policies"]}
    blob = yaml.safe_dump(by_id["pii_national_id"])
    for token in ("NID", "passport", "birth"):
        assert token.lower() in blob.lower(), f"pii_national_id no longer mentions {token}"


def test_doxxing_is_layered(doc: dict) -> None:
    """One identifier is ordinary text; three is a doxxing bundle. The layering must survive edits."""
    by_id = {policy["id"]: policy for policy in doc["policies"]}
    assert by_id["doxxing_bundle"]["entities"]["min_count"] >= 3
    intent = by_id["doxxing_intent_bundle"]
    assert intent["entities"]["min_count"] >= 2 and intent["require_all_clauses"] is True
    assert "doxxing_publication_language" in by_id


def test_exceptions_are_narrow(doc: dict) -> None:
    """An exception like `.*` or `^` disarms a rule completely; keep them phrase-shaped."""
    for policy in doc["policies"]:
        for exception in policy.get("exceptions") or []:
            stripped = exception.strip()
            assert len(stripped) >= 12, f"{policy['id']}: exception too short to be meaningful: {stripped!r}"
            assert stripped not in {".*", "^", "$", "\\w+"}
            assert not re.fullmatch(r"[.\*\+\?]+", stripped)


def test_yaml_file_has_no_duplicated_backslash_misunderstandings(doc: dict) -> None:
    """A pattern that arrives here as `\\\\b` (escaped backslash + literal b) matches no word boundary.

    This catches the inverse of the double-quote trap: after a bad scripted edit, `\\b` in the file
    becomes a literal backslash followed by 'b'.
    """
    suspicious = []
    for policy in doc["policies"]:
        for pattern in _patterns(policy):
            if "\\\\b" in pattern or "\\\\s" in pattern or "\\\\w" in pattern:
                suspicious.append(f"{policy['id']}: {pattern[:60]}")
    assert not suspicious, f"double-escaped regex classes in {POLICIES.name}: {suspicious}"


def test_the_header_warning_survives_edits(doc: dict) -> None:
    """The file's own header carries the rules contributors must not rediscover the hard way."""
    head = POLICIES.read_text(encoding="utf-8")[:2600].lower()
    assert "bengali" in head or "bangla" in head, "the header no longer warns about Bengali/\\b"
    assert "\\u" in head or "unicode escape" in head, "the header no longer bans \\uXXXX escapes"


def test_ast_sanity_of_the_corpus_generator() -> None:
    """scripts/dataset_corpus.py is data-as-code; a syntax error there silently freezes an old JSONL."""
    source = (ROOT / "scripts" / "dataset_corpus.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = [node for node in tree.body if isinstance(node, ast.AnnAssign) or isinstance(node, ast.Assign)]
    assert len(assignments) > 10, "the corpus module looks truncated"
    tuples = [node for node in ast.walk(tree) if isinstance(node, ast.Tuple)]
    assert len(tuples) >= 200, f"only {len(tuples)} rows in the corpus module - did a generator run overwrite it?"
