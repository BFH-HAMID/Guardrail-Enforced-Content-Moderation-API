"""Policy bundle: YAML-driven moderation rules + per-mode thresholds.

Policies are data, not code: :file:`config/policies.yaml` defines severity, action
and matching clauses so an operator can tighten or relax the service without a
deploy. The bundle is validated with Pydantic at load time (a typo'd action is a
startup error, not a 3am surprise) and hot-reloaded when the file's mtime changes.

A policy is a *clause set*: it fires when any of its clauses match, unless
``require_all_clauses`` is set. Clauses cover cheap deterministic detectors
(regex / lexicon / obfuscation / script mix / encoded payloads) and the outputs of
the expensive detectors (classifier score, LLM category, Presidio entity type).
``exceptions`` are second-pass regexes that downgrade a hit - this is how we keep
"how do I write a villain who hates people?" out of the hate-speech bucket without
weakening real hate speech.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.config import ModeThresholds
from app.core.text import PayloadSignals, indic_safe_pattern

PolicyStage = Literal["stage1", "stage2", "stage3_output"]
PolicyAction = Literal["block", "rewrite", "mask", "allow", "flag", "disclaimer"]
SEVERITY_WEIGHT: dict[str, float] = {"info": 0.2, "low": 0.4, "medium": 0.6, "high": 0.8, "critical": 0.95}
SEVERITIES = tuple(SEVERITY_WEIGHT)
MODES = ("strict", "balanced", "lenient")


class RegexClause(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patterns: list[str] = Field(min_length=1)
    flags: int = re.IGNORECASE
    boost: float = 0.35
    apply_to: Literal["view", "text", "both"] = "both"

    @field_validator("patterns")
    @classmethod
    def _compilable(cls, patterns: list[str]) -> list[str]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:  # pragma: no cover - config error path
                raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
        return patterns


class LexiconClause(BaseModel):
    model_config = ConfigDict(extra="forbid")
    terms: dict[str, list[str]] = Field(default_factory=dict, description="language code -> terms")
    boost: float = 0.4
    whole_word: bool = True
    min_matches: int = 1


class ClassifierClause(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Literal["toxicity", "prompt_injection", "jailbreak", "nsfw", "any"] = "any"
    min_score: float = 0.5
    boost: float = 0.3


class LlmCategoryClause(BaseModel):
    model_config = ConfigDict(extra="forbid")
    categories: list[str] = Field(min_length=1)
    min_score: float = 0.55
    boost: float = 0.35


class ObfuscationClause(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_score: float = 0.5
    flags: list[str] = Field(default_factory=list)
    boost: float = 0.25


#: policy-facing signal name -> key in PayloadSignals.as_dict()
PAYLOAD_SIGNAL_KEYS: dict[str, str] = {
    "base64": "base64_spans",
    "hex": "hex_spans",
    "url_encoded": "url_encoded",
    "rot13": "rot13_suspect",
    "unicode_escape": "unicode_escape",
    "prompt_like_json": "prompt_like_json",
}


class PayloadClause(BaseModel):
    model_config = ConfigDict(extra="forbid")
    signals: list[Literal["base64", "hex", "url_encoded", "rot13", "unicode_escape", "prompt_like_json"]] = Field(default_factory=list)
    min_score: float = 0.3
    boost: float = 0.4


class ScriptClause(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scripts: list[str] = Field(min_length=1)
    min_share: float = 0.2
    boost: float = 0.1


class EntityClause(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entities: list[str] = Field(min_length=1)
    min_count: int = 1
    boost: float = 0.3


class SubstitutionRule(BaseModel):
    """Deterministic regex substitution used when a policy asks for a rewrite."""

    model_config = ConfigDict(extra="forbid")
    pattern: str
    replacement: str
    description: str = ""

    @field_validator("pattern")
    @classmethod
    def _compilable(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as exc:  # pragma: no cover
            raise ValueError(f"invalid substitution regex: {exc}") from exc
        return v


class Stage1Clause(BaseModel):
    """Condition a later-stage policy on the previous stage's verdict.

    This is what lets ``policies.yaml`` say "if Stage 1 flagged self-harm *and* the
    user is asking for help, attach crisis resources" without any Python change.
    """

    model_config = ConfigDict(extra="forbid")
    categories: list[str] = Field(min_length=1)
    min_score: float = 0.3
    decisions: list[Literal["pass", "flag", "block"]] = Field(default_factory=list)
    boost: float = 0.5


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str = ""
    stage: PolicyStage = "stage2"
    category: str
    severity: Literal["info", "low", "medium", "high", "critical"] = "medium"
    action: PolicyAction = "block"
    min_score: float = 0.5
    require_all_clauses: bool = False
    exceptions: list[str] = Field(default_factory=list)
    replacement: str | None = None
    rewrite_hint: str | None = None
    disclaimer: str | None = None
    pii_entities: list[str] = Field(default_factory=list)
    requires_policy_hits: list[str] = Field(default_factory=list)
    applies_to_modes: list[Literal["strict", "balanced", "lenient"]] = Field(default_factory=list)

    regex: RegexClause | None = None
    lexicon: LexiconClause | None = None
    classifier: ClassifierClause | None = None
    llm_categories: LlmCategoryClause | None = None
    obfuscation: ObfuscationClause | None = None
    payloads: PayloadClause | None = None
    script: ScriptClause | None = None
    entities: EntityClause | None = None
    stage1: Stage1Clause | None = None
    rewrite_substitutions: list[SubstitutionRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def _at_least_one_clause(self) -> "Policy":
        clauses = [self.regex, self.lexicon, self.classifier, self.llm_categories, self.obfuscation, self.payloads, self.script, self.entities, self.stage1]
        if self.action != "allow" and all(c is None for c in clauses) and not self.pii_entities:
            raise ValueError(f"policy {self.id!r} has no matching clause; it could never fire")
        if self.action == "mask" and not (self.entities or self.pii_entities or self.regex):
            raise ValueError(f"mask policy {self.id!r} needs entities, pii_entities or regex to know what to mask")
        for pattern in self.exceptions:
            try:
                re.compile(pattern)
            except re.error as exc:  # pragma: no cover
                raise ValueError(f"policy {self.id!r}: invalid exception regex: {exc}") from exc
        return self

    # ---------------------------------------------------------------- derived
    @property
    def weight(self) -> float:
        return SEVERITY_WEIGHT[self.severity]

    @property
    def clause_count(self) -> int:
        return sum(
            1
            for clause in (self.regex, self.lexicon, self.classifier, self.llm_categories, self.obfuscation, self.payloads, self.script, self.entities)
            if clause is not None
        )

    def applies_to(self, mode: str) -> bool:
        return not self.applies_to_modes or mode in self.applies_to_modes

    def action_for(self, mode: str) -> PolicyAction:
        """``lenient`` mode downgrades blocking to rewriting for the soft categories."""
        action = self.action
        if mode == "lenient" and action == "block" and self.severity in {"low", "medium", "info"}:
            return "rewrite"
        if mode == "strict" and action in {"flag", "rewrite"} and self.severity == "critical":
            return "block"
        return action


class RuleHit(BaseModel):
    """One policy firing on one piece of text."""

    model_config = ConfigDict(extra="forbid")

    policy_id: str
    category: str
    severity: str
    action: PolicyAction
    score: float
    reason: str
    matched: list[str] = Field(default_factory=list)
    spans: list[tuple[int, int]] = Field(default_factory=list)
    detector: str = "policy"

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ModeOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")
    toxicity_block: float | None = None
    injection_block: float | None = None
    jailbreak_block: float | None = None
    sensitive_topic_flag: float | None = None
    stage1_consensus: Literal["any", "both"] | None = None
    llm_weight: float | None = None
    classifier_weight: float | None = None

    def merged(self, base: ModeThresholds) -> ModeThresholds:
        updates = {k: v for k, v in self.model_dump().items() if v is not None}
        return ModeThresholds(**{**base.model_dump(), **updates})


DEFAULT_MODE_THRESHOLDS: dict[str, ModeThresholds] = {
    "strict": ModeThresholds(
        toxicity_block=0.50, injection_block=0.45, jailbreak_block=0.45, sensitive_topic_flag=0.45,
        stage1_consensus="any", llm_weight=0.70, classifier_weight=0.30,
    ),
    "balanced": ModeThresholds(
        toxicity_block=0.72, injection_block=0.60, jailbreak_block=0.60, sensitive_topic_flag=0.60,
        stage1_consensus="any", llm_weight=0.60, classifier_weight=0.40,
    ),
    "lenient": ModeThresholds(
        toxicity_block=0.86, injection_block=0.78, jailbreak_block=0.78, sensitive_topic_flag=0.80,
        stage1_consensus="both", llm_weight=0.50, classifier_weight=0.50,
    ),
}


class PolicyBundle(BaseModel):
    """Validated ``policies.yaml`` + the thresholds that gate it."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    version: str = "0.0.0"
    name: str = "default"
    policies: list[Policy] = Field(default_factory=list)
    mode_thresholds: dict[str, ModeThresholds] = Field(default_factory=lambda: dict(DEFAULT_MODE_THRESHOLDS))
    source_path: Path | None = None
    loaded_at: float = Field(default_factory=time.time)
    checksum: str = "unloaded"

    @field_validator("policies")
    @classmethod
    def _unique_ids(cls, policies: list[Policy]) -> list[Policy]:
        seen: set[str] = set()
        for policy in policies:
            if policy.id in seen:
                raise ValueError(f"duplicate policy id {policy.id!r}")
            seen.add(policy.id)
        return policies

    # ---------------------------------------------------------------- lookups
    def by_stage(self, stage: PolicyStage, mode: str = "balanced") -> list[Policy]:
        return [p for p in self.policies if p.stage == stage and p.applies_to(mode)]

    def get(self, policy_id: str) -> Policy | None:
        return next((p for p in self.policies if p.id == policy_id), None)

    def thresholds(self, mode: str) -> ModeThresholds:
        return self.mode_thresholds.get(mode, self.mode_thresholds["balanced"])

    def mask_entities(self) -> dict[str, str]:
        """Entity type -> placeholder, collected from every mask action in the file."""
        out: dict[str, str] = {}
        for policy in self.policies:
            if policy.action == "mask":
                for entity in policy.pii_entities or (policy.entities.entities if policy.entities else []):
                    out[entity] = policy.replacement or f"[{entity}]"
        return out

    # -------------------------------------------------------------- matching
    @staticmethod
    def _regex_hits(policy: Policy, text: str, view: str) -> list[tuple[int, int, str]]:
        if not policy.regex:
            return []
        targets: list[tuple[str, int, int]] = []
        clause = policy.regex
        if clause.apply_to in {"text", "both"}:
            targets.append((text, 0, 0))
        if clause.apply_to in {"view", "both"}:
            targets.append((view, 1, 0))
        hits: list[tuple[int, int, str]] = []
        for haystack, _source, _off in targets:
            for raw_pattern in clause.patterns:
                pattern = indic_safe_pattern(raw_pattern)
                for match in re.finditer(f"(?i){pattern}" if clause.flags & re.IGNORECASE else pattern, haystack):
                    hits.append((match.start(), match.end(), match.group(0)))
            if hits:
                break  # first pattern that fires is enough; avoids O(patterns x text) noise
        return hits

    @staticmethod
    def _lexicon_hits(policy: Policy, view: str) -> list[tuple[int, int, str]]:
        if not policy.lexicon or not policy.lexicon.terms:
            return []
        hits: list[tuple[int, int, str]] = []
        for _lang, terms in policy.lexicon.terms.items():
            for term in terms:
                term = term.lower().strip()
                if not term:
                    continue
                if policy.lexicon.whole_word and re.match(r"^[a-z0-9\u0980-\u09ff ]+$", term):
                    # Whole-word for a Bengali term: the trailing \b is dropped by
                    # indic_safe_pattern because a final vowel sign is not \w in Python.
                    pattern = indic_safe_pattern(
                        r"\b" + re.escape(term).replace(r"\ ", r"[\s\-_]+") + r"\b"
                    )
                else:
                    pattern = re.escape(term)
                for match in re.finditer(pattern, view):
                    hits.append((match.start(), match.end(), match.group(0)))
        return hits

    def evaluate(
        self,
        *,
        stage: PolicyStage,
        mode: str,
        text: str,
        view: str,
        scores: dict[str, float] | None = None,
        llm_categories: dict[str, float] | None = None,
        obfuscation_score: float = 0.0,
        obfuscation_flags: list[str] | None = None,
        payloads: PayloadSignals | None = None,
        script_mix: dict[str, float] | None = None,
        entities: list[dict[str, Any]] | None = None,
        stage1_result: Any = None,
    ) -> list[RuleHit]:
        """Run every policy for ``stage`` against the detector outputs."""
        scores = scores or {}
        llm_categories = llm_categories or {}
        script_mix = script_mix or {}
        entities = entities or []
        hits: list[RuleHit] = []

        for policy in self.by_stage(stage, mode):
            if policy.action == "allow":
                continue
            threshold = policy.min_score * ({"strict": 0.8, "balanced": 1.0, "lenient": 1.25}[mode])
            evidence: list[tuple[int, int, str]] = []
            contributions: list[float] = []
            detectors: list[str] = []
            reasons: list[str] = []

            if policy.regex:
                found = self._regex_hits(policy, text, view)
                if len(found) >= 1:
                    evidence.extend(found)
                    contributions.append(policy.regex.boost)
                    detectors.append("regex")
                    reasons.append(f"pattern match: {', '.join(sorted({h[2][:40] for h in found})[:3])}")

            if policy.lexicon:
                found = self._lexicon_hits(policy, view)
                if len(found) >= max(1, policy.lexicon.min_matches):
                    evidence.extend(found)
                    contributions.append(policy.lexicon.boost + 0.05 * (len(found) - 1))
                    detectors.append("lexicon")
                    reasons.append(f"lexicon terms: {', '.join(sorted({h[2][:24] for h in found})[:4])}")

            if policy.classifier:
                label = policy.classifier.label
                candidate = max(scores.values()) if label == "any" else scores.get(label, 0.0)
                if candidate >= max(policy.classifier.min_score, threshold * 0.5):
                    contributions.append(policy.classifier.boost * candidate)
                    detectors.append("classifier")
                    reasons.append(f"{label} classifier score {candidate:.2f}")

            if policy.llm_categories:
                best_cat, best_score = "", 0.0
                for cat, val in llm_categories.items():
                    if cat in policy.llm_categories.categories and val > best_score:
                        best_cat, best_score = cat, val
                if best_score >= max(policy.llm_categories.min_score, threshold * 0.5):
                    contributions.append(policy.llm_categories.boost * best_score)
                    detectors.append("llm_judge")
                    reasons.append(f"judge category {best_cat} @ {best_score:.2f}")

            if policy.obfuscation:
                clause = policy.obfuscation
                flag_match = (not clause.flags) or any(f in (obfuscation_flags or []) for f in clause.flags)
                if flag_match and obfuscation_score >= clause.min_score:
                    contributions.append(clause.boost * min(1.0, obfuscation_score + 0.2))
                    detectors.append("obfuscation")
                    reasons.append(f"obfuscation score {obfuscation_score:.2f} flags={','.join(obfuscation_flags or ['-'])}")

            if policy.payloads:
                clause = policy.payloads
                sniffed = payloads.as_dict() if payloads else {}
                wanted = clause.signals or list(PAYLOAD_SIGNAL_KEYS)
                active = [name for name in wanted if sniffed.get(PAYLOAD_SIGNAL_KEYS[name])]
                score = (payloads.score if payloads else 0.0) or (0.6 if active else 0.0)
                if active and score >= clause.min_score:
                    contributions.append(clause.boost * score)
                    detectors.append("payload")
                    reasons.append(f"encoded payload: {', '.join(active)}")

            if policy.script:
                share = max((script_mix.get(s, 0.0) for s in policy.script.scripts), default=0.0)
                if share >= policy.script.min_share:
                    contributions.append(policy.script.boost)
                    detectors.append("script")
                    reasons.append(f"{policy.script.scripts[0]} script share {share:.2f}")

            if policy.entities:
                wanted = set(policy.entities.entities)
                matching = [e for e in entities if e.get("entity_type") in wanted]
                if len(matching) >= policy.entities.min_count:
                    contributions.append(policy.entities.boost + 0.05 * (len(matching) - 1))
                    evidence.extend((int(e["start"]), int(e["end"]), str(e.get("text", ""))) for e in matching)
                    detectors.append("pii")
                    reasons.append("entities: " + ", ".join(sorted({str(e['entity_type']) for e in matching})))

            if policy.stage1:
                clause = policy.stage1
                stage1 = stage1_result
                if stage1 is not None and (not clause.decisions or stage1.decision.value in clause.decisions):
                    overlapping = [c for c in stage1.categories if c in clause.categories]
                    peak = max((stage1.toxicity, stage1.prompt_injection, stage1.jailbreak), default=0.0)
                    if overlapping and peak >= clause.min_score:
                        contributions.append(clause.boost * (0.6 + 0.4 * peak))
                        detectors.append("stage1")
                        reasons.append(f"stage1 flagged {', '.join(overlapping[:3])} @ {peak:.2f}")
                        for violation in stage1.violations:
                            if violation.category in overlapping and violation.span:
                                evidence.append((violation.span[0], violation.span[1], violation.excerpt or ""))

            if policy.pii_entities and not policy.entities:
                wanted = set(policy.pii_entities)
                matching = [e for e in entities if e.get("entity_type") in wanted]
                if matching:
                    contributions.append(0.35 + 0.05 * (len(matching) - 1))
                    evidence.extend((int(e["start"]), int(e["end"]), str(e.get("text", ""))) for e in matching)
                    detectors.append("pii")
                    reasons.append("PII entities: " + ", ".join(sorted({str(e['entity_type']) for e in matching})))

            if not contributions:
                continue

            if policy.require_all_clauses:
                # Documented in the module docstring and relied on by two policies
                # (doxxing_intent_bundle, harassment_softener): the policy is only safe because
                # *every* configured clause has to agree. Without this check the flag was
                # silently ignored and, for example, "my manager calls me an idiot" got
                # de-escalated on the lexicon hit alone.
                required = {
                    "regex": "regex",
                    "lexicon": "lexicon",
                    "classifier": "classifier",
                    "llm_categories": "llm_judge",
                    "obfuscation": "obfuscation",
                    "payloads": "payload",
                    "script": "script",
                    "entities": "pii",
                    "stage1": "stage1",
                }
                seen = set(detectors)
                if any(
                    getattr(policy, name, None) is not None and detector not in seen
                    for name, detector in required.items()
                ):
                    continue

            score = min(1.0, policy.weight * 0.45 + sum(contributions))
            if score < threshold:
                continue

            if policy.exceptions:
                haystack = f"{text}\n{view}"
                if any(re.search(f"(?i){indic_safe_pattern(pattern)}", haystack) for pattern in policy.exceptions):
                    continue  # documented safe use (news quote, fictional writing, self-help...)

            spans = [(s, e) for s, e, _ in sorted(evidence)]
            matched = [m for _s, _e, m in sorted(evidence)]
            hits.append(
                RuleHit(
                    policy_id=policy.id,
                    category=policy.category,
                    severity=policy.severity,
                    action=policy.action_for(mode),
                    score=round(score, 4),
                    reason="; ".join(reasons)[:600],
                    matched=[m[:160] for m in matched][:12],
                    spans=spans[:40],
                    detector="+".join(dict.fromkeys(detectors)) or "policy",
                )
            )
        # Second pass: drop hits whose gating prerequisites did not fire. This is how
        # "mask a personal name only when it sits next to another identifier" works.
        fired = {h.policy_id for h in hits}
        kept: list[RuleHit] = []
        for hit in hits:
            policy = self.get(hit.policy_id)
            if policy and policy.requires_policy_hits and not set(policy.requires_policy_hits) & fired:
                continue
            kept.append(hit)
        return kept

    def exceptions_for(self, policy_id: str) -> list[str]:
        policy = self.get(policy_id)
        return list(policy.exceptions) if policy else []

    def summary(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "source": str(self.source_path) if self.source_path else "builtin",
            "checksum": self.checksum[:16],
            "policies": len(self.policies),
            "by_stage": {stage: len(self.by_stage(stage)) for stage in ("stage1", "stage2", "stage3_output")},
            "thresholds": {mode: self.thresholds(mode).model_dump() for mode in MODES},
        }


# ------------------------------------------------------------------ loading


def _checksum(path: Path) -> str:
    return hashlib_sha256(path.read_bytes())


def hashlib_sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def load_policy_bundle(path: str | Path | None) -> PolicyBundle:
    """Load + validate the YAML bundle. Missing file -> built-in default bundle."""
    if path is None:
        return PolicyBundle(version="builtin", name="builtin-default")
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"policies file not found: {file_path}")
    raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{file_path}: top level YAML must be a mapping")
    payload = dict(raw)
    payload.pop("comment", None)
    modes_raw = payload.pop("modes", {}) or {}
    thresholds = dict(DEFAULT_MODE_THRESHOLDS)
    for mode, override in modes_raw.items():
        if mode not in MODES:
            raise ValueError(f"{file_path}: unknown mode {mode!r} (expected one of {MODES})")
        thresholds[mode] = ModeOverride.model_validate(override or {}).merged(thresholds[mode])
    payload["mode_thresholds"] = thresholds
    payload["source_path"] = file_path
    payload["checksum"] = _checksum(file_path)
    bundle = PolicyBundle.model_validate(payload)
    if not bundle.policies:
        raise ValueError(f"{file_path}: no policies defined")
    return bundle


class PolicyRegistry:
    """Cached bundle with mtime-based hot reload (policies change without a deploy)."""

    def __init__(self, path: str | Path | None) -> None:
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._bundle: PolicyBundle | None = None

    @property
    def path(self) -> Path | None:
        return self._path

    def current(self) -> PolicyBundle:
        with self._lock:
            if self._bundle is None:
                self._bundle = load_policy_bundle(self._path)
                return self._bundle
            if self._path is not None and self._path.exists():
                if self._bundle.checksum != _checksum(self._path):
                    self._bundle = load_policy_bundle(self._path)
            return self._bundle

    def reload(self) -> PolicyBundle:
        with self._lock:
            self._bundle = load_policy_bundle(self._path)
            return self._bundle


_REGISTRY: PolicyRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_registry(path: str | Path | None = None) -> PolicyRegistry:
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None or (path is not None and Path(path) != _REGISTRY.path):
            _REGISTRY = PolicyRegistry(path)
        return _REGISTRY


def get_policy_bundle(path: str | Path | None = None) -> PolicyBundle:
    if path is None:
        from app.core.config import get_settings

        path = get_settings().policies_file
    return get_registry(path).current()


def policy_bundle_as_json(path: str | Path | None = None) -> str:  # pragma: no cover - debug helper
    return json.dumps(get_policy_bundle(path).summary(), indent=2, default=str)
