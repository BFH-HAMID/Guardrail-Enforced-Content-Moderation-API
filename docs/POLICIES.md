# Writing and tuning policies

`config/policies.yaml` is the only file most changes should touch: a new rule, a wider exception, a
threshold. No restart is needed in dev (mtime hot-reload) and `POST /admin/reload-policies` forces
it in prod.

## Shape

Illustrative skeleton of the schema — `config/policies.yaml` is the reference for what actually ships:

```yaml
version: "1.0.0"
# one preset per moderation mode (the file's real values)
modes:
  strict:
    toxicity_block: 0.5
    injection_block: 0.45
    jailbreak_block: 0.45
    sensitive_topic_flag: 0.45
    stage1_consensus: any
    llm_weight: 0.7
    classifier_weight: 0.3
  balanced:
    toxicity_block: 0.72
    injection_block: 0.6
    jailbreak_block: 0.6
    sensitive_topic_flag: 0.6
    stage1_consensus: any
    llm_weight: 0.6
    classifier_weight: 0.4
  lenient:
    toxicity_block: 0.86
    injection_block: 0.78
    jailbreak_block: 0.78
    sensitive_topic_flag: 0.8
    stage1_consensus: both
    llm_weight: 0.5
    classifier_weight: 0.5

policies:
  - id: dangerous_goods                  # unique; referenced in logs, /explain, and the eval corpus
    name: Weapons and dangerous goods
    description: >-                        # required by CI: an operator must be able to read /policies and understand why
      Sourcing, buying or building weapons and controlled goods, including Bangla phrasing.
    stage: stage2                          # stage1 | stage2 | stage3_output
    category: dangerous_goods              # the label clients and the corpus see
    severity: high                         # info | low | medium | high | critical
    action: block                          # block | flag | mask | rewrite | redact | disclaimer | allow
    min_score: 0.55                        # scaled by the mode factor below
    require_all_clauses: false
    regex:
      patterns:                            # see the gotchas below - note the doubled backslashes
        - "\\b(?:make|build)\\s+(?:a\\s+)?(?:bomb|explosive)\\b"
    exceptions:
      - "(?:novel|story|screenplay|news report|documentary)\\b[^.]{0,60}(?:bomb|explosive)"
    rewrite_hint: null
    disclaimer: null
```

Clauses available (any policy needs at least one, unless `action: allow`):

| clause | fires when | notes |
| --- | --- | --- |
| `regex.patterns` | any pattern matches the detection view | first match wins per policy |
| `lexicon.terms` | term present (word-boundary aware) | good for slur lists; `min_matches` supported |
| `classifier.categories` | the optional Stage-1 classifier names them | degrades to no-op when the classifier is off |
| `llm_categories` | the judge named them | same degradation rule |
| `obfuscation` | obfuscation score ≥ clause threshold | attack signal, not content class |
| `payloads` | base64/hex/URL blobs with suspicious decoded content | pairs with the decoded view |
| `script` | script mix (e.g. Latin-in-Bengali) | used to relax `[A-Za-z]` assumptions |
| `entities` | N PII spans of given types | `min_count`, `require_distinct_types` |
| `pii_entities` | a PII type alone is enough (masking) | drives `action: mask` |
| `stage1` | upstream flagged a category | reads Stage 1's **flat** score fields |
| `rewrite_substitutions` | always, once the policy fired | soften-with-replacement, e.g. mild slurs |

`score = min(1, weight·0.45 + Σ clause)` must reach `min_score × {strict 0.8, balanced 1.0,
lenient 1.25}`.

## The gotchas that cost real time here

1. **Patterns are stored in YAML double-quoted scalars, so a backslash must be doubled in the
   file:** `"\\b(?:make|build)\\b"`. Writing `"\b"` into a double-quoted scalar is a
   `yaml.safe_load` error (`unknown escape character`); writing `"\w"` is worse — it loads, and
   silently changes meaning. Prefer single quotes (`'\\b…'` vs `'\b…'`) when unsure, and always
   re-load the file before committing.
2. **Never write `\u09a8`-style escapes for Bangla.** Put the literal characters in the file. The
   loader and the corpus both carry real Unicode, and an escape that survives YAML can still fail to
   match once the sanitizer NFKC-normalizes the text.
3. **`\b` is unusable next to Bengali.** Bengali vowel signs are `\W`, so `\b` after a Bangla letter
   matches nothing — the rule silently never fires. Use the helpers in
   [`app/core/text.py`](../app/core/text.py): `indic_safe_pattern(source)` / `compile_pattern(...)`,
   which swap boundary assertions for lookarounds that work for Indic scripts. The engine applies
   them for you at `regex`, `lexicon` and `exceptions` evaluation; scripts that edit the YAML
   directly must not re-introduce raw `\b` next to non-ASCII.
4. **Heuristic and classifier scores ignore your `exceptions`.** A policy-level exception cannot
   downgrade a Stage-1 block. Frame checks ("this is quoted dialogue", "this is a newsroom sentence",
   "passive ideation without a plan") therefore live in
   [`app/pipeline/injection.py`](../app/pipeline/injection.py) — see `_SELF_HARM_RE`,
   `_VIOLENCE_INTENT_RE`, `_CRITICAL_FRAME_RE`, `_QUOTED_SPEECH_RE`.
5. **`action: flag` never blocks.** If a row must block, the policy needs `action: block` *and* a
   score above threshold; `flag` rows land in the review queue and the caller still gets their text.
6. **A bare `\w+` slot in a multi-clause pattern will over-fire.** A guard written as
   `nobody (would|will) (miss|care)` matched "the old stadium, nobody would miss it". Qualify the
   slot (`(?:she|he|they|i)`) or anchor the object.
7. **`min_score` above ~0.75 on harassment-style policies turns sarcasm into blocks.** The
   soften-with-`rewrite_substitutions` path exists for exactly that band.

## Editing the file safely

The three checks `scripts/run_checks.sh` and CI run:

```bash
python - <<'PY'
import re, yaml, pathlib
doc = yaml.safe_load(pathlib.Path("config/policies.yaml").read_text(encoding="utf-8"))
ids = [p["id"] for p in doc["policies"]]
assert len(ids) == len(set(ids)), "duplicate ids"
for p in doc["policies"]:
    assert (p.get("description") or "").strip(), p["id"]
    for pat in ((p.get("regex") or {}).get("patterns") or []) + list(p.get("exceptions") or []):
        re.compile(pat, re.IGNORECASE)
print(len(doc["policies"]), "policies ok")
PY
```

Inserting patterns programmatically: match the file's *stored* form (`"\\b…"`), not the regex source
form, and assert the replacement actually changed something (`assert new != old`) — a silent no-op
patch is the most common way a "fixed" policy stays broken.

## The tuning loop

```bash
# 1. reproduce the miss
python scripts/why.py "the failing text" --mode balanced --json | head -40
# 2. change policies.yaml or injection.py
# 3. re-score the tuned corpus, and the held-out set
python scripts/evaluate.py --limit 0 --modes balanced strict lenient --failures
python scripts/evaluate.py --dataset tests/data/heldout_adversarial.jsonl --limit 0
```

Reading `why`: `fired: false` with `downgraded_by_exception` set means the rule matched and an
exception saved it — the most common reason a policy "looks broken" and is not.

Rules for this loop, learned the hard way:

* **Add a pattern, don't add a label.** If the model should not see it, the corpus row's
  `expect_status` does not get downgraded to make the number look better.
* **Relabel only when the intent was wrong**, never to fit the detector. Rows whose rationale changed
  during this project (crisis-education rows that should be *allowed with resources* rather than
  silently blocked) are listed per-row in [EVALUATION_REPORT.md](../EVALUATION_REPORT.md).
* **Widen the exception, not the policy,** when a false block appears: over-blocking is the failure
  users feel first.
* **Tune against the tuned corpus only.** `tests/data/heldout_adversarial.jsonl` is scored, never
  fed back — the moment you add a pattern to make a held-out row pass, it stops being held out.
  Regenerate it by hand, occasionally, with new attacks.
* Every corpus row needs a `rationale`. A label nobody can defend is a label nobody should trust.

## When a rule needs code

Only these cases justify touching Python instead of YAML:

* the signal is not textual (rate, repetition, script mix, decoded payload structure) → heuristic layer;
* the exemption needs parse-level context the regex cannot see (quoted speech, narrative frame) →
  `injection.py`;
* the action is not expressible (structured redaction, entity-aware masking) → `stage2_policy.py`;
* a new clause type → `policy.py`'s `_evaluate_*` methods + a `Policy` field + a test.

In each case, keep the stage contract: return a `StageResult` with `decision`, `categories`,
`violations` (with spans, so reviewers can point), `reasons`, and never raise — raise and the chain
fails closed, which is correct but useless to a reviewer.
