# Architecture

<p align="center">
  <img src="assets/pipeline.svg" alt="Guardrail chain diagram: untrusted text, three stages, v1 envelope, fail-closed branch" width="100%">
</p>

One request, three stages, one envelope. The interesting parts are the boundaries: what each stage
is allowed to decide, and what happens when it cannot decide.

```
POST /moderate {text, mode, include_trace, return_reasoning, metadata}
      │
      ├─ auth (X-API-Key → salted digest lookup, scopes)          401/403
      ├─ rate limit (per key, in-process token bucket)             429 + Retry-After
      │
      ▼
 app/api/routes.py::moderate
      │   normalize once (sanitize_text) → PipelineContext
      ▼
 ┌───────────────────────────── ModerationPipeline.run (global timeout, fail-closed) ─────────────────────────────┐
 │                                                                                                                  │
 │  Stage 1  toxicity + prompt injection                                                                            │
 │    heuristic analyzer (weight-scored patterns, obfuscation, encoded payloads, script mix)                        │
 │    [optional] transformers classifier        [optional] LLM judge (small model, schema'd)                       │
 │    fuse(scores, consensus)  →  PASS | FLAG | BLOCK + categories + violations                                     │
 │                              │                                                                                   │
 │  Stage 2  content policy + rewrite           ◄── short-circuit: a BLOCK here ends the chain                     │
 │    PII detection (BD formats + optional Presidio) → policies.yaml (46 rules, per-mode thresholds)               │
 │    mask → substitute → RE-SCAN own output → [optional] LLM rewrite → re-validate → redact → disclaimers       │
 │                              │                                                                                   │
 │  Stage 3  output                                                                            │
 │    passthrough: the Stage-2 text  |  assistant: generated answer                             │
 │    output check: leak detectors (PII, system prompt, jailbreak compliance) [+ full policies in assistant mode]  │
 │    on failure: substitute a safe fallback                                                      │
 │                                                                                                  │
 └─────────────────────────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
 app/post/postprocess.py::finalize   confidence aggregate → Prometheus → SQL row (+trace) → retention sweep
      │
      ▼
 app/post/formatter.py::format_response  →  v1 envelope (+ pipeline_trace if requested)
```

## The context object

`PipelineContext` ([`app/pipeline/base.py`](../app/pipeline/base.py)) is the only thing stages share.
It carries `raw_text` (never mutated), `text` (normalized), `view` (the *detection* view: normalized
text + decoded payload text + homoglyph forms, capped at 12 000 chars), the obfuscation/payload/
script signals, entities, and the accumulating `stage1`/`stage2`/`stage3` results plus the trace.

Two rules that fall out of this design:

* **Detection and output are different strings.** The view may contain things the returned text does
  not (decoded base64, rejoined letter spacing). Policies match the view; the caller gets `text`.
  That is why an encoded payload can be blocked without the response echoing the plaintext, and why
  span offsets for masking always refer to `text`, never the view.
* **Each stage hands the next one a *verdict*, not a vibe.** `Stage1Result` carries flat scores,
  categories, violations with spans, and `detector_detail`. Stage 2's `stage1` clause reads those
  flat fields — which is why a `Stage1Result` built without `toxicity=` looks like "nothing was
  detected upstream" (there is a test for exactly that trap).

## Why three stages instead of one prompt

A single "please moderate this" prompt conflates three decisions with different failure modes:

| decision | who is best at it | what a wrong answer costs |
| --- | --- | --- |
| "is this attacking the model?" | cheap heuristics + patterns, high recall | an attacker gets instructions |
| "what may we show a human / send downstream?" | declarative policy + PII masking | legal exposure, doxxing, over-blocking |
| "is the *answer* safe?" | the same detectors re-aimed at the output | a clean prompt ending in a harmful reply |

Splitting them means each leg has its own timeout, its own degradation path, and its own audit
trail. `/explain/{request_id}` is only useful because the stages recorded separately.

## Guardrails on the pipeline itself

These are the properties the tests in [`tests/`](../tests/) defend:

1. **Fail closed at every level.** A stage raising, timing out, or returning garbage produces
   `blocked` with `blocked.reason="moderation_error"`; the HTTP layer returns 503 with the same
   envelope plus `Retry-After`, and `flags.fail_closed=true`. There is no code path where an error
   yields `allowed`.
2. **Blocked content is never echoed.** `safe_output=""` on a block; only bounded excerpts survive
   in the audit row. A test asserts the full input string is absent from the response body.
3. **Rewrites are verified.** Stage 2 re-evaluates its own output; the optional LLM rewrite is
   discarded (redaction instead) if it still matches. An unverified rewrite is a bypass channel.
4. **Stage 3 does not re-litigate Stage 2** in `passthrough`. It runs *leak* detectors only, so a
   protective exemption (fiction, newsroom, help-seeking) cannot be silently undone at the last step.
5. **Metrics before storage.** `finalize()` records Prometheus counters before the DB write, and a
   failed write sets `X-Moderation-Storage: degraded` without changing the verdict. Losing an audit
   row must not blind the dashboard or 500 at the caller.
6. **Config is validated at import.** A nonsensical timeout/threshold is a startup failure.
7. **Nothing secret in the log line.** API keys are digested (salted with `api_key_pepper`), user ids
   hashed, raw bodies opt-in, and blocked text never enters the *response*.

## The policy engine

[`app/pipeline/policy.py`](../app/pipeline/policy.py) compiles YAML into `Policy` models, each
holding up to nine clause types that all contribute to one score:

```
score = min(1, weight · 0.45 + Σ clause contributions)   ≥   min_score × mode_factor
```

* mode factor: `strict 0.8 · balanced 1.0 · lenient 1.25` — one knob, three products;
* first match per policy wins (a policy fires once, no matter how many patterns hit), so
  `matched[]` stays reviewable;
* `require_all_clauses: true` makes a multi-clause policy conjunctive (used for doxxing bundles:
  identifiers **and** publication intent);
* `exceptions` are `re.search` against `text + view`; a match removes the hit. Exceptions live in
  YAML so an operator can widen them without touching Python — with the deliberate exception (pun
  intended) of the heuristic-score bypass described below.

Heuristic/classifier scores from Stage 1 block *independently of* policy `action: flag` and ignore
YAML `exceptions`. That asymmetry is load-bearing: it is why fiction/news/clinical exemptions had to
be written into [`injection.py`](../app/pipeline/injection.py) rather than the policy file — a
policy-level exception could not have downgraded an upstream score.

## Extension points

| want | touch |
| --- | --- |
| a new rule | `config/policies.yaml` (+ a corpus row + a test) — see [POLICIES.md](POLICIES.md) |
| a new detector signal | `HeuristicAnalyzer.analyze` + `Scorecard` field + fuse weights |
| a new provider (LLM/judge) | `app/pipeline/llm/` router chain; nothing in the stages changes |
| a new stage | subclass `Stage`, implement `_run`, insert into `ModerationPipeline.stages`, add a `StageName` |
| a new response shape | a `schema_version` in `formatter.py`; the v1 envelope is a contract, so version it |
| a different storage backend | `app/db/` models + repository; the pipeline never touches SQL |

A new stage must answer one question explicitly: *what happens when this stage fails?* The base
class answers "blocked" by default; overriding that is how a guardrail becomes an open door.

## Numbers, measured on this codebase

Rules-only (no judge), 225-row corpus, single process, dev box: p50 ≈ 55 ms, p95 ≈ 85 ms, p99 ≈
105 ms; ~1.7 s wall for the whole corpus at concurrency 8. The heuristic layer is ~0.5 ms/text, so
the tail is policy-regex and (when enabled) LLM latency. Cost, when the judge is on, is visible per
request in `usage.estimated_cost_usd` and aggregated in `/stats.cost`.
