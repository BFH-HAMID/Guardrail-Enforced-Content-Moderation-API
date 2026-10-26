# Configuration

Every setting is a `GUARD_*` environment variable (or a line in `.env`, or a real env var — see
*Precedence* below), defined once in [`app/core/config.py`](../app/core/config.py) and validated at
import time. There is no second source of truth: the table below is generated from the live model.

Regenerate it after touching `Settings`:

```bash
python scripts/gen_config_doc.py            # rewrite the block
python scripts/gen_config_doc.py --check    # what CI runs
```

<!-- BEGIN GENERATED: settings table (scripts/gen_config_doc.py) -->
| variable | default | meaning |
| --- | --- | --- |
| `GUARD_APP_NAME` | `Guardrail-Enforced Content Moderation API` | — |
| `GUARD_VERSION` | `1.0.0` | — |
| `GUARD_ENVIRONMENT` | `dev` | prod turns on the strict validator set (timeouts, secret hygiene, fail-fast policy load). |
| `GUARD_API_V1_PREFIX` | `` | — |
| `GUARD_HOST` | `0.0.0.0` | — |
| `GUARD_PORT` | `8000` | — |
| `GUARD_CORS_ORIGINS` | `*` | — |
| `GUARD_CORS_ALLOW_CREDENTIALS` | `False` | — |
| `GUARD_LOG_LEVEL` | `INFO` | — |
| `GUARD_LOG_JSON` | `True` | — |
| `GUARD_LOG_REQUEST_BODIES` | `False` | raw user text into application logs. Leave off. |
| `GUARD_MAX_LOG_FIELD_CHARS` | `2000` | — |
| `GUARD_DATABASE_URL` | `sqlite+aiosqlite:///./data/moderation.db` | empty string disables persistence; moderation still runs, but nothing is auditable. |
| `GUARD_DB_ECHO` | `False` | — |
| `GUARD_DB_POOL_SIZE` | `5` | — |
| `GUARD_DB_MAX_OVERFLOW` | `10` | — |
| `GUARD_DB_CREATE_ALL_ON_STARTUP` | `True` | — |
| `GUARD_AUTH_ENABLED` | `True` | dev only. With it off, callers are `anonymous` and no FK is written. |
| `GUARD_API_KEY_HEADER` | `X-API-Key` | — |
| `GUARD_API_KEY_PEPPER` | `••••` | different pepper = different digests, so a leaked DB cannot be rainbow-tabled. |
| `GUARD_BOOTSTRAP_API_KEYS` | `••••` | comma separated; digested + salted with `api_key_pepper`, never stored raw. |
| `GUARD_ALLOW_QUERY_STRING_API_KEY` | `False` | — |
| `GUARD_RATE_LIMIT_ENABLED` | `True` | — |
| `GUARD_RATE_LIMIT_REQUESTS_PER_MINUTE` | `120` | per key, in-process. Scale with replicas, not `--workers`. |
| `GUARD_RATE_LIMIT_BURST` | `20` | — |
| `GUARD_RATE_LIMIT_BACKEND` | `memory` | — |
| `GUARD_REDIS_URL` | `redis://localhost:6379/0` | — |
| `GUARD_MAX_INPUT_CHARS` | `8000` | 422 beyond this - truncating an input to *allow* it would silently drop evidence. |
| `GUARD_MIN_INPUT_CHARS` | `1` | — |
| `GUARD_NORMALIZE_UNICODE` | `True` | — |
| `GUARD_STRIP_CONTROL_CHARS` | `True` | — |
| `GUARD_MAX_METADATA_ITEMS` | `16` | — |
| `GUARD_MAX_METADATA_KEY_CHARS` | `64` | — |
| `GUARD_MAX_METADATA_VALUE_CHARS` | `256` | — |
| `GUARD_DEFAULT_MODE` | `balanced` | per-request `mode` overrides it; the preset only scales policy thresholds. |
| `GUARD_POLICIES_PATH` | `config/policies.yaml` | resolved against the project root (cwd-independent); hot-reloaded on mtime change, `POST /admin/reload-policies` forces it. |
| `GUARD_PIPELINE_REQUEST_TIMEOUT_S` | `60.0` | the ceiling: past it, the response is fail-closed blocked, never a pass. |
| `GUARD_STAGE1_TIMEOUT_S` | `20.0` | — |
| `GUARD_STAGE2_TIMEOUT_S` | `25.0` | — |
| `GUARD_STAGE3_TIMEOUT_S` | `45.0` | — |
| `GUARD_STORE_TRACE` | `True` | persist the per-stage trace that `/explain/{request_id}` renders. |
| `GUARD_TRACE_PREVIEW_CHARS` | `400` | — |
| `GUARD_STAGE1_USE_CLASSIFIER` | `True` | — |
| `GUARD_STAGE1_CLASSIFIER_MODEL` | `protectai/deberta-v3-base-prompt-injection-v2` | — |
| `GUARD_STAGE1_CLASSIFIER_DEVICE` | `auto` | — |
| `GUARD_STAGE1_USE_LLM_JUDGE` | `True` | — |
| `GUARD_STAGE1_FAIL_CLOSED_ON_LLM_ERROR` | `True` | if the judge is configured and errors, the row is blocked with `moderation_error`. |
| `GUARD_USE_GUARDRAILS` | `False` | — |
| `GUARD_STAGE2_USE_PRESIDIO` | `True` | — |
| `GUARD_STAGE2_ALLOW_LLM_REWRITE` | `True` | the rewrite is re-scanned; a rewrite that still hits is redacted instead. |
| `GUARD_STAGE3_MODE` | `passthrough` | `passthrough` moderates and forwards; `assistant` also generates the answer. |
| `GUARD_STAGE3_OUTPUT_CHECK` | `True` | — |
| `GUARD_LLM_PROVIDER` | `auto` | — |
| `GUARD_LLM_FALLBACK_ORDER` | `gemini, openai, anthropic, ollama, local` | — |
| `GUARD_OPENAI_API_KEY` | `••••` | — |
| `GUARD_GEMINI_API_KEY` | `••••` | — |
| `GUARD_ANTHROPIC_API_KEY` | `••••` | — |
| `GUARD_OLLAMA_BASE_URL` | `http://localhost:11434` | — |
| `GUARD_OPENAI_BASE_URL` | `https://api.openai.com/v1` | — |
| `GUARD_ANTHROPIC_BASE_URL` | `https://api.anthropic.com/v1` | — |
| `GUARD_GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta` | — |
| `GUARD_SMALL_MODEL` | `gemini-2.0-flash-lite` | — |
| `GUARD_MAIN_MODEL` | `gemini-2.0-flash` | — |
| `GUARD_OLLAMA_SMALL_MODEL` | `llama3.2:1b` | — |
| `GUARD_OLLAMA_MAIN_MODEL` | `llama3.1:8b` | — |
| `GUARD_OPENAI_SMALL_MODEL` | `gpt-4o-mini` | — |
| `GUARD_OPENAI_MAIN_MODEL` | `gpt-4o` | — |
| `GUARD_ANTHROPIC_SMALL_MODEL` | `claude-3-5-haiku-latest` | — |
| `GUARD_ANTHROPIC_MAIN_MODEL` | `claude-3-5-sonnet-latest` | — |
| `GUARD_LLM_TEMPERATURE` | `0.0` | — |
| `GUARD_LLM_MAX_TOKENS` | `1024` | — |
| `GUARD_LLM_REQUEST_TIMEOUT_S` | `30.0` | — |
| `GUARD_LLM_MAX_RETRIES` | `2` | — |
| `GUARD_LLM_RETRY_BACKOFF_S` | `0.4` | — |
| `GUARD_LLM_MAX_CONCURRENCY` | `8` | — |
| `GUARD_LLM_COST_PER_1K_PROMPT_TOKENS` | `0.0006` | — |
| `GUARD_LLM_COST_PER_1K_COMPLETION_TOKENS` | `0.0024` | — |
| `GUARD_STORE_RAW_INPUT` | `True` | false keeps hash + preview only. Trace and violations still reference excerpts. |
| `GUARD_STORE_MAX_CHARS` | `4000` | — |
| `GUARD_ENCRYPT_STORED_TEXT` | `False` | at-rest encryption for stored text; the key comes from `api_key_pepper`. |
| `GUARD_ENCRYPTION_KEY` | `••••` | — |
| `GUARD_RETENTION_DAYS` | `30` | 0 disables the sweep. `POST /admin/retention` runs it on demand. |
| `GUARD_RETENTION_SWEEP_INTERVAL_S` | `21600` | — |
| `GUARD_HASH_USER_IDS` | `True` | keeps `metadata.user_id` non-reversible in the audit row. |
| `GUARD_USER_ID_HASH_SALT` | `••••` | — |
| `GUARD_METRICS_ENABLED` | `True` | — |
| `GUARD_METRICS_PATH` | `/metrics` | — |
| `GUARD_STATS_CACHE_TTL_S` | `5.0` | — |
| `GUARD_REQUEST_ID_HEADER` | `X-Request-ID` | — |
| `GUARD_HEALTHCHECK_PATH` | `/health` | — |
| `GUARD_SEED_DEMO_DATA_ON_STARTUP` | `False` | — |
| `GUARD_STRICT_MODE_REQUIRED_FOR_BLOCKING` | `False` | — |
<!-- END GENERATED -->

## Precedence and validation

1. explicit `Settings(...)` constructor args (tests, `create_app(settings=...)`);
2. real environment variables;
3. `./.env` at the project root (fixed path, no override).

Unknown `GUARD_*` variables are **ignored**, not rejected - which is why
`tests/test_docs.py` asserts that every variable named in these docs is a real field. A typo in a
runbook does not fail loudly here; it silently keeps the default.

Then the model runs its own validators, which is the part that matters operationally:

* `pipeline_request_timeout_s > 0` — fail-closed needs a bounded wait;
* `stage1+stage2+stage3 timeouts ≤ pipeline_request_timeout_s` — in `prod` this is fatal, elsewhere a
  warning, because the global ceiling wins anyway and a dev should not be blocked by it;
* thresholds in `0..1`, `rate_limit_*` positive, `max_input_chars > min_input_chars`;
* LLM provider order entries must be providers this code knows about;
* `environment=prod` + an empty `bootstrap_api_keys`/default pepper combination is rejected.

A config mistake therefore cannot survive startup, which is the whole point: a moderation service
that boots with `stage1_threshold=1.5` would pass everything forever.

## Profiles worth copying

**Fully offline (no LLM at all).** The rule layer carries the load; every response says
`flags.degraded=true` so a client can tell rule-only from model-backed.

```bash
export GUARD_LLM_FALLBACK_ORDER=
export GUARD_STAGE1_USE_LLM_JUDGE=false
export GUARD_STAGE2_ALLOW_LLM_REWRITE=false
```

**Public chat product.** Strictest inputs, no raw text retention, everything traceable.

```bash
export GUARD_ENVIRONMENT=prod
export GUARD_DEFAULT_MODE=strict
export GUARD_STORE_RAW_INPUT=false
export GUARD_RETENTION_DAYS=14
```

**Reviewer-assisted queue** (a human reads the flags): `balanced` mode plus `store_trace=true`
so `/explain/{request_id}` can show *why* the row landed in the queue.

**Assistant-style deployment** (this service also answers, not just cleans):

```bash
export GUARD_STAGE3_MODE=assistant   # output is generated, so stage3_output policies fully apply
```

## Privacy knobs

`store_raw_input`, `store_max_chars`, `encrypt_stored_text`, `hash_user_ids` +
`user_id_hash_salt`, `log_request_bodies`, `retention_days`, `max_log_field_chars`. The defaults
keep enough context to review a decision (excerpt, category, policy id, scores) without keeping a
copy of every user's message body. Turning `store_raw_input` off does **not** weaken moderation —
detection happens in-process before anything is written — it only changes what the audit trail can
show afterwards.
