# Deployment

Three ways to run this, in increasing order of seriousness.

## 1. Bare process (dev / a single VM)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"                      # or: pip install -r deploy/requirements.txt
export GUARD_ENVIRONMENT=dev
export GUARD_BOOTSTRAP_API_KEYS="$(python -c "import secrets; print('gmk_'+secrets.token_urlsafe(24))")"
export GUARD_API_KEY_PEPPER="$(python -c "import secrets; print(secrets.token_urlsafe(24))")"
export GUARD_DATABASE_URL="sqlite+aiosqlite:///data/guardrail.db"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`data/` must exist and be writable if you keep SQLite (`mkdir -p data`). The bootstrap key is
printed once at startup and stored only as a salted digest — put it in your client, not in git.

Smoke test:

```bash
curl -s localhost:8000/health | python -m json.tool
curl -s -X POST localhost:8000/moderate -H "X-API-Key: $GUARD_BOOTSTRAP_API_KEYS" -H 'content-type: application/json' \
  -d '{"text":"Email me at someone@example.com about the invoice","mode":"balanced"}' | python -m json.tool
```

Expect `status=rewritten` with `[EMAIL]` in `safe_output`. If you get that, the whole chain,
policy file, PII layer and post-processing are alive.

## 2. Docker

```bash
cd deploy
cp .env.example .env && $EDITOR .env        # the bootstrap key, pepper and DB password are required
docker compose up -d --build                # add  --profile ui  for the reviewer console
docker compose logs -f api | tail -20
curl -s localhost:8000/health | head -c 300
```

Build variants:

| command | what you get |
| --- | --- |
| `docker build -f deploy/Dockerfile .` | rules + optional-LLM API (~180 MB) |
| `--build-arg WITH_CLASSIFIER=true` | adds the transformers Stage 1 classifier (CPU wheels) |
| `--build-arg WITH_UI=true` | adds Streamlit so the `ui` compose profile works |

The container runs as uid 10001, needs a writable `GUARD_DATABASE_URL` target, and its healthcheck
calls `/health`. One worker by default: the rate limiter and the policy-file mtime watcher are
per-process, so scale with **replicas** (or move the limiter to Redis) rather than `--workers`.

## 3. Kubernetes notes

No manifests are shipped on purpose — your cluster's ingress, cert and secret tooling will fight a
checked-in file. What matters:

* **Readiness/liveness**: `GET /health` (200 on `ok`/`degraded`, 503 only when the DB is unreachable).
  Give `readinessProbe.failureThreshold` a few tries: the first policy load warms regexes.
* **Rolling updates**: `terminationGracePeriodSeconds` ≥ `GUARD_PIPELINE_REQUEST_TIMEOUT_S` + 5, or
  in-flight moderations get killed.
* **Config**: mount `config/policies.yaml` via a ConfigMap and set `GUARD_POLICIES_PATH`. `POST
  /admin/reload-policies` (admin scope) re-reads it without a restart; the app also hot-reloads on
  mtime change for the container's own copy.
* **Secrets**: `GUARD_BOOTSTRAP_API_KEYS`, `GUARD_API_KEY_PEPPER`, `GUARD_GEMINI_API_KEY`, the DSN —
  all env (never files in the image). The pepper is what makes the stored key digests useless on
  their own, so treat it with the same care as the keys.
* **HPA**: scale on `guardrail_moderation_latency_ms` p95 or in-flight requests, not CPU: the chain
  is I/O-bound on the LLM legs and CPU-bound on regex backtracking-free matching, which do not
  track each other.
* **PodDisruptionBudget** ≥ 1: a moderation API that is briefly unavailable should be *retryable*,
  and `/moderate` answers 503 + `Retry-After` precisely so clients treat it that way.

## Reverse proxy

`deploy/nginx.conf` is a copy-paste example: TLS termination, `/metrics` bound to loopback only,
`client_max_body_size` matched to `GUARD_MAX_INPUT_CHARS`, and a long read timeout so a slow LLM
leg does not turn into a 502 (a 502 at the proxy is a *client-visible moderation failure* even
though the service behaved correctly).

## What "healthy" means here

`GET /health` reports the components honestly, and the response is only 200 while moderation is
*safe*, not merely while it is *up*:

| condition | `/health` | `/moderate` behaviour |
| --- | --- | --- |
| DB down, rules fine | `degraded` (storage: failed) | still moderates; logs are lost and `X-Moderation-Storage: degraded` is set |
| LLM/judge down | `degraded` | rule layer only, `flags.degraded=true` |
| policy file invalid at startup | process refuses to boot | nothing is served rather than serving an unguarded endpoint |
| policy reload fails mid-life | `ok` (previous bundle still live) | `POST /admin/reload-policies` returns 500 and the old rules keep applying |
| chain raises inside the handler | `ok`/`degraded` | fail-closed: `status=blocked`, `blocked.reason=moderation_error`, HTTP 503 + `Retry-After: 5` |

The last two rows are the guarantee that matters: **there is no state in which this service returns
`allowed` because it could not check the text.**

## Alerts worth wiring

* `moderation_fail_closed_total` increasing → the chain is refusing things it could not inspect. Page for this.
* `moderation_degraded_total` / `llm_errors_total` rising together → the judge or rewrite leg is down; verdicts are rule-layer only.
* `status=blocked` with `blocked.reason=moderation_error` in the log stream → same signal, per-request.
* `pipeline_latency_seconds` p99 > `GUARD_PIPELINE_REQUEST_TIMEOUT_S * 0.8` → timeout-driven fail-closes incoming (`stage_latency_seconds` names which stage).
* block-rate per mode changing >2× day-over-day → a policy edit or a traffic shift; check `POST /evaluate` before rolling back.
