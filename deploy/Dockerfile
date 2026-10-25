# syntax=docker/dockerfile:1
# Guardrail-Enforced Content Moderation API
#
# Build from the REPO ROOT (the code imports `app.*`, so the context has to be the root):
#   docker build -f deploy/Dockerfile -t guardrail-api:local .
#
# The Stage 1 classifier (transformers + torch) is intentionally NOT installed here: it is a ~2 GB
# layer that most deployments do not need, because the rule layer works without it. Enable it with
# the optional build arg below, which switches to the ML base image.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# curl is only used by HEALTHCHECK; drop it in the final layer if that matters to you.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

# --- dependencies first, so code edits do not invalidate this layer ---------------------------
COPY deploy/requirements.txt ./deploy/requirements.txt
RUN pip install -r deploy/requirements.txt

# `docker build --build-arg WITH_CLASSIFIER=true` adds the Stage 1 transformers classifier (CPU
# wheels, ~200 MB). Without it the chain still blocks on rules, but every response carries
# flags.degraded=true so an operator can see the verdict came from the rule layer only - the
# classifier is an upgrade, never a dependency of a safe verdict.
# `--build-arg WITH_UI=true` adds the Streamlit reviewer console (compose `--profile ui`).
ARG WITH_UI=false
RUN if [ "$WITH_UI" = "true" ]; then pip install --no-cache-dir "streamlit>=1.38" "pandas>=2.2" "plotly>=5.22"; fi

ARG WITH_CLASSIFIER=false
RUN if [ "$WITH_CLASSIFIER" = "true" ]; then \
      pip install --no-cache-dir "transformers>=4.44" "torch>=2.3" --index-url https://download.pytorch.org/whl/cpu; \
    fi

# --- the application --------------------------------------------------------------------------
COPY app ./app
COPY config ./config
COPY scripts ./scripts
# tests/data is the labelled corpus that POST /evaluate scores; it is copied (and *not* the rest of
# tests/) so an operator can re-measure the running container instead of trusting a stale report.
COPY tests/data ./tests/data
COPY ui ./ui
COPY pyproject.toml README.md ./

# Dedicated unprivileged user, and a writable directory for the SQLite file / metric tmpfs.
RUN useradd --create-home --uid 10001 guardrail \
    && mkdir -p /srv/data /var/log/guardrail \
    && chown -R guardrail:guardrail /srv /var/log/guardrail
USER guardrail

EXPOSE 8000

ENV GUARD_ENVIRONMENT=prod \
    GUARD_DATABASE_URL=sqlite+aiosqlite:////srv/data/guardrail.db \
    GUARD_POLICIES_PATH=/srv/app/config/policies.yaml \
    GUARD_LOG_LEVEL=INFO

# One worker by default: Stage 1's classifier and the rate limiter are per-process objects, so
# scale with replicas rather than --workers unless you move the limiter to Redis.
# Override at run time, e.g. `docker run ... guardrail-api uvicorn app.main:app --workers 4`.
# `--timeout-graceful-shutdown 30` lets in-flight moderations finish (the chain itself is capped
# at GUARD_PIPELINE_REQUEST_TIMEOUT_S, so this only needs to cover that).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "30"]

HEALTHCHECK --interval=30s --timeout=4s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

LABEL org.opencontainers.image.title="guardrail-moderation-api" \
      org.opencontainers.image.description="3-stage guardrail-enforced content moderation API (toxicity/injection -> policy + PII rewrite -> safe output)" \
      org.opencontainers.image.licenses="MIT"
