"""Application configuration.

Everything is env-driven (``GUARD_*``) with safe defaults so the service can boot
with zero external dependencies: SQLite for storage, the ``local`` LLM provider and
fail-closed moderation.

Two invariants are enforced here rather than at the call sites:

* **Fail closed** - any setting that would silently disable safety (e.g. an empty
  global timeout, a negative threshold) is rejected at import time.
* **Secrets are never logged** - API keys live on this object and are only ever
  read by :mod:`app.pipeline.llm`; ``repr()``/``model_dump()`` mask them.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALID_PROVIDERS = ("auto", "local", "ollama", "openai", "gemini", "anthropic")
SecretFields = ("openai_api_key", "gemini_api_key", "anthropic_api_key", "bootstrap_api_keys")


class ModeThresholds(BaseModel):
    """Score cut-offs used to combine Stage-1 detectors into a decision."""

    toxicity_block: float = 0.72
    injection_block: float = 0.60
    jailbreak_block: float = 0.60
    sensitive_topic_flag: float = 0.60
    stage1_consensus: Literal["any", "both"] = "any"
    llm_weight: float = 0.6
    classifier_weight: float = 0.4

    @field_validator(
        "toxicity_block", "injection_block", "jailbreak_block", "sensitive_topic_flag", "llm_weight", "classifier_weight"
    )
    @classmethod
    def _unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("thresholds and weights must live in [0, 1]")
        return round(float(v), 4)


class Settings(BaseSettings):
    """Runtime configuration for the moderation gateway."""

    model_config = SettingsConfigDict(
        env_file=os.getenv("GUARD_ENV_FILE", str(PROJECT_ROOT / ".env")),
        env_prefix="GUARD_",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------------------------------------------------------- metadata
    app_name: str = "Guardrail-Enforced Content Moderation API"
    version: str = "1.0.0"
    environment: Literal["dev", "staging", "prod"] = "dev"
    api_v1_prefix: str = ""

    # ------------------------------------------------------------------ server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = False

    # ----------------------------------------------------------------- logging
    log_level: str = "INFO"
    log_json: bool = True
    log_request_bodies: bool = False  # keep OFF in prod (raw user text is sensitive)
    max_log_field_chars: int = 2000

    # ---------------------------------------------------------------- database
    database_url: str = "sqlite+aiosqlite:///./data/moderation.db"
    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_create_all_on_startup: bool = True

    # -------------------------------------------------------------------- auth
    auth_enabled: bool = True
    api_key_header: str = "X-API-Key"
    # Pepper mixed into the SHA-256 so a stolen DB cannot be rainbow-tabled.
    api_key_pepper: SecretStr = SecretStr("change-me-in-prod")
    # Comma separated keys created on first boot (dev convenience / cold start).
    bootstrap_api_keys: SecretStr = SecretStr("")
    allow_query_string_api_key: bool = False  # never recommended: URLs leak into logs

    # ------------------------------------------------------------- rate limits
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 120
    rate_limit_burst: int = 20
    rate_limit_backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"

    # ------------------------------------------------------------------ inputs
    max_input_chars: int = 8000
    min_input_chars: int = 1
    normalize_unicode: bool = True
    strip_control_chars: bool = True
    max_metadata_items: int = 16
    max_metadata_key_chars: int = 64
    max_metadata_value_chars: int = 256

    # ---------------------------------------------------------------- pipeline
    default_mode: Literal["strict", "balanced", "lenient"] = "balanced"
    policies_path: str = str(PROJECT_ROOT / "config" / "policies.yaml")
    pipeline_request_timeout_s: float = 60.0
    stage1_timeout_s: float = 20.0
    stage2_timeout_s: float = 25.0
    stage3_timeout_s: float = 45.0
    store_trace: bool = True
    trace_preview_chars: int = 400

    # Stage 1
    stage1_use_classifier: bool = True
    stage1_classifier_model: str = "protectai/deberta-v3-base-prompt-injection-v2"
    stage1_classifier_device: Literal["auto", "cpu", "cuda"] = "auto"
    stage1_use_llm_judge: bool = True
    stage1_fail_closed_on_llm_error: bool = True

    # Stage 2
    use_guardrails: bool = False  # optional Guardrails.ai validator layer
    stage2_use_presidio: bool = True
    stage2_allow_llm_rewrite: bool = True

    # Stage 3
    stage3_mode: Literal["passthrough", "assistant"] = "passthrough"
    stage3_output_check: bool = True

    # --------------------------------------------------------------------- LLM
    llm_provider: Literal["auto", "local", "ollama", "openai", "gemini", "anthropic"] = "auto"
    llm_fallback_order: list[str] = Field(default_factory=lambda: ["gemini", "openai", "anthropic", "ollama", "local"])
    openai_api_key: SecretStr = SecretStr("")
    gemini_api_key: SecretStr = SecretStr("")
    anthropic_api_key: SecretStr = SecretStr("")
    ollama_base_url: str = "http://localhost:11434"
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    small_model: str = "gemini-2.0-flash-lite"  # Stage 1 judge
    main_model: str = "gemini-2.0-flash"  # Stage 3 generation
    ollama_small_model: str = "llama3.2:1b"
    ollama_main_model: str = "llama3.1:8b"
    openai_small_model: str = "gpt-4o-mini"
    openai_main_model: str = "gpt-4o"
    anthropic_small_model: str = "claude-3-5-haiku-latest"
    anthropic_main_model: str = "claude-3-5-sonnet-latest"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_request_timeout_s: float = 30.0
    llm_max_retries: int = 2
    llm_retry_backoff_s: float = 0.4
    llm_max_concurrency: int = 8
    llm_cost_per_1k_prompt_tokens: float = 0.0006
    llm_cost_per_1k_completion_tokens: float = 0.0024

    # ------------------------------------------------------------ post / store
    store_raw_input: bool = True
    store_max_chars: int = 4000
    encrypt_stored_text: bool = False
    encryption_key: SecretStr = SecretStr("")  # Fernet key, only needed when encrypting
    retention_days: int = 30
    retention_sweep_interval_s: int = 6 * 3600
    hash_user_ids: bool = True
    user_id_hash_salt: SecretStr = SecretStr("moderation-demo-salt")

    # -------------------------------------------------------------- monitoring
    metrics_enabled: bool = True
    metrics_path: str = "/metrics"
    stats_cache_ttl_s: float = 5.0

    # ------------------------------------------------------------------- misc
    request_id_header: str = "X-Request-ID"
    healthcheck_path: str = "/health"
    seed_demo_data_on_startup: bool = False
    strict_mode_required_for_blocking: bool = False

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("llm_fallback_order", mode="before")
    @classmethod
    def _split_fallback(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("llm_provider")
    @classmethod
    def _known_provider(cls, v: str) -> str:
        if v not in VALID_PROVIDERS:
            raise ValueError(f"llm_provider must be one of {VALID_PROVIDERS}")
        return v

    @model_validator(mode="after")
    def _safety_invariants(self) -> Settings:
        if self.pipeline_request_timeout_s <= 0:
            raise ValueError("pipeline_request_timeout_s must be > 0 (fail-closed needs a bounded wait)")
        stage_sum = self.stage1_timeout_s + self.stage2_timeout_s + self.stage3_timeout_s
        if stage_sum > self.pipeline_request_timeout_s:
            # Not fatal (the global timeout wins) but worth failing loudly in CI.
            if self.environment == "prod":
                raise ValueError("sum of stage timeouts exceeds pipeline_request_timeout_s in prod")
        if self.max_input_chars < 64:
            raise ValueError("max_input_chars is unrealistically small")
        if self.llm_max_retries < 0 or self.llm_request_timeout_s <= 0:
            raise ValueError("llm_max_retries must be >= 0 and llm_request_timeout_s > 0")
        if self.environment == "prod":
            if self.api_key_pepper.get_secret_value() in {"", "change-me-in-prod"}:
                raise ValueError("GUARD_API_KEY_PEPPER must be set in prod")
            if "*" in self.cors_origins:
                raise ValueError("CORS wildcard is not allowed in prod")
        return self

    # --------------------------------------------------------------- helpers
    @property
    def policies_file(self) -> Path:
        return Path(self.policies_path)

    @property
    def mode_thresholds(self) -> dict[str, ModeThresholds]:
        """Stage-1 thresholds come from ``policies.yaml`` (single source of truth)."""
        from app.pipeline.policy import get_policy_bundle  # local import: avoids a cycle

        return get_policy_bundle(self.policies_file).mode_thresholds

    def resolved_provider(self) -> str:
        """Turn ``auto`` into a concrete provider based on available credentials."""
        if self.llm_provider != "auto":
            return self.llm_provider
        for name, secret in (
            ("gemini", self.gemini_api_key),
            ("openai", self.openai_api_key),
            ("anthropic", self.anthropic_api_key),
        ):
            if secret and secret.get_secret_value().strip():
                return name
        return "local"

    def has_provider(self, provider: str) -> bool:
        if provider == "local":
            return True
        if provider == "ollama":
            return bool(self.ollama_base_url)
        if provider == "openai":
            return bool(self.openai_api_key.get_secret_value().strip())
        if provider == "anthropic":
            return bool(self.anthropic_api_key.get_secret_value().strip())
        if provider == "gemini":
            return bool(self.gemini_api_key.get_secret_value().strip())
        return False

    def model_dump_masked(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        for field in SecretFields:
            if field in data:
                data[field] = "***"
        if "api_key_pepper" in data:
            data["api_key_pepper"] = "***"
        if "encryption_key" in data:
            data["encryption_key"] = "***" if data["encryption_key"] else ""
        return data

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<Settings env={self.environment} db={self.database_url.split('://')[0]} provider={self.resolved_provider()}>"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor - import this everywhere instead of instantiating Settings."""
    return Settings()


def reload_settings() -> Settings:  # pragma: no cover - test helper
    get_settings.cache_clear()
    return get_settings()
