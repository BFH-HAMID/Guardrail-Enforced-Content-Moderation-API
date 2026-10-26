"""Provider-agnostic LLM access for every stage.

One router (:class:`LLMRouter`) exposes ``chat()`` / ``chat_json()`` and knows how to:

* talk to **Gemini**, **OpenAI**, **Anthropic**, **Ollama** over HTTP (httpx, async),
* fall back through ``llm_fallback_order`` when a provider errors, and
* use the built-in **local** engine when no credentials exist, so the API stays fully
  functional offline (and in CI) instead of pretending a call happened.

Reliability contract: bounded timeouts, bounded retries with exponential backoff +
jitter, a concurrency semaphore, and a hard distinction between
:class:`LLMNotConfigured` (a capability the deployment does not have - callers may
degrade deliberately) and :class:`LLMError` (a call that failed - callers must fail
closed). Silently treating "the judge was unreachable" as "the content is safe" is
the exact bug this file exists to prevent.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
FIRST_BRACE_RE = re.compile(r"\{")


class LLMError(RuntimeError):
    """A provider call failed. Callers fail closed on this."""

    retryable = False

    def __init__(self, message: str, *, status_code: int | None = None, raw: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.raw = raw[:400]


class LLMTransientError(LLMError):
    retryable = True


class LLMNotConfigured(LLMError):
    """No credentials / no local engine for this capability."""

    retryable = False


@dataclass(slots=True)
class ChatMessage:
    role: str  # system | user | assistant
    content: str


@dataclass(slots=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    attempts: int = 1
    finish_reason: str = "stop"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_cost_usd(self) -> float:
        s = get_settings()
        return round(
            self.prompt_tokens / 1000.0 * s.llm_cost_per_1k_prompt_tokens
            + self.completion_tokens / 1000.0 * s.llm_cost_per_1k_completion_tokens,
            6,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "attempts": self.attempts,
            "estimated_cost_usd": self.estimated_cost_usd,
            "finish_reason": self.finish_reason,
        }


def estimate_tokens(text: str) -> int:
    """Cheap token proxy (~4 chars/token, +40% for non-Latin scripts)."""
    if not text:
        return 0
    non_ascii = sum(1 for ch in text if ord(ch) > 0x2FFF)
    base = len(text) / 4.0
    return max(1, int(base + (non_ascii * 0.4)))


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a completion (handles fences + prose)."""
    if not text:
        return None
    candidate = text.strip()
    fenced = JSON_FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = FIRST_BRACE_RE.search(candidate)
    if not start:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start.start(), len(candidate)):
        ch = candidate[index]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = candidate[start.start() : index + 1]
                try:
                    parsed = json.loads(chunk)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


# --------------------------------------------------------------------- providers


class Provider(Protocol):
    name: str

    async def chat(self, messages: list[ChatMessage], *, model: str, temperature: float, max_tokens: int, json_mode: bool, timeout: float, settings: Settings) -> LLMResponse: ...

    def model_for(self, settings: Settings, tier: str) -> str: ...


class _HTTPProvider:
    """Shared retry/backoff/timeout machinery for the network providers."""

    name = "http"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._semaphore = asyncio.Semaphore(max(1, settings.llm_max_concurrency))

    # -- helpers ---------------------------------------------------------------
    def _backoff(self, attempt: int) -> float:
        base = self.settings.llm_retry_backoff_s * (2**attempt)
        return base * (0.6 + random.random() * 0.8)

    async def _post_with_retry(self, url: str, *, json_body: dict[str, Any], headers: dict[str, str], timeout: float) -> tuple[dict[str, Any], int]:
        attempts = max(1, self.settings.llm_max_retries + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                async with self._semaphore:
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        response = await client.post(url, json=json_body, headers=headers)
                if response.status_code < 400:
                    try:
                        return response.json(), attempt + 1
                    except ValueError as exc:
                        raise LLMTransientError(f"non-JSON success body from {self.name}: {exc}", raw=response.text[:200]) from exc
                if response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
                    last_error = LLMTransientError(
                        f"{self.name} returned {response.status_code}", status_code=response.status_code, raw=response.text
                    )
                else:
                    raise LLMError(
                        f"{self.name} rejected the request ({response.status_code})",
                        status_code=response.status_code,
                        raw=response.text,
                    )
            except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as exc:
                last_error = LLMTransientError(f"{self.name} transport error: {exc.__class__.__name__}", raw=str(exc))
            except LLMError:
                raise
            except Exception as exc:  # pragma: no cover - unexpected client failure
                last_error = LLMTransientError(f"{self.name} client error: {exc}", raw=str(exc))
            if attempt + 1 < attempts:
                delay = self._backoff(attempt)
                retry_after = None
                if isinstance(last_error, LLMTransientError) and last_error.raw:
                    match = re.search(r"retry-after\"?\s*:\s*\"?(\d{1,3})", last_error.raw, re.IGNORECASE)
                    if match:
                        retry_after = min(float(match.group(1)), 10.0)
                await asyncio.sleep(retry_after if retry_after is not None else delay)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _usage(payload: dict[str, Any], *paths: tuple[str, str]) -> tuple[int, int]:
        prompt = completion = 0
        for p_key, c_key in paths:
            section = payload.get(p_key)
            if isinstance(section, dict):
                prompt = int(section.get(c_key) or 0)
        return prompt, completion


class OpenAIProvider(_HTTPProvider):
    name = "openai"

    def model_for(self, settings: Settings, tier: str) -> str:
        return settings.openai_main_model if tier == "main" else settings.openai_small_model

    async def chat(self, messages, *, model, temperature, max_tokens, json_mode, timeout, settings) -> LLMResponse:
        if not settings.openai_api_key.get_secret_value().strip():
            raise LLMNotConfigured("openai: no API key configured")
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        payload, attempts = await self._post_with_retry(f"{settings.openai_base_url.rstrip('/')}/chat/completions", json_body=body, headers=headers, timeout=timeout)
        choice = (payload.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        usage = payload.get("usage") or {}
        return LLMResponse(
            text=text,
            provider=self.name,
            model=model,
            prompt_tokens=int(usage.get("prompt_tokens") or estimate_tokens("".join(m.content for m in messages))),
            completion_tokens=int(usage.get("completion_tokens") or estimate_tokens(text)),
            latency_ms=(time.perf_counter() - started) * 1000,
            attempts=attempts,
            finish_reason=choice.get("finish_reason") or "stop",
        )


class AnthropicProvider(_HTTPProvider):
    name = "anthropic"

    def model_for(self, settings: Settings, tier: str) -> str:
        return settings.anthropic_main_model if tier == "main" else settings.anthropic_small_model

    async def chat(self, messages, *, model, temperature, max_tokens, json_mode, timeout, settings) -> LLMResponse:
        if not settings.anthropic_api_key.get_secret_value().strip():
            raise LLMNotConfigured("anthropic: no API key configured")
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns: list[dict[str, Any]] = [
            {"role": "assistant" if m.role == "assistant" else "user", "content": m.content} for m in messages if m.role != "system"
        ]
        if json_mode:
            turns.append({"role": "assistant", "content": "{"})
            system = (system + "\n\nRespond with a single JSON object and nothing else.").strip()
        body = {
            "model": model,
            "system": system,
            "messages": turns,
            "max_tokens": max_tokens + (1 if json_mode else 0),
            "temperature": temperature,
        }
        headers = {
            "x-api-key": settings.anthropic_api_key.get_secret_value(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        started = time.perf_counter()
        payload, attempts = await self._post_with_retry(f"{settings.anthropic_base_url.rstrip('/')}/messages", json_body=body, headers=headers, timeout=timeout)
        parts = [block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"]
        text = "".join(parts).strip()
        if json_mode and text and not text.endswith("}"):
            text = "{" + text
        text = text.lstrip()
        usage = payload.get("usage") or {}
        return LLMResponse(
            text=text,
            provider=self.name,
            model=model,
            prompt_tokens=int(usage.get("input_tokens") or estimate_tokens("".join(m.content for m in messages))),
            completion_tokens=int(usage.get("output_tokens") or estimate_tokens(text)),
            latency_ms=(time.perf_counter() - started) * 1000,
            attempts=attempts,
            finish_reason=payload.get("stop_reason") or "stop",
        )


class GeminiProvider(_HTTPProvider):
    name = "gemini"

    def model_for(self, settings: Settings, tier: str) -> str:
        return settings.main_model if tier == "main" else settings.small_model

    async def chat(self, messages, *, model, temperature, max_tokens, json_mode, timeout, settings) -> LLMResponse:
        if not settings.gemini_api_key.get_secret_value().strip():
            raise LLMNotConfigured("gemini: no API key configured")
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        contents = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in messages
            if m.role != "system"
        ]
        generation_config: dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if json_mode:
            generation_config["responseMimeType"] = "application/json"
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
            "safetySettings": [
                {"category": cat, "threshold": "BLOCK_ONLY_HIGH"}
                for cat in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
            ],
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"{settings.gemini_base_url.rstrip('/')}/models/{model}:generateContent"
        headers = {
            "x-goog-api-key": settings.gemini_api_key.get_secret_value(),
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        payload, attempts = await self._post_with_retry(url, json_body=body, headers=headers, timeout=timeout)
        candidate = (payload.get("candidates") or [{}])[0]
        parts = candidate.get("content", {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts).strip()
        if not text:
            blocked = payload.get("promptFeedback", {}).get("blockReason") or candidate.get("finishReason")
            raise LLMError(
                f"gemini returned no text (blocked={blocked})",
                raw=json.dumps(payload)[:300],
            )
        usage = payload.get("usageMetadata") or {}
        return LLMResponse(
            text=text,
            provider=self.name,
            model=model,
            prompt_tokens=int(usage.get("promptTokenCount") or estimate_tokens("".join(m.content for m in messages))),
            completion_tokens=int(usage.get("candidatesTokenCount") or estimate_tokens(text)),
            latency_ms=(time.perf_counter() - started) * 1000,
            attempts=attempts,
            finish_reason=candidate.get("finishReason") or "STOP",
        )


class OllamaProvider(_HTTPProvider):
    """Local/self-hosted models: Llama-Guard-3-1B, Qwen2.5-1.5B, llama3.2:1b ..."""

    name = "ollama"

    def model_for(self, settings: Settings, tier: str) -> str:
        return settings.ollama_main_model if tier == "main" else settings.ollama_small_model

    async def chat(self, messages, *, model, temperature, max_tokens, json_mode, timeout, settings) -> LLMResponse:
        if not settings.ollama_base_url:
            raise LLMNotConfigured("ollama: no base url configured")
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system" if m.role == "system" else ("assistant" if m.role == "assistant" else "user"), "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            body["format"] = "json"
        started = time.perf_counter()
        payload, attempts = await self._post_with_retry(
            f"{settings.ollama_base_url.rstrip('/')}/api/chat", json_body=body, headers={"Content-Type": "application/json"}, timeout=timeout
        )
        text = ((payload.get("message") or {}).get("content") or "").strip()
        return LLMResponse(
            text=text,
            provider=self.name,
            model=model,
            prompt_tokens=int(payload.get("prompt_eval_count") or estimate_tokens("".join(m.content for m in messages))),
            completion_tokens=int(payload.get("eval_count") or estimate_tokens(text)),
            latency_ms=(time.perf_counter() - started) * 1000,
            attempts=attempts,
            finish_reason=payload.get("done_reason") or "stop",
        )


class LocalProvider:
    """Offline engine used when no provider credentials exist.

    It is *not* a language model and never claims to be one. It gives the pipeline a
    deterministic, auditable completion so the service degrades to "rules + masking"
    instead of failing, and it reports ``provider="local"`` in every trace so the
    Streamlit demo and the logs make the difference obvious to the operator.
    """

    name = "local"

    def model_for(self, settings: Settings, tier: str) -> str:
        return "local-deterministic-v1"

    async def chat(self, messages, *, model, temperature, max_tokens, json_mode, timeout, settings) -> LLMResponse:
        started = time.perf_counter()
        user_turns = [m.content for m in messages if m.role != "system"]
        prompt = "\n\n".join(user_turns)
        system = "\n".join(m.content for m in messages if m.role == "system")
        if json_mode:
            # A JSON completion is asked for: return an explicitly abstaining object so
            # the caller knows no judgement was made, rather than inventing one.
            text = json.dumps(
                {
                    "engine": "local-deterministic",
                    "available": False,
                    "note": "No LLM provider configured; the judge could not run. "
                    "Callers must fall back to deterministic detectors or fail closed.",
                }
            )
        elif "text sanitiser" in system:
            text = json.dumps({"rewritten_text": None, "available": False, "engine": "local-deterministic"})
        else:
            text = (
                "[offline mode - no generation model configured] The request was accepted and "
                "sanitised by the moderation pipeline; set a provider API key to enable Stage-3 "
                f"generation. Request gist: {prompt.strip()[:240] or 'empty'}"
            )
        await asyncio.sleep(0)  # keep the call genuinely awaitable/cooperative
        return LLMResponse(
            text=text,
            provider=self.name,
            model=model,
            prompt_tokens=estimate_tokens(prompt + system),
            completion_tokens=estimate_tokens(text),
            latency_ms=(time.perf_counter() - started) * 1000,
            attempts=1,
            finish_reason="local",
        )


PROVIDER_CLASSES: dict[str, type] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
    "local": LocalProvider,
}


class LLMRouter:
    """Pick a provider, honour fallbacks, cache instances, expose tiered models."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._cache: dict[str, Any] = {}
        self.last_error: str | None = None

    # ------------------------------------------------------------------ wiring
    def provider(self, name: str) -> Any:
        if name not in PROVIDER_CLASSES:
            raise LLMError(f"unknown provider {name!r}")
        if name not in self._cache:
            self._cache[name] = PROVIDER_CLASSES[name](self.settings) if name != "local" else LocalProvider()
        return self._cache[name]

    def candidate_providers(self, preferred: str | None = None) -> list[str]:
        """Order of providers to try. Configured ones first, ``local`` last."""
        s = self.settings
        if s.llm_provider != "auto":
            primary = s.llm_provider
        else:
            primary = preferred or s.resolved_provider()
        order = [primary]
        for name in s.llm_fallback_order:
            if name not in order:
                order.append(name)
        if "local" not in order:
            order.append("local")
        return [name for name in order if s.has_provider(name) or name == "local"] or ["local"]

    def model_for(self, provider_name: str, tier: str) -> str:
        return self.provider(provider_name).model_for(self.settings, tier)

    # -------------------------------------------------------------------- call
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tier: str = "main",
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        timeout: float | None = None,
    ) -> LLMResponse:
        """Chat with fallback. Raises :class:`LLMError` only when *every* provider fails."""
        s = self.settings
        temperature = s.llm_temperature if temperature is None else temperature
        max_tokens = s.llm_max_tokens if max_tokens is None else max_tokens
        timeout = s.llm_request_timeout_s if timeout is None else timeout
        errors: dict[str, str] = {}
        for name in ([provider] if provider else self.candidate_providers()):
            active = name or s.resolved_provider()
            try:
                impl = self.provider(active)
                chosen_model = model or self.model_for(active, tier)
                response = await impl.chat(
                    messages,
                    model=chosen_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    timeout=timeout,
                    settings=s,
                )
                response.raw["errors"] = errors or {}
                return response
            except LLMNotConfigured as exc:
                errors[active] = f"not-configured: {exc}"
                continue
            except LLMError as exc:
                errors[active] = str(exc)[:200]
                log.warning("llm provider failed, trying fallback", extra={"provider": active, "error": str(exc)[:200], "remaining": True})
                continue
            except Exception as exc:  # pragma: no cover - defensive
                errors[active] = f"{exc.__class__.__name__}: {exc}"
                continue
        self.last_error = json.dumps(errors)[:500]
        raise LLMError(f"all LLM providers failed: {self.last_error}")

    async def chat_json(
        self,
        messages: list[ChatMessage],
        *,
        tier: str = "main",
        provider: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        retries_on_parse_error: int = 1,
    ) -> tuple[dict[str, Any], LLMResponse]:
        """Chat + strict JSON extraction, with an explicit reformat retry."""
        attempts = max(1, retries_on_parse_error + 1)
        last: LLMResponse | None = None
        for attempt in range(attempts):
            response = await self.chat(messages, tier=tier, provider=provider, model=model, json_mode=True, timeout=timeout)
            last = response
            data = extract_json(response.text)
            if data is not None:
                return data, response
            log.warning(
                "llm returned unparsable json",
                extra={"attempt": attempt + 1, "provider": response.provider, "preview": response.text[:160]},
            )
            if attempt + 1 < attempts:
                messages = messages + [
                    ChatMessage("assistant", response.text[:1000]),
                    ChatMessage("user", "That was not valid JSON. Reply again with ONLY the JSON object, no prose, no code fences."),
                ]
        raise LLMError("llm_json_parse_failed", raw=(last.text[:200] if last else ""))


_ROUTER: LLMRouter | None = None


def get_router(settings: Settings | None = None) -> LLMRouter:
    global _ROUTER
    if _ROUTER is None or (settings is not None and _ROUTER.settings is not settings):
        _ROUTER = LLMRouter(settings)
    return _ROUTER


def reset_router() -> None:  # pragma: no cover - test helper
    global _ROUTER
    _ROUTER = None
