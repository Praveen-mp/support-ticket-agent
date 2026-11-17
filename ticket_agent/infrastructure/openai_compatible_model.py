"""JsonModel for OpenAI-compatible chat APIs (used with Groq's free tier).

Plain httpx plus the same RateLimiter and RetryPolicy as the helpdesk client, so both
external dependencies are throttled and retried the same way.
Groq's free tier allows about 30 requests and 8,000 tokens per minute; we space requests
under the request limit and honour Retry-After when the token limit is hit.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import Any

import httpx

from ticket_agent.application.errors import ModelError
from ticket_agent.domain.models import Usage
from ticket_agent.infrastructure.llm import Effort, Pricing, parse_json_object
from ticket_agent.infrastructure.resilience import RateLimiter, RetryPolicy

logger = logging.getLogger(__name__)

# Strict-schema decoding occasionally fails server-side; a fresh attempt usually works.
_RETRYABLE_400_CODES = {"json_validate_failed"}


class OpenAICompatibleJsonModel:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        effort: Effort,
        pricing: Pricing,
        rate_limiter: RateLimiter,
        retry_policy: RetryPolicy,
        max_tokens: int = 2500,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            transport=transport,
        )
        self._model = model
        self._effort: Effort = effort
        self._pricing = pricing
        self._limiter = rate_limiter
        self._retry = retry_policy
        self._max_tokens = max_tokens
        self._sleep = sleep
        self._rng = rng or random.Random()

    def call(self, system: str, user: str, schema: dict[str, Any]) -> tuple[dict[str, Any], Usage]:
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": schema, "strict": True},
            },
            "reasoning_effort": self._effort,
            "max_completion_tokens": self._max_tokens,
        }
        data = self._post_with_retries(body)
        usage = self._usage_of(data.get("usage") or {})
        choice = (data.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length":
            raise ModelError("model output hit max tokens before finishing", usage.cost_usd)
        content = (choice.get("message") or {}).get("content") or ""
        return parse_json_object(content, usage), usage

    def _post_with_retries(self, body: dict[str, Any]) -> dict[str, Any]:
        problem = ""
        for attempt in range(1, self._retry.max_attempts + 1):
            self._limiter.acquire()
            try:
                response = self._http.post("/chat/completions", json=body)
            except httpx.TransportError as exc:
                # Re-asking a model has no side effects beyond cost, so all network errors retry.
                problem = f"{type(exc).__name__}: {exc}"
                self._wait(self._retry.backoff(attempt, self._rng), attempt, problem)
                continue

            status = response.status_code
            if status == 200:
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ModelError("model API returned a non-object response")
                return payload
            code, message = _error_details(response)
            problem = f"HTTP {status} {code}: {message}".strip()
            if status == 429:
                wait = self._retry.retry_after(
                    response.headers.get("retry-after"), attempt, self._rng
                )
                self._limiter.pause(wait)
                logger.warning("model rate limited: wait_s=%.1f attempt=%s", wait, attempt)
                continue
            if status >= 500 or code in _RETRYABLE_400_CODES:
                self._wait(self._retry.backoff(attempt, self._rng), attempt, problem)
                continue
            raise ModelError(f"model call failed: {problem}")
        raise ModelError(f"model call failed after {self._retry.max_attempts} attempts: {problem}")

    def _wait(self, delay: float, attempt: int, problem: str) -> None:
        if attempt >= self._retry.max_attempts:
            return
        logger.warning(
            "transient model error, retrying: attempt=%s delay_s=%.2f problem=%s",
            attempt,
            delay,
            problem,
        )
        self._sleep(delay)

    def _usage_of(self, raw: dict[str, Any]) -> Usage:
        prompt = int(raw.get("prompt_tokens") or 0)
        cached = int((raw.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        return self._pricing.usage(
            input_tokens=prompt - cached,
            output_tokens=int(raw.get("completion_tokens") or 0),
            cache_read_tokens=cached,
        )


def _error_details(response: httpx.Response) -> tuple[str, str]:
    try:
        error = response.json().get("error") or {}
    except (ValueError, AttributeError):
        return "", response.text[:200]
    if not isinstance(error, dict):
        return "", str(error)[:200]
    return str(error.get("code") or error.get("type") or ""), str(error.get("message") or "")[:300]
