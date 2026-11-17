"""JsonModel backed by the Anthropic Messages API (structured outputs)."""

from __future__ import annotations

from typing import Any

import anthropic
from anthropic.types import MessageParam, OutputConfigParam, TextBlockParam

from ticket_agent.application.errors import ModelError
from ticket_agent.domain.models import Usage
from ticket_agent.infrastructure.llm import Effort, Pricing, parse_json_object


class AnthropicJsonModel:
    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        effort: Effort,
        pricing: Pricing,
        max_tokens: int = 4000,
    ) -> None:
        self._client = client
        self._model = model
        self._effort: Effort = effort
        self._pricing = pricing
        self._max_tokens = max_tokens

    def call(self, system: str, user: str, schema: dict[str, Any]) -> tuple[dict[str, Any], Usage]:
        # The system prompt is identical for every ticket, so it is marked cacheable.
        system_blocks: list[TextBlockParam] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
        messages: list[MessageParam] = [{"role": "user", "content": user}]
        output_config: OutputConfigParam = {
            "effort": self._effort,
            "format": {"type": "json_schema", "schema": schema},
        }
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_blocks,
                messages=messages,
                output_config=output_config,
            )
        except anthropic.APIError as exc:
            raise ModelError(f"model call failed: {type(exc).__name__}: {exc}") from exc

        raw = response.usage
        usage = self._pricing.usage(
            input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens,
            cache_read_tokens=raw.cache_read_input_tokens or 0,
            cache_write_tokens=raw.cache_creation_input_tokens or 0,
        )
        if response.stop_reason == "refusal":
            raise ModelError("model refused the request", usage.cost_usd)
        if response.stop_reason == "max_tokens":
            raise ModelError("model output hit max_tokens before finishing", usage.cost_usd)
        text = "".join(block.text for block in response.content if block.type == "text")
        return parse_json_object(text, usage), usage
