"""Groq / OpenAI-compatible adapter against a MockTransport: request shape, retries, cost."""

import json
import random

import httpx
import pytest

from ticket_agent.application.errors import ModelError
from ticket_agent.infrastructure.llm import Pricing
from ticket_agent.infrastructure.openai_compatible_model import OpenAICompatibleJsonModel
from ticket_agent.infrastructure.resilience import RateLimiter, RetryPolicy

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}


class RecordingLimiter(RateLimiter):
    def __init__(self) -> None:
        super().__init__(1000, sleep=lambda _s: None)
        self.pauses: list[float] = []

    def pause(self, seconds: float) -> None:
        self.pauses.append(seconds)


def completion(content: str, finish: str = "stop", cached: int = 0) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": 3000,
                "completion_tokens": 500,
                "prompt_tokens_details": {"cached_tokens": cached},
            },
        },
    )


def make_model(responses: list, limiter=None) -> tuple[OpenAICompatibleJsonModel, list]:
    requests: list[httpx.Request] = []
    queue: list[httpx.Response | Exception] = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    model = OpenAICompatibleJsonModel(
        base_url="https://api.groq.test/openai/v1",
        api_key="gsk_test",
        model="openai/gpt-oss-120b",
        effort="low",
        pricing=Pricing(0.15, 0.60, cache_read_factor=0.5),
        rate_limiter=limiter or RecordingLimiter(),
        retry_policy=RetryPolicy(max_attempts=4, max_retry_after_s=60),
        transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
        rng=random.Random(0),
    )
    return model, requests


def test_sends_strict_schema_and_parses_json():
    model, requests = make_model([completion('{"a": "ok"}')])
    data, usage = model.call("system text", "user text", SCHEMA)

    assert data == {"a": "ok"}
    body = json.loads(requests[0].content)
    assert requests[0].url.path == "/openai/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer gsk_test"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert body["reasoning_effort"] == "low"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert usage.cost_usd == pytest.approx((3000 * 0.15 + 500 * 0.60) / 1_000_000)


def test_cached_prompt_tokens_are_billed_at_the_cache_rate():
    model, _ = make_model([completion('{"a": "ok"}', cached=2000)])
    _, usage = model.call("s", "u", SCHEMA)
    assert usage.input_tokens == 1000
    assert usage.cache_read_tokens == 2000
    assert usage.cost_usd == pytest.approx((1000 * 0.15 + 2000 * 0.075 + 500 * 0.60) / 1e6)


def test_waits_for_retry_after_on_429():
    limiter = RecordingLimiter()
    rate_limited = httpx.Response(
        429,
        json={"error": {"code": "rate_limit_exceeded", "message": "tokens per minute"}},
        headers={"retry-after": "7"},
    )
    model, requests = make_model([rate_limited, completion('{"a": "ok"}')], limiter=limiter)
    data, _ = model.call("s", "u", SCHEMA)
    assert data == {"a": "ok"}
    assert limiter.pauses == [7.0]
    assert len(requests) == 2


def test_retries_server_errors_network_errors_and_schema_glitches():
    glitch = httpx.Response(400, json={"error": {"code": "json_validate_failed", "message": "x"}})
    model, requests = make_model(
        [httpx.Response(503), httpx.ReadTimeout("slow"), glitch, completion('{"a": "ok"}')]
    )
    assert model.call("s", "u", SCHEMA)[0] == {"a": "ok"}
    assert len(requests) == 4


def test_bad_request_is_not_retried():
    bad = httpx.Response(400, json={"error": {"code": "invalid_request", "message": "bad model"}})
    model, requests = make_model([bad])
    with pytest.raises(ModelError, match="bad model"):
        model.call("s", "u", SCHEMA)
    assert len(requests) == 1


def test_gives_up_after_max_attempts():
    model, _ = make_model([httpx.Response(503)] * 4)
    with pytest.raises(ModelError, match="after 4 attempts"):
        model.call("s", "u", SCHEMA)


def test_truncated_output_is_an_error():
    model, _ = make_model([completion('{"a": "o', finish="length")])
    with pytest.raises(ModelError, match="max tokens"):
        model.call("s", "u", SCHEMA)
