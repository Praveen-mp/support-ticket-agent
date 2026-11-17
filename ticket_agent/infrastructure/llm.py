"""Provider-neutral LLM adapters for the TicketAssessor and ReplyVerifier ports.

The model gets no tools: it can only fill a fixed JSON schema. Whatever a ticket says, the
worst it can do is produce a wrong field value, which the policy layer then checks.
Swapping provider means swapping the JsonModel; nothing here changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from ticket_agent.application.errors import ModelError
from ticket_agent.application.ports import ModelResult
from ticket_agent.domain.faq import FaqArticle, FaqKnowledgeBase
from ticket_agent.domain.models import (
    Action,
    Assessment,
    Category,
    Language,
    Priority,
    ReplyCheck,
    SensitiveTopic,
    Ticket,
    Usage,
)
from ticket_agent.prompts import ASSESS_PROMPT, REVISE_PROMPT, VERIFY_PROMPT, load_prompt

Effort = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens. Cache factors are relative to the input price."""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_factor: float = 1.0
    cache_write_factor: float = 1.0

    def usage(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> Usage:
        """input_tokens means uncached input; cached tokens are passed separately."""
        cost = (
            input_tokens * self.input_per_mtok
            + output_tokens * self.output_per_mtok
            + cache_read_tokens * self.input_per_mtok * self.cache_read_factor
            + cache_write_tokens * self.input_per_mtok * self.cache_write_factor
        ) / 1_000_000
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cost_usd=cost,
            calls=1,
        )


class JsonModel(Protocol):
    """One model call that must return a JSON object matching `schema`."""

    def call(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> tuple[dict[str, Any], Usage]: ...


def parse_json_object(text: str, usage: Usage) -> dict[str, Any]:
    """Parses a model's JSON answer; any failure is a ModelError that keeps the cost."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelError(f"model returned invalid JSON: {exc}", usage.cost_usd) from exc
    if not isinstance(data, dict):
        raise ModelError("model returned JSON that is not an object", usage.cost_usd)
    return data


class LlmTicketAssessor:
    def __init__(self, model: JsonModel, knowledge_base: FaqKnowledgeBase) -> None:
        self._model = model
        self._system = load_prompt(ASSESS_PROMPT).replace("{{FAQ}}", knowledge_base.full_text)
        self._schema = _assessment_schema(knowledge_base.article_ids)

    def assess(self, ticket: Ticket) -> ModelResult[Assessment]:
        data, usage = self._model.call(self._system, format_ticket(ticket), self._schema)
        try:
            assessment = Assessment.model_validate(data)
        except ValidationError as exc:
            raise ModelError(f"assessment does not fit the schema: {exc}", usage.cost_usd) from exc
        return ModelResult(value=assessment, usage=usage)


class LlmReplyVerifier:
    def __init__(self, model: JsonModel) -> None:
        self._model = model
        self._system = load_prompt(VERIFY_PROMPT)

    def verify(
        self,
        ticket: Ticket,
        draft_reply: str,
        expected_language: Language,
        articles: list[FaqArticle],
    ) -> ModelResult[ReplyCheck]:
        user = (
            f"{format_ticket(ticket)}\n\n"
            f"{format_articles(articles)}\n\n"
            f"<draft_reply>\n{_neutralise(draft_reply, 'draft_reply')}\n</draft_reply>"
        )
        data, usage = self._model.call(self._system, user, VERIFY_SCHEMA)
        try:
            check = ReplyCheck.build(
                expected_language=expected_language,
                reply_language=str(data["reply_language"]),
                back_translation=str(data["back_translation"]),
                unsupported_claims=tuple(str(c) for c in data["unsupported_claims"]),
                answers_question=bool(data["answers_question"]),
            )
        except (KeyError, TypeError) as exc:
            raise ModelError(f"reply check does not fit the schema: {exc}", usage.cost_usd) from exc
        return ModelResult(value=check, usage=usage)


class LlmReplyReviser:
    def __init__(self, model: JsonModel) -> None:
        self._model = model
        self._system = load_prompt(REVISE_PROMPT)

    def revise(
        self,
        ticket: Ticket,
        draft_reply: str,
        language: Language,
        articles: list[FaqArticle],
        problems: list[str],
    ) -> ModelResult[str]:
        problem_list = "\n".join(f"- {_neutralise(p, 'problems')}" for p in problems)
        user = (
            f"{format_ticket(ticket)}\n\n"
            f"{format_articles(articles)}\n\n"
            f"<draft_reply>\n{_neutralise(draft_reply, 'draft_reply')}\n</draft_reply>\n\n"
            f"<required_language>{language.value}</required_language>\n\n"
            f"<problems>\n{problem_list}\n</problems>"
        )
        data, usage = self._model.call(self._system, user, REVISE_SCHEMA)
        draft = data.get("draft_reply")
        if not isinstance(draft, str):
            raise ModelError("revised draft does not fit the schema", usage.cost_usd)
        return ModelResult(value=draft, usage=usage)


def format_articles(articles: list[FaqArticle]) -> str:
    article_text = "\n\n".join(f"## {a.id} · {a.title}\n{a.text}" for a in articles)
    return f"<help_centre_articles>\n{article_text}\n</help_centre_articles>"


def format_ticket(ticket: Ticket) -> str:
    """Wraps ticket fields in tags. Tag look-alikes inside the text are neutralised so a ticket
    cannot close the block early and pose as instructions."""
    return (
        "<ticket>\n"
        f"channel: {_neutralise(ticket.channel, 'ticket')}\n"
        f"customer_plan: {_neutralise(ticket.customer_plan, 'ticket')}\n"
        f"subject: {_neutralise(ticket.subject, 'ticket')}\n"
        f"body:\n{_neutralise(ticket.body, 'ticket')}\n"
        "</ticket>"
    )


def _neutralise(text: str, tag: str) -> str:
    return text.replace(f"<{tag}>", f"[{tag}]").replace(f"</{tag}>", f"[/{tag}]")


def _enum_array(values: list[str]) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string", "enum": values}}


def _assessment_schema(faq_ids: tuple[str, ...]) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "language": {"type": "string", "enum": [v.value for v in Language]},
        "english_summary": {"type": "string"},
        "sensitive_topics": _enum_array([t.value for t in SensitiveTopic]),
        "faq_ids": _enum_array(list(faq_ids)),
        "fully_answered_by_faq": {"type": "boolean"},
        "is_spam_or_empty": {"type": "boolean"},
        "category": {"type": "string", "enum": [c.value for c in Category]},
        "priority": {"type": "string", "enum": [p.value for p in Priority]},
        "priority_reason": {"type": "string"},
        "proposed_action": {"type": "string", "enum": [a.value for a in Action]},
        "reason": {"type": "string"},
        "draft_reply": {"type": "string"},
        "handoff_note": {"type": "string"},
        "confidence": {"type": "number"},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "back_translation": {"type": "string"},
        "reply_language": {"type": "string", "enum": ["nl", "en", "de", "other"]},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "answers_question": {"type": "boolean"},
    },
    "required": ["back_translation", "reply_language", "unsupported_claims", "answers_question"],
    "additionalProperties": False,
}

REVISE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"draft_reply": {"type": "string"}},
    "required": ["draft_reply"],
    "additionalProperties": False,
}
