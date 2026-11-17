"""Domain types: tickets, the model's assessment, the reply check and the final decision."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_REASON_CHARS = 400


class Language(StrEnum):
    NL = "nl"
    EN = "en"
    DE = "de"


class Category(StrEnum):
    BUG = "bug"
    BILLING = "billing"
    FEATURE_REQUEST = "feature_request"
    QUESTION = "question"
    OTHER = "other"


class Priority(StrEnum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"

    @property
    def rank(self) -> int:
        """1 is the most urgent."""
        return int(self.value[1])


class Action(StrEnum):
    REPLY = "reply"
    HANDOFF = "handoff"
    CLOSE = "close"


class SensitiveTopic(StrEnum):
    """Topics the client's rules always send to a human, even if the FAQ seems to cover them."""

    REFUND_OR_DISPUTED_CHARGE = "refund_or_disputed_charge"
    CANCELLATION_OR_CONTRACT_CHANGE = "cancellation_or_contract_change"
    PERSONAL_DATA_REQUEST = "personal_data_request"
    SECURITY_INCIDENT = "security_incident"
    LEGAL_THREAT = "legal_threat"
    CHURN_THREAT = "churn_threat"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class Ticket(_Frozen):
    """A customer ticket as received from the helpdesk. Its text is untrusted data."""

    id: str = Field(min_length=1)
    subject: str = ""
    body: str = ""
    channel: str = ""
    customer_plan: str = ""
    received_at: str = ""

    @field_validator("subject", "body", "channel", "customer_plan", "received_at", mode="before")
    @classmethod
    def _none_to_empty(cls, value: Any) -> Any:
        return "" if value is None else value

    @property
    def is_blank(self) -> bool:
        return not (self.subject.strip() or self.body.strip())


class Assessment(_Frozen):
    """What the model concluded about one ticket. Advisory only: the policy has the final word."""

    language: Language
    english_summary: str
    sensitive_topics: tuple[SensitiveTopic, ...] = ()
    faq_ids: tuple[str, ...] = ()
    fully_answered_by_faq: bool
    is_spam_or_empty: bool
    category: Category
    priority: Priority
    priority_reason: str
    proposed_action: Action
    reason: str
    draft_reply: str = ""
    handoff_note: str = ""
    confidence: float

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        return min(1.0, max(0.0, value))


class ReplyCheck(_Frozen):
    """Independent check of a draft reply, done without trusting the drafting call."""

    reply_language: str
    back_translation: str
    unsupported_claims: tuple[str, ...] = ()
    answers_question: bool
    problems: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.problems

    @classmethod
    def build(
        cls,
        *,
        expected_language: Language,
        reply_language: str,
        back_translation: str,
        unsupported_claims: tuple[str, ...],
        answers_question: bool,
    ) -> ReplyCheck:
        """Derives pass/fail from the checker's observations with fixed rules, not its opinion."""
        problems: list[str] = []
        if reply_language != expected_language.value:
            problems.append(
                f"reply is in '{reply_language}', customer wrote in '{expected_language.value}'"
            )
        problems.extend(f"claim not in FAQ: {claim}" for claim in unsupported_claims)
        if not answers_question:
            problems.append("reply does not answer the customer's question")
        return cls(
            reply_language=reply_language,
            back_translation=back_translation,
            unsupported_claims=unsupported_claims,
            answers_question=answers_question,
            problems=tuple(problems),
        )


class Decision(_Frozen):
    """The one decision sent to the helpdesk for a ticket."""

    ticket_id: str
    action: Action
    language: Language
    category: Category
    priority: Priority
    reason: str = Field(min_length=1, max_length=MAX_REASON_CHARS)
    draft_reply: str | None = None
    handoff_note: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _action_has_its_text(self) -> Self:
        if self.action is Action.REPLY and not (self.draft_reply or "").strip():
            raise ValueError("a reply needs a draft_reply")
        if self.action is Action.HANDOFF and not (self.handoff_note or "").strip():
            raise ValueError("a handoff needs a handoff_note")
        return self

    def to_api_payload(self) -> dict[str, Any]:
        """Body for POST /tickets/{id}/{action}. Only sends the text field the action needs."""
        payload: dict[str, Any] = {
            "language": self.language.value,
            "category": self.category.value,
            "priority": self.priority.value,
            "reason": self.reason,
        }
        if self.action is Action.REPLY:
            payload["draft_reply"] = self.draft_reply
        if self.action is Action.HANDOFF:
            payload["handoff_note"] = self.handoff_note
        if self.confidence is not None:
            payload["confidence"] = round(self.confidence, 2)
        return payload


class Usage(_Frozen):
    """Tokens and cost of model calls. Adds up across calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            calls=self.calls + other.calls,
        )


class DecisionOutcome(_Frozen):
    """A decision plus everything needed to explain it in the log and the eval."""

    decision: Decision
    assessment: Assessment | None = None
    reply_check: ReplyCheck | None = None
    policy_notes: tuple[str, ...] = ()
    usage: Usage = Usage()
