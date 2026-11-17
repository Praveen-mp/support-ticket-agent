"""Scores decisions against human labels.

Labels come only from GET /examples (written by Nordavo). The model under test never
writes or edits a label. Safety is scored first, in the same order Nordavo scores the run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict

from ticket_agent.domain.models import (
    Action,
    Category,
    DecisionOutcome,
    Language,
    Priority,
    Ticket,
)

_FAQ_REF = re.compile(r"\bF\d{2}\b")


class ExampleLabel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    language: Language
    category: Category
    priority: Priority
    action: Action
    reason: str = ""

    @property
    def faq_ids(self) -> set[str]:
        return set(_FAQ_REF.findall(self.reason))


class LabeledExample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    subject: str = ""
    body: str = ""
    channel: str = ""
    customer_plan: str = ""
    label: ExampleLabel

    def to_ticket(self) -> Ticket:
        return Ticket(
            id=self.id,
            subject=self.subject,
            body=self.body,
            channel=self.channel,
            customer_plan=self.customer_plan,
        )


@dataclass(frozen=True)
class ExampleResult:
    example_id: str
    label: ExampleLabel
    outcome: DecisionOutcome | None
    error: str = ""

    @property
    def predicted_action(self) -> Action | None:
        return self.outcome.decision.action if self.outcome else None

    @property
    def unsafe_reply(self) -> bool:
        """A reply went out where Nordavo says it should not have."""
        return self.predicted_action is Action.REPLY and self.label.action is not Action.REPLY

    @property
    def wrong_close(self) -> bool:
        """A real customer would be ignored."""
        return self.predicted_action is Action.CLOSE and self.label.action is not Action.CLOSE

    @property
    def missed_reply(self) -> bool:
        """Sent to a human although the FAQ answers it. Safe, but not useful."""
        return self.label.action is Action.REPLY and self.predicted_action is not Action.REPLY

    def field_matches(self) -> dict[str, bool]:
        if self.outcome is None:
            return dict.fromkeys(("language", "category", "priority", "action"), False)
        decision = self.outcome.decision
        return {
            "language": decision.language is self.label.language,
            "category": decision.category is self.label.category,
            "priority": decision.priority is self.label.priority,
            "action": decision.action is self.label.action,
        }

    def to_row(self) -> dict[str, Any]:
        decision = self.outcome.decision if self.outcome else None
        assessment = self.outcome.assessment if self.outcome else None
        check = self.outcome.reply_check if self.outcome else None
        return {
            "id": self.example_id,
            "expected": self.label.model_dump(mode="json"),
            "predicted": decision.model_dump(mode="json", exclude_none=True) if decision else None,
            "matches": self.field_matches(),
            "unsafe_reply": self.unsafe_reply,
            "wrong_close": self.wrong_close,
            "missed_reply": self.missed_reply,
            "faq_ids_predicted": list(assessment.faq_ids) if assessment else [],
            "model_proposed": assessment.proposed_action.value if assessment else None,
            "policy_notes": list(self.outcome.policy_notes) if self.outcome else [],
            "reply_check": check.model_dump(mode="json") if check else None,
            "cost_usd": round(self.outcome.usage.cost_usd, 6) if self.outcome else 0.0,
            "error": self.error,
        }


@dataclass
class EvalReport:
    results: list[ExampleResult] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        total = len(self.results)
        accuracy = {
            name: sum(r.field_matches()[name] for r in self.results)
            for name in ("language", "category", "priority", "action")
        }
        replies = [r for r in self.results if r.predicted_action is Action.REPLY]
        reply_language_ok = sum(
            1
            for r in replies
            if r.outcome
            and r.outcome.reply_check
            and r.outcome.reply_check.reply_language == r.label.language.value
        )
        faq_cases = [r for r in self.results if r.label.action is Action.REPLY and r.label.faq_ids]
        faq_match = sum(
            1
            for r in faq_cases
            if r.outcome
            and r.outcome.assessment
            and r.label.faq_ids <= set(r.outcome.assessment.faq_ids)
        )
        cost = sum(r.outcome.usage.cost_usd for r in self.results if r.outcome)
        decided = sum(1 for r in self.results if r.outcome)
        return {
            "examples": total,
            "errors": total - decided,
            "unsafe_replies": sum(r.unsafe_reply for r in self.results),
            "wrong_closes": sum(r.wrong_close for r in self.results),
            "missed_replies": sum(r.missed_reply for r in self.results),
            "accuracy": {k: f"{v}/{total}" for k, v in accuracy.items()},
            "replies_in_right_language": f"{reply_language_ok}/{len(replies)}",
            "faq_article_match": f"{faq_match}/{len(faq_cases)}",
            "cost_usd_total": round(cost, 6),
            "cost_usd_per_100_tickets": round(cost / decided * 100, 4) if decided else None,
        }
