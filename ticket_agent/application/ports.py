"""Interfaces the application layer depends on. Infrastructure provides the implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ticket_agent.domain.faq import FaqArticle
from ticket_agent.domain.models import (
    Assessment,
    Decision,
    DecisionOutcome,
    Language,
    ReplyCheck,
    Ticket,
    Usage,
)


@dataclass(frozen=True)
class ModelResult[T]:
    value: T
    usage: Usage


@dataclass(frozen=True)
class RemoteTicketState:
    ticket_id: str
    handled: bool


class TicketAssessor(Protocol):
    """Reads one (masked) ticket and proposes a classification, action and draft."""

    def assess(self, ticket: Ticket) -> ModelResult[Assessment]: ...


class ReplyVerifier(Protocol):
    """Checks a draft reply against the cited FAQ articles, independently of the drafter."""

    def verify(
        self,
        ticket: Ticket,
        draft_reply: str,
        expected_language: Language,
        articles: list[FaqArticle],
    ) -> ModelResult[ReplyCheck]: ...


class ReplyReviser(Protocol):
    """Rewrites a draft once to fix named problems. The result is checked again from scratch."""

    def revise(
        self,
        ticket: Ticket,
        draft_reply: str,
        language: Language,
        articles: list[FaqArticle],
        problems: list[str],
    ) -> ModelResult[str]: ...


class Helpdesk(Protocol):
    def list_open_tickets(self, max_tickets: int) -> list[Ticket]: ...

    def get_ticket_state(self, ticket_id: str) -> RemoteTicketState: ...

    def submit_decision(self, decision: Decision) -> None: ...

    @property
    def retry_count(self) -> int:
        """Total HTTP retries so far. The runner logs the difference per ticket."""
        ...


class DecisionStore(Protocol):
    """Write-ahead record of decisions, so a re-run reuses them instead of deciding again."""

    def load(self, ticket_id: str, fingerprint: str) -> DecisionOutcome | None: ...

    def save(self, ticket_id: str, fingerprint: str, outcome: DecisionOutcome) -> None: ...

    def mark_submitted(self, ticket_id: str) -> None: ...


class RunLog(Protocol):
    def write(self, entry: dict[str, Any]) -> None: ...
