"""Test doubles. The fake helpdesk is a working in-memory server behind httpx.MockTransport,
so client tests exercise real HTTP handling (status codes, headers, JSON, timeouts)."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from ticket_agent.application.ports import ModelResult
from ticket_agent.domain.faq import FaqArticle, FaqKnowledgeBase
from ticket_agent.domain.models import (
    Action,
    Assessment,
    Category,
    Language,
    Priority,
    ReplyCheck,
    Ticket,
    Usage,
)

BASE_URL = "https://helpdesk.test/functions/v1/helpdesk"
FAQ_PATH = Path(__file__).resolve().parent.parent / "data" / "faq.md"
_REQUIRED = {"language", "category", "priority", "reason"}
_ACTION_FIELD = {"reply": "draft_reply", "handoff": "handoff_note"}


def load_kb() -> FaqKnowledgeBase:
    return FaqKnowledgeBase.from_file(FAQ_PATH)


def ticket_dict(
    ticket_id: str, subject: str = "Vraag", body: str = "Hoe werkt dit?"
) -> dict[str, Any]:
    return {
        "id": ticket_id,
        "received_at": "2026-10-05T09:02:00+00:00",
        "channel": "email",
        "customer_plan": "Basis",
        "subject": subject,
        "body": body,
    }


Fault = Callable[[httpx.Request], httpx.Response | None]


class FakeHelpdeskServer:
    """Behaves like the documented API: pagination, 409 on a second answer, 422 on bad bodies.

    Faults can be injected per call. `random_503_rate` mimics the real API's random failures.
    """

    def __init__(self, tickets: list[dict[str, Any]], random_503_rate: float = 0.0, seed: int = 7):
        self.tickets = {t["id"]: t for t in tickets}
        self.handled: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str]] = []
        self.accepted_posts: list[tuple[str, str]] = []
        self.conflicts = 0
        self.faults: list[Fault] = []
        self._rate = random_503_rate
        self._rng = random.Random(seed)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/functions/v1/helpdesk")
        self.calls.append((request.method, path))
        if request.headers.get("Authorization") != "Bearer test-key":
            return _error(401, "unauthorized")
        if not request.headers.get("X-Candidate-Email"):
            return _error(400, "missing_email")
        for fault in list(self.faults):
            response = fault(request)
            if response is not None:
                return response
        if self._rate and path.startswith("/tickets") and self._rng.random() < self._rate:
            return _error(503, "service_unavailable")
        return self._route(request, path)

    def _route(self, request: httpx.Request, path: str) -> httpx.Response:
        parts = [p for p in path.split("/") if p]
        if request.method == "GET" and parts == ["status"]:
            return httpx.Response(
                200, json={"open": self.open_count(), "handled": len(self.handled)}
            )
        if request.method == "GET" and parts == ["tickets"]:
            return self._list(request)
        if request.method == "GET" and len(parts) == 2 and parts[0] == "tickets":
            return self._get(parts[1])
        if request.method == "POST" and len(parts) == 3 and parts[0] == "tickets":
            return self.apply_post(parts[1], parts[2], json.loads(request.content or b"{}"))
        if request.method == "POST" and parts == ["practice", "reset"]:
            self.handled.clear()
            return httpx.Response(200, json={"reopened": len(self.tickets)})
        return _error(404, "not_found")

    def open_count(self) -> int:
        return len(self.tickets) - len(self.handled)

    def apply_post(self, ticket_id: str, action: str, body: dict[str, Any]) -> httpx.Response:
        if ticket_id not in self.tickets or action not in ("reply", "handoff", "close"):
            return _error(404, "not_found")
        if ticket_id in self.handled:
            self.conflicts += 1
            return _error(409, "already_handled")
        missing = sorted(_REQUIRED - body.keys())
        if action in _ACTION_FIELD and not body.get(_ACTION_FIELD[action]):
            missing.append(_ACTION_FIELD[action])
        if missing:
            return _error(422, "validation_failed", problems=[f"{m} is required" for m in missing])
        self.handled[ticket_id] = {"action": action, **body}
        self.accepted_posts.append((ticket_id, action))
        return httpx.Response(200, json={"ok": True})

    def _list(self, request: httpx.Request) -> httpx.Response:
        limit = min(int(request.url.params.get("limit", "10")), 10)
        cursor = request.url.params.get("cursor")
        only_open = request.url.params.get("status", "open") == "open"
        ids = sorted(i for i in self.tickets if not (only_open and i in self.handled))
        if cursor:
            ids = [i for i in ids if i > cursor]
        page = ids[:limit]
        next_cursor = page[-1] if len(ids) > limit else None
        return httpx.Response(
            200, json={"tickets": [self.tickets[i] for i in page], "next_cursor": next_cursor}
        )

    def _get(self, ticket_id: str) -> httpx.Response:
        if ticket_id not in self.tickets:
            return _error(404, "not_found")
        handling = self.handled.get(ticket_id)
        status = "handled" if handling else "open"
        return httpx.Response(
            200, json={**self.tickets[ticket_id], "status": status, "handling": handling}
        )


def _error(status: int, code: str, **extra: Any) -> httpx.Response:
    headers = {"Retry-After": "1"} if status == 429 else {}
    return httpx.Response(status, json={"error": code, **extra}, headers=headers)


def make_assessment(**overrides: Any) -> Assessment:
    values: dict[str, Any] = {
        "language": Language.NL,
        "english_summary": "How do I add a new employee to the rota?",
        "sensitive_topics": (),
        "faq_ids": ("F02",),
        "fully_answered_by_faq": True,
        "is_spam_or_empty": False,
        "category": Category.QUESTION,
        "priority": Priority.P4,
        "priority_reason": "How-to question.",
        "proposed_action": Action.REPLY,
        "reason": "How-to question answered by F02.",
        "draft_reply": (
            "Hoi Anouk, ga naar Team → Medewerker toevoegen, vul naam, e-mailadres en "
            "contracturen in en klik op Uitnodigen. Met vriendelijke groet, Team Roosterly"
        ),
        "handoff_note": "",
        "confidence": 0.93,
    }
    values.update(overrides)
    return Assessment.model_validate(values)


class FakeAssessor:
    def __init__(self, decide: Callable[[Ticket], Assessment] | None = None, cost: float = 0.001):
        self._decide = decide or (lambda _ticket: make_assessment())
        self._cost = cost
        self.seen: list[Ticket] = []

    def assess(self, ticket: Ticket) -> ModelResult[Assessment]:
        self.seen.append(ticket)
        return ModelResult(self._decide(ticket), Usage(cost_usd=self._cost, calls=1))


class FakeVerifier:
    """Reports `problems` on every call, or only on the first `failing_calls` calls."""

    def __init__(
        self,
        problems: tuple[str, ...] = (),
        cost: float = 0.0005,
        failing_calls: int | None = None,
    ):
        self._problems = problems
        self._cost = cost
        self._failing_calls = failing_calls
        self.calls = 0
        self.drafts: list[str] = []

    def verify(
        self,
        ticket: Ticket,
        draft_reply: str,
        expected_language: Language,
        articles: list[FaqArticle],
    ) -> ModelResult[ReplyCheck]:
        self.calls += 1
        self.drafts.append(draft_reply)
        failing = self._failing_calls is None or self.calls <= self._failing_calls
        problems = self._problems if failing else ()
        check = ReplyCheck(
            reply_language=expected_language.value,
            back_translation="Hi Anouk, go to Team → Add employee ...",
            answers_question=not problems,
            problems=problems,
        )
        return ModelResult(check, Usage(cost_usd=self._cost, calls=1))


class FakeReviser:
    """Returns `new_draft`, or the old draft unchanged when none is given."""

    def __init__(self, new_draft: str | None = None, cost: float = 0.0005):
        self._new_draft = new_draft
        self._cost = cost
        self.problems_seen: list[list[str]] = []

    def revise(
        self,
        ticket: Ticket,
        draft_reply: str,
        language: Language,
        articles: list[FaqArticle],
        problems: list[str],
    ) -> ModelResult[str]:
        self.problems_seen.append(problems)
        draft = self._new_draft if self._new_draft is not None else draft_reply
        return ModelResult(draft, Usage(cost_usd=self._cost, calls=1))


class ListRunLog:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def write(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)
