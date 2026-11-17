import random

import httpx
import pytest

from tests.fakes import BASE_URL, FakeHelpdeskServer, ticket_dict
from ticket_agent.application.errors import (
    AlreadyHandledError,
    HelpdeskAuthError,
    HelpdeskError,
    HelpdeskUnavailableError,
    OutcomeUnknownError,
    RequestRejectedError,
)
from ticket_agent.domain.models import Action, Category, Decision, Language, Priority
from ticket_agent.infrastructure.helpdesk_client import HelpdeskClient, parse_handled
from ticket_agent.infrastructure.resilience import RateLimiter, RetryPolicy


class RecordingLimiter(RateLimiter):
    def __init__(self) -> None:
        super().__init__(1000, sleep=lambda _s: None)
        self.pauses: list[float] = []

    def pause(self, seconds: float) -> None:
        self.pauses.append(seconds)


def make_client(server: FakeHelpdeskServer, key: str = "test-key", limiter=None) -> HelpdeskClient:
    return HelpdeskClient(
        base_url=BASE_URL,
        api_key=key,
        candidate_email="me@example.com",
        rate_limiter=limiter or RecordingLimiter(),
        retry_policy=RetryPolicy(max_attempts=4),
        transport=server.transport(),
        sleep=lambda _s: None,
        rng=random.Random(0),
    )


def reply_decision(ticket_id: str = "T001") -> Decision:
    return Decision(
        ticket_id=ticket_id,
        action=Action.REPLY,
        language=Language.NL,
        category=Category.QUESTION,
        priority=Priority.P4,
        reason="Answered by F02.",
        draft_reply="Hoi, ga naar Team → Medewerker toevoegen.",
        confidence=0.9,
    )


def fail_first(n: int, response_or_exc, method: str = "", path_suffix: str = ""):
    """Fault that triggers on the first n matching requests."""
    remaining = {"n": n}

    def fault(request: httpx.Request):
        if method and request.method != method:
            return None
        if path_suffix and not request.url.path.endswith(path_suffix):
            return None
        if remaining["n"] <= 0:
            return None
        remaining["n"] -= 1
        if isinstance(response_or_exc, Exception):
            raise response_or_exc
        return response_or_exc

    return fault


def test_lists_all_pages_of_open_tickets():
    server = FakeHelpdeskServer([ticket_dict(f"T{i:03d}") for i in range(1, 26)])
    tickets = make_client(server).list_open_tickets(max_tickets=100)
    assert [t.id for t in tickets] == [f"T{i:03d}" for i in range(1, 26)]
    assert sum(1 for m, p in server.calls if p == "/tickets") == 3


def test_respects_max_tickets():
    server = FakeHelpdeskServer([ticket_dict(f"T{i:03d}") for i in range(1, 26)])
    assert len(make_client(server).list_open_tickets(max_tickets=12)) == 12


def test_retries_503_and_counts_retries():
    server = FakeHelpdeskServer([ticket_dict("T001")])
    server.faults.append(fail_first(2, httpx.Response(503, json={"error": "service_unavailable"})))
    client = make_client(server)
    assert client.get_ticket_state("T001").handled is False
    assert client.retry_count == 2


def test_429_waits_for_retry_after():
    server = FakeHelpdeskServer([ticket_dict("T001")])
    server.faults.append(
        fail_first(
            1, httpx.Response(429, json={"error": "rate_limited"}, headers={"Retry-After": "3"})
        )
    )
    limiter = RecordingLimiter()
    make_client(server, limiter=limiter).get_ticket_state("T001")
    assert limiter.pauses == [3.0]


def test_post_503_is_retried_because_nothing_was_saved():
    server = FakeHelpdeskServer([ticket_dict("T001")])
    server.faults.append(
        fail_first(1, httpx.Response(503, json={"error": "service_unavailable"}), method="POST")
    )
    make_client(server).submit_decision(reply_decision())
    assert server.accepted_posts == [("T001", "reply")]


def test_409_is_raised_and_never_retried():
    server = FakeHelpdeskServer([ticket_dict("T001")])
    client = make_client(server)
    client.submit_decision(reply_decision())
    with pytest.raises(AlreadyHandledError):
        client.submit_decision(reply_decision())
    assert sum(1 for m, _ in server.calls if m == "POST") == 2


def test_post_read_timeout_is_outcome_unknown_and_not_retried():
    server = FakeHelpdeskServer([ticket_dict("T001")])
    server.faults.append(fail_first(1, httpx.ReadTimeout("slow"), method="POST"))
    with pytest.raises(OutcomeUnknownError):
        make_client(server).submit_decision(reply_decision())
    assert sum(1 for m, _ in server.calls if m == "POST") == 1


def test_post_connect_error_is_retried_because_nothing_was_sent():
    server = FakeHelpdeskServer([ticket_dict("T001")])
    server.faults.append(fail_first(1, httpx.ConnectError("refused"), method="POST"))
    make_client(server).submit_decision(reply_decision())
    assert server.accepted_posts == [("T001", "reply")]


def test_post_502_is_outcome_unknown():
    server = FakeHelpdeskServer([ticket_dict("T001")])
    server.faults.append(fail_first(1, httpx.Response(502), method="POST"))
    with pytest.raises(OutcomeUnknownError):
        make_client(server).submit_decision(reply_decision())


def test_get_read_timeout_is_retried():
    server = FakeHelpdeskServer([ticket_dict("T001")])
    server.faults.append(fail_first(1, httpx.ReadTimeout("slow"), method="GET"))
    assert make_client(server).get_ticket_state("T001").handled is False


def test_gives_up_after_max_attempts():
    server = FakeHelpdeskServer([ticket_dict("T001")])
    server.faults.append(fail_first(99, httpx.Response(503, json={"error": "service_unavailable"})))
    with pytest.raises(HelpdeskUnavailableError):
        make_client(server).get_ticket_state("T001")
    assert len(server.calls) == 4


def test_422_carries_the_problems():
    server = FakeHelpdeskServer([ticket_dict("T001")])
    bad = reply_decision().model_copy(update={"draft_reply": ""})
    with pytest.raises(RequestRejectedError) as excinfo:
        make_client(server).submit_decision(bad)
    assert excinfo.value.problems == ["draft_reply is required"]


def test_wrong_key_is_auth_error():
    server = FakeHelpdeskServer([ticket_dict("T001")])
    with pytest.raises(HelpdeskAuthError):
        make_client(server, key="wrong").get_status()


def test_sends_identity_headers():
    seen = {}

    def capture(request: httpx.Request):
        seen.update(request.headers)

    server = FakeHelpdeskServer([])
    server.faults.append(capture)
    make_client(server).get_status()
    assert seen["authorization"] == "Bearer test-key"
    assert seen["x-candidate-email"] == "me@example.com"


@pytest.mark.parametrize(
    ("payload", "handled"),
    [
        ({"id": "T1", "status": "open"}, False),
        ({"id": "T1", "status": "handled"}, True),
        ({"id": "T1", "handled": None}, False),
        ({"id": "T1", "handling": {"action": "reply"}}, True),
        ({"ticket": {"id": "T1", "status": "open"}}, False),
        ({"id": "T1", "handled": True}, True),
    ],
)
def test_parse_handled_accepts_likely_shapes(payload, handled):
    assert parse_handled(payload) is handled


def test_parse_handled_refuses_to_guess():
    with pytest.raises(HelpdeskError):
        parse_handled({"id": "T1", "subject": "x"})
