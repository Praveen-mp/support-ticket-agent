"""End-to-end runner tests: real HelpdeskClient + real decider/policy against the fake server.

Only the model is faked. These are the tests behind the "no customer gets two answers" claim.
"""

import json
import random

import httpx
import pytest

from tests.fakes import (
    BASE_URL,
    FakeAssessor,
    FakeHelpdeskServer,
    FakeReviser,
    FakeVerifier,
    ListRunLog,
    load_kb,
    ticket_dict,
)
from ticket_agent.application.decider import TicketDecider
from ticket_agent.application.errors import ModelError
from ticket_agent.application.runner import RunLimits, TicketRunner
from ticket_agent.domain.policy import PolicyConfig, TicketPolicy
from ticket_agent.infrastructure.helpdesk_client import HelpdeskClient
from ticket_agent.infrastructure.resilience import RateLimiter, RetryPolicy
from ticket_agent.infrastructure.storage import JsonDecisionStore


class SimulatedCrash(BaseException):
    """Stands in for the process being killed. BaseException so nothing catches it."""


def make_runner(server, tmp_path, assessor=None, limits=None, run_id="r1"):
    kb = load_kb()
    assessor = assessor or FakeAssessor()
    decider = TicketDecider(
        assessor=assessor,
        verifier=FakeVerifier(),
        reviser=FakeReviser(),
        policy=TicketPolicy(kb, PolicyConfig()),
        knowledge_base=kb,
        model_id="test-model",
    )
    client = HelpdeskClient(
        base_url=BASE_URL,
        api_key="test-key",
        candidate_email="me@example.com",
        rate_limiter=RateLimiter(1000, sleep=lambda _s: None),
        retry_policy=RetryPolicy(),
        transport=server.transport(),
        sleep=lambda _s: None,
        rng=random.Random(3),
    )
    log = ListRunLog()
    runner = TicketRunner(
        helpdesk=client,
        decider=decider,
        store=JsonDecisionStore(tmp_path / "decisions.json"),
        run_log=log,
        limits=limits or RunLimits(),
        run_id=run_id,
    )
    return runner, log, assessor


def tickets(n: int):
    return [ticket_dict(f"T{i:03d}") for i in range(1, n + 1)]


def on_post(ticket_id: str, action):
    """Fault that runs `action(request)` on the first POST for ticket_id."""
    fired = {"done": False}

    def fault(request: httpx.Request):
        if fired["done"] or request.method != "POST":
            return None
        if f"/tickets/{ticket_id}/" not in request.url.path:
            return None
        fired["done"] = True
        return action(request)

    return fault


def test_handles_every_ticket_exactly_once_despite_random_503s(tmp_path):
    server = FakeHelpdeskServer(tickets(30), random_503_rate=0.2, seed=11)
    runner, log, _ = make_runner(server, tmp_path)
    summary = runner.run()
    assert summary.submitted == 30
    assert summary.failed == 0
    assert server.conflicts == 0
    assert sorted(t for t, _ in server.accepted_posts) == [f"T{i:03d}" for i in range(1, 31)]
    assert len(log.entries) == 30
    assert sum(e["http_retries"] for e in log.entries) > 0


def test_timeout_after_post_landed_is_reconciled_not_resent(tmp_path):
    server = FakeHelpdeskServer(tickets(3))

    def land_then_time_out(request):
        server.apply_post("T002", "reply", json.loads(request.content))
        raise httpx.ReadTimeout("response lost")

    server.faults.append(on_post("T002", land_then_time_out))
    runner, log, _ = make_runner(server, tmp_path)
    summary = runner.run()

    assert summary.submitted == 3
    assert server.conflicts == 0
    assert server.accepted_posts.count(("T002", "reply")) == 1
    statuses = {e["ticket_id"]: e["status"] for e in log.entries}
    assert statuses["T002"] == "submitted_reconciled"


def test_crash_after_post_then_rerun_never_answers_twice(tmp_path):
    server = FakeHelpdeskServer(tickets(6))

    def land_then_crash(request):
        server.apply_post("T003", "reply", json.loads(request.content))
        raise SimulatedCrash

    server.faults.append(on_post("T003", land_then_crash))
    runner, _, _ = make_runner(server, tmp_path, run_id="first")
    with pytest.raises(SimulatedCrash):
        runner.run()
    assert server.open_count() == 3

    rerun, _, _ = make_runner(server, tmp_path, run_id="second")
    summary = rerun.run()
    assert summary.submitted == 3
    assert server.open_count() == 0
    assert server.conflicts == 0
    assert len(server.accepted_posts) == 6


def test_crash_before_post_reuses_the_saved_decision(tmp_path):
    server = FakeHelpdeskServer(tickets(4))

    def crash(_request):
        raise SimulatedCrash

    server.faults.append(on_post("T002", crash))
    runner, _, first_assessor = make_runner(server, tmp_path, run_id="first")
    with pytest.raises(SimulatedCrash):
        runner.run()
    assert [t.id for t in first_assessor.seen] == ["T001", "T002"]

    rerun, log, second_assessor = make_runner(server, tmp_path, run_id="second")
    summary = rerun.run()
    # T002 was decided before the crash: reused, not paid for again.
    assert [t.id for t in second_assessor.seen] == ["T003", "T004"]
    assert summary.reused_decisions == 1
    assert server.open_count() == 0
    assert next(e for e in log.entries if e["ticket_id"] == "T002")["decision_reused"] is True


def test_dry_run_posts_nothing(tmp_path):
    server = FakeHelpdeskServer(tickets(3))
    runner, log, _ = make_runner(server, tmp_path)
    summary = runner.run(dry_run=True)
    assert server.accepted_posts == []
    assert summary.submitted == 0
    assert {e["status"] for e in log.entries} == {"dry_run"}


def test_stops_at_cost_budget(tmp_path):
    server = FakeHelpdeskServer(tickets(5))
    runner, _, _ = make_runner(
        server, tmp_path, assessor=FakeAssessor(cost=0.3), limits=RunLimits(max_cost_usd=0.5)
    )
    summary = runner.run()
    assert summary.submitted == 2
    assert "cost budget" in summary.stopped_early
    assert server.open_count() == 3


def test_model_failures_leave_tickets_open_and_stop_after_three(tmp_path):
    server = FakeHelpdeskServer(tickets(5))

    def broken(_ticket):
        raise ModelError("provider down")

    runner, log, _ = make_runner(server, tmp_path, assessor=FakeAssessor(broken))
    summary = runner.run()
    assert summary.failed == 3
    assert "in a row" in summary.stopped_early
    assert server.accepted_posts == []
    assert all(e["status"] == "failed" for e in log.entries)


def test_ticket_handled_elsewhere_is_skipped(tmp_path):
    server = FakeHelpdeskServer(tickets(2))
    runner, _log, assessor = make_runner(server, tmp_path)
    # Another process answers T001 between our list call and our per-ticket check.
    original = server._get

    def get_after_race(ticket_id):
        if ticket_id == "T001" and "T001" not in server.handled:
            server.apply_post(
                "T001",
                "close",
                {"language": "nl", "category": "other", "priority": "P4", "reason": "spam"},
            )
        return original(ticket_id)

    server._get = get_after_race
    summary = runner.run()
    assert summary.skipped_already_handled == 1
    assert [t.id for t in assessor.seen] == ["T002"]
    assert server.conflicts == 0
