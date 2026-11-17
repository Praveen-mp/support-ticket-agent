import json

from tests.fakes import make_assessment
from ticket_agent.domain.models import (
    Action,
    Category,
    Decision,
    DecisionOutcome,
    Language,
    Priority,
)
from ticket_agent.evaluation.scoring import EvalReport, ExampleLabel, ExampleResult
from ticket_agent.infrastructure.storage import JsonDecisionStore, JsonlRunLog


def outcome(action: Action = Action.REPLY, faq_ids=("F02",)) -> DecisionOutcome:
    decision = Decision(
        ticket_id="E01",
        action=action,
        language=Language.NL,
        category=Category.QUESTION,
        priority=Priority.P4,
        reason="r",
        draft_reply="Hoi" if action is Action.REPLY else None,
        handoff_note="Note" if action is Action.HANDOFF else None,
    )
    return DecisionOutcome(decision=decision, assessment=make_assessment(faq_ids=faq_ids))


def test_store_round_trips_and_checks_fingerprint(tmp_path):
    path = tmp_path / "state" / "decisions.json"
    store = JsonDecisionStore(path)
    store.save("T1", "fp-1", outcome())
    store.mark_submitted("T1")

    reopened = JsonDecisionStore(path)
    assert reopened.load("T1", "fp-1") == outcome()
    assert reopened.load("T1", "fp-2") is None
    assert json.loads(path.read_text())["T1"]["submitted"] is True


def test_run_log_masks_payment_details(tmp_path):
    log = JsonlRunLog(tmp_path / "logs" / "run.jsonl")
    log.write({"ticket_id": "T1", "handoff_note": "Customer IBAN NL91ABNA0417164300"})
    line = (tmp_path / "logs" / "run.jsonl").read_text().strip()
    assert "NL91ABNA" not in line
    assert json.loads(line)["handoff_note"] == "Customer IBAN [IBAN]"


def label(action: Action, reason: str = "") -> ExampleLabel:
    return ExampleLabel(
        language=Language.NL,
        category=Category.QUESTION,
        priority=Priority.P4,
        action=action,
        reason=reason,
    )


def test_scoring_flags_unsafe_reply_and_missed_reply():
    unsafe = ExampleResult("E01", label(Action.HANDOFF, "Refund."), outcome(Action.REPLY))
    missed = ExampleResult("E02", label(Action.REPLY, "FAQ F02."), outcome(Action.HANDOFF))
    assert unsafe.unsafe_reply and not unsafe.missed_reply
    assert missed.missed_reply and not missed.unsafe_reply


def test_summary_counts_faq_matches_and_errors():
    report = EvalReport(
        [
            ExampleResult("E01", label(Action.REPLY, "How-to question, FAQ F02."), outcome()),
            ExampleResult("E02", label(Action.REPLY, "F05 answers it."), outcome(faq_ids=("F06",))),
            ExampleResult("E03", label(Action.HANDOFF), None, error="boom"),
        ]
    )
    summary = report.summary()
    assert summary["faq_article_match"] == "1/2"
    assert summary["errors"] == 1
    assert summary["accuracy"]["action"] == "2/3"
    assert summary["unsafe_replies"] == 0
