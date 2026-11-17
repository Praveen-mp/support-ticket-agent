from tests.fakes import FakeAssessor, FakeReviser, FakeVerifier, load_kb, make_assessment
from ticket_agent.application.decider import TicketDecider
from ticket_agent.domain.models import Action, Ticket
from ticket_agent.domain.policy import PolicyConfig, TicketPolicy

ENGLISH_F12_REPLY = (
    "Hi, the Roosterly app works on iOS 16 and later, so it does not run on iOS 15. "
    "Please update the phone to iOS 16 or later, or use another phone. "
    "Kind regards, the Roosterly team"
)
CRASH_TICKET = Ticket(
    id="T1",
    subject="App crashes",
    body="The app crashes immediately on my iPhone 8 (iOS 15). Other staff are fine.",
)


def make_decider(
    assessor=None, verifier=None, reviser=None
) -> tuple[TicketDecider, FakeAssessor, FakeVerifier]:
    kb = load_kb()
    assessor = assessor or FakeAssessor()
    verifier = verifier or FakeVerifier()
    decider = TicketDecider(
        assessor=assessor,
        verifier=verifier,
        reviser=reviser or FakeReviser(),
        policy=TicketPolicy(kb, PolicyConfig()),
        knowledge_base=kb,
        model_id="test-model",
    )
    return decider, assessor, verifier


def test_reply_path_runs_both_checks_and_sums_cost():
    decider, _assessor, verifier = make_decider()
    outcome = decider.decide(Ticket(id="T1", subject="Nieuwe medewerker", body="Hoe?"))
    assert outcome.decision.action is Action.REPLY
    assert verifier.calls == 1
    assert outcome.usage.calls == 2
    assert round(outcome.usage.cost_usd, 6) == 0.0015


def test_blank_ticket_closes_without_any_model_call():
    decider, assessor, _verifier = make_decider()
    outcome = decider.decide(Ticket(id="T1", subject="  ", body="\n"))
    assert outcome.decision.action is Action.CLOSE
    assert assessor.seen == []
    assert outcome.usage.calls == 0


def test_model_only_sees_masked_payment_details():
    decider, assessor, _ = make_decider()
    decider.decide(Ticket(id="T1", subject="Factuur", body="Mijn IBAN is NL91ABNA0417164300"))
    assert "NL91ABNA" not in assessor.seen[0].body
    assert "[IBAN]" in assessor.seen[0].body


def test_verifier_is_skipped_when_policy_already_blocks():
    decider, _, verifier = make_decider()
    outcome = decider.decide(Ticket(id="T1", subject="Refund", body="Ik wil een terugbetaling."))
    assert outcome.decision.action is Action.HANDOFF
    assert verifier.calls == 0


def test_failed_reply_check_turns_reply_into_handoff():
    decider, _, _ = make_decider(verifier=FakeVerifier(problems=("claim not in FAQ: refund",)))
    outcome = decider.decide(Ticket(id="T1", subject="Vraag", body="Hoe voeg ik iemand toe?"))
    assert outcome.decision.action is Action.HANDOFF
    assert any("reply check" in note for note in outcome.policy_notes)
    assert "Why a human" in (outcome.decision.handoff_note or "")


def test_handoff_proposal_skips_verifier():
    assessor = FakeAssessor(
        lambda _t: make_assessment(
            proposed_action="handoff",
            draft_reply="",
            handoff_note="Bug in clock-in.",
            category="bug",
            fully_answered_by_faq=False,
            faq_ids=(),
        )
    )
    decider, _, verifier = make_decider(assessor=assessor)
    outcome = decider.decide(Ticket(id="T1", subject="Bug", body="App crasht bij inklokken"))
    assert outcome.decision.action is Action.HANDOFF
    assert verifier.calls == 0


def test_fingerprint_changes_with_ticket_text():
    decider, _, _ = make_decider()
    a = decider.fingerprint(Ticket(id="T1", subject="a", body="b"))
    b = decider.fingerprint(Ticket(id="T1", subject="a", body="c"))
    assert a != b
    assert a == decider.fingerprint(Ticket(id="T1", subject="a", body="b"))


def test_mislabelled_english_ticket_is_corrected_and_dutch_draft_blocked():
    # The reviser hands the Dutch draft back unchanged, so it stays blocked.
    decider, _, verifier = make_decider(
        assessor=FakeAssessor(lambda _t: make_assessment(language="nl", faq_ids=("F12",)))
    )
    outcome = decider.decide(CRASH_TICKET)
    assert outcome.decision.language.value == "en"
    assert outcome.decision.action is Action.HANDOFF
    assert outcome.policy_notes[0] == "language corrected from nl to en by text check"
    assert verifier.calls == 0


def test_draft_in_wrong_language_is_rewritten_and_then_sent():
    reviser = FakeReviser(new_draft=ENGLISH_F12_REPLY)
    decider, _, verifier = make_decider(
        assessor=FakeAssessor(lambda _t: make_assessment(language="nl", faq_ids=("F12",))),
        reviser=reviser,
    )
    outcome = decider.decide(CRASH_TICKET)
    assert outcome.decision.action is Action.REPLY
    assert outcome.decision.draft_reply == ENGLISH_F12_REPLY
    assert reviser.problems_seen == [["draft is in nl, customer wrote in en"]]
    assert verifier.calls == 1
    assert any(n.startswith("draft rewritten once") for n in outcome.policy_notes)
    assert outcome.usage.calls == 3


def test_failed_reply_check_is_rewritten_once_and_verified_again():
    verifier = FakeVerifier(
        problems=("claim not in FAQ: link arrives in 10 minutes",), failing_calls=1
    )
    reviser = FakeReviser(
        new_draft="Hoi, ga naar Team → Medewerker toevoegen. Groet, Team Roosterly"
    )
    decider, _, _ = make_decider(verifier=verifier, reviser=reviser)
    outcome = decider.decide(Ticket(id="T1", subject="Vraag", body="Hoe voeg ik iemand toe?"))
    assert outcome.decision.action is Action.REPLY
    assert verifier.drafts[1] == outcome.decision.draft_reply
    assert reviser.problems_seen == [["reply check: claim not in FAQ: link arrives in 10 minutes"]]
    assert outcome.reply_check is not None and outcome.reply_check.passed


def test_rewrite_happens_at_most_once():
    reviser = FakeReviser()
    decider, _, verifier = make_decider(
        verifier=FakeVerifier(problems=("claim not in FAQ: x",)), reviser=reviser
    )
    outcome = decider.decide(Ticket(id="T1", subject="Vraag", body="Hoe voeg ik iemand toe?"))
    assert outcome.decision.action is Action.HANDOFF
    assert len(reviser.problems_seen) == 1
    assert verifier.calls == 2


def test_ineligible_ticket_is_never_rewritten():
    reviser = FakeReviser(new_draft=ENGLISH_F12_REPLY)
    decider, _, verifier = make_decider(reviser=reviser)
    outcome = decider.decide(Ticket(id="T1", subject="Refund", body="Ik wil een terugbetaling."))
    assert outcome.decision.action is Action.HANDOFF
    assert reviser.problems_seen == []
    assert verifier.calls == 0


def test_p1_ticket_never_gets_an_automatic_reply():
    decider, _, verifier = make_decider(
        assessor=FakeAssessor(lambda _t: make_assessment(priority="P1"))
    )
    outcome = decider.decide(Ticket(id="T1", subject="Vraag", body="Hoe voeg ik iemand toe?"))
    assert outcome.decision.action is Action.HANDOFF
    assert verifier.calls == 0
    assert any("P1 tickets always go to a human" in n for n in outcome.policy_notes)
