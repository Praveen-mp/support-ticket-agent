import pytest

from tests.fakes import load_kb, make_assessment
from ticket_agent.domain.models import Action, Language, Priority, SensitiveTopic, Ticket
from ticket_agent.domain.policy import PolicyConfig, TicketPolicy, detect_sensitive_topics

TICKET = Ticket(id="T1", subject="Nieuwe medewerker", body="Hoe voeg ik iemand toe?")


@pytest.fixture
def policy() -> TicketPolicy:
    return TicketPolicy(load_kb(), PolicyConfig(confidence_threshold=0.75))


def _decide(policy, assessment, ticket=TICKET):
    keyword_topics = detect_sensitive_topics(f"{ticket.subject}\n{ticket.body}")
    blockers = policy.reply_blockers(ticket, assessment, keyword_topics)
    return policy.decide(ticket, assessment, keyword_topics, blockers), blockers


def test_grounded_reply_is_sent(policy):
    (decision, notes), blockers = _decide(policy, make_assessment())
    assert blockers == []
    assert decision.action is Action.REPLY
    assert decision.draft_reply and "Medewerker toevoegen" in decision.draft_reply
    assert decision.handoff_note is None
    assert notes == []


def test_refund_keyword_forces_handoff_even_if_model_missed_it(policy):
    ticket = Ticket(id="T2", subject="Terugbetaling", body="Ik wil mijn geld terug voor augustus.")
    assessment = make_assessment(category="billing", priority="P4")
    (decision, notes), _ = _decide(policy, assessment, ticket)
    assert decision.action is Action.HANDOFF
    assert decision.priority is Priority.P2  # money involved
    assert decision.draft_reply is None
    assert "Why a human" in (decision.handoff_note or "")
    assert any("refund_or_disputed_charge" in n for n in notes)


def test_security_topic_from_model_is_p1_handoff(policy):
    assessment = make_assessment(
        sensitive_topics=("security_incident",),
        proposed_action="handoff",
        priority="P3",
        draft_reply="",
        handoff_note="Customer sees logins they do not recognise.",
    )
    (decision, _), _ = _decide(policy, assessment)
    assert decision.action is Action.HANDOFF
    assert decision.priority is Priority.P1


def test_low_confidence_blocks_reply(policy):
    (decision, _notes), blockers = _decide(policy, make_assessment(confidence=0.5))
    assert decision.action is Action.HANDOFF
    assert any("confidence 0.50" in b for b in blockers)
    assert "Sent to a human by policy" in decision.reason


def test_partly_answered_question_is_not_replied(policy):
    (decision, _), blockers = _decide(policy, make_assessment(fully_answered_by_faq=False))
    assert decision.action is Action.HANDOFF
    assert "the FAQ does not fully answer the question" in blockers


@pytest.mark.parametrize(
    ("draft", "expected_problem"),
    [
        ("Ga naar https://roosterly-help.com voor hulp.", "URL not in the FAQ"),
        ("Het Plus-abonnement kost €5 per medewerker.", "numbers not in the cited FAQ: 5"),
        ("Uw IBAN NL91ABNA0417164300 is genoteerd.", "payment details"),
    ],
)
def test_ungrounded_drafts_are_blocked(policy, draft, expected_problem):
    assessment = make_assessment(faq_ids=("F05",), draft_reply=draft)
    (decision, _), blockers = _decide(policy, assessment)
    assert decision.action is Action.HANDOFF
    assert any(expected_problem in b for b in blockers), blockers


def test_numbers_from_cited_article_are_allowed(policy):
    draft = "Basis kost €4 en Plus €6 per actieve medewerker per maand, exclusief btw."
    assessment = make_assessment(faq_ids=("F05",), draft_reply=draft)
    (decision, _), blockers = _decide(policy, assessment)
    assert blockers == []
    assert decision.action is Action.REPLY


def test_unknown_faq_id_blocks_reply(policy):
    _, blockers = _decide(policy, make_assessment(faq_ids=("F99",)))
    assert any("unknown FAQ ids" in b for b in blockers)


def test_confident_spam_is_closed(policy):
    assessment = make_assessment(
        is_spam_or_empty=True,
        proposed_action="close",
        category="other",
        draft_reply="",
        faq_ids=(),
        confidence=0.95,
    )
    (decision, _), _ = _decide(policy, assessment)
    assert decision.action is Action.CLOSE


def test_unsure_close_becomes_handoff(policy):
    assessment = make_assessment(
        is_spam_or_empty=True,
        proposed_action="close",
        draft_reply="",
        faq_ids=(),
        confidence=0.6,
    )
    (decision, _notes), _ = _decide(policy, assessment)
    assert decision.action is Action.HANDOFF
    assert decision.handoff_note


def test_policy_never_lowers_priority(policy):
    assessment = make_assessment(
        priority="P1",
        proposed_action="handoff",
        draft_reply="",
        sensitive_topics=("refund_or_disputed_charge",),
        handoff_note="Refund.",
    )
    (decision, _), _ = _decide(policy, assessment)
    assert decision.priority is Priority.P1


def test_blank_ticket_closes_without_model(policy):
    decision = policy.close_blank_ticket(Ticket(id="T9"))
    assert decision.action is Action.CLOSE


@pytest.mark.parametrize(
    ("text", "topic"),
    [
        (
            "Ik wil mijn abonnement opzeggen per 1 januari",
            SensitiveTopic.CANCELLATION_OR_CONTRACT_CHANGE,
        ),
        ("We zijn dubbel afgeschreven deze maand", SensitiveTopic.REFUND_OR_DISPUTED_CHARGE),
        ("Ik heb mijn advocaat ingeschakeld", SensitiveTopic.LEGAL_THREAT),
        ("Iemand heeft ons account gehackt!", SensitiveTopic.SECURITY_INCIDENT),
        ("Verwijder al mijn gegevens alstublieft", SensitiveTopic.PERSONAL_DATA_REQUEST),
        ("Dan stappen we over naar een concurrent", SensitiveTopic.CHURN_THREAT),
        ("Ich möchte kündigen", SensitiveTopic.CANCELLATION_OR_CONTRACT_CHANGE),
    ],
)
def test_keyword_backstop_finds_always_human_topics(text, topic):
    assert topic in detect_sensitive_topics(text)


@pytest.mark.parametrize(
    "text",
    [
        "Waar worden onze gegevens opgeslagen? Voldoen jullie aan de AVG?",
        "Er is een storing in de app",
        "Hoe kan ik upgraden naar Plus?",
        "Hoe voeg ik een nieuwe medewerker toe?",
    ],
)
def test_keyword_backstop_does_not_fire_on_ordinary_questions(text):
    assert detect_sensitive_topics(text) == frozenset()


def test_posted_reason_includes_the_priority_reason(policy):
    (decision, _), _ = _decide(policy, make_assessment(priority_reason="How-to, no urgency."))
    assert decision.reason == ("How-to question answered by F02. Priority P4: How-to, no urgency.")


def test_raised_priority_explains_the_client_rule(policy):
    assessment = make_assessment(
        priority="P4",
        proposed_action="handoff",
        draft_reply="",
        sensitive_topics=("legal_threat",),
        handoff_note="Customer threatens legal steps over an invoice.",
    )
    (decision, _), _ = _decide(policy, assessment)
    assert decision.priority is Priority.P1
    assert decision.reason.endswith("Priority P1: security or legal risk (client rule).")


def test_long_reason_keeps_the_priority_line(policy):
    (decision, _), _ = _decide(policy, make_assessment(reason="x" * 600))
    assert len(decision.reason) <= 400
    assert decision.reason.endswith("Priority P4: How-to question.")


def test_dutch_handoff_note_falls_back_to_english_summary(policy):
    assessment = make_assessment(
        proposed_action="handoff",
        draft_reply="",
        fully_answered_by_faq=False,
        handoff_note="De klant wil weten waarom de app niet werkt bij het inklokken.",
        english_summary="The customer asks why clock-in does not work in the app.",
    )
    (decision, notes), _ = _decide(policy, assessment)
    assert decision.handoff_note == "The customer asks why clock-in does not work in the app."
    assert "model handoff note was not in English, not used" in notes


def test_no_english_text_gives_generic_english_note(policy):
    assessment = make_assessment(
        proposed_action="handoff",
        draft_reply="",
        handoff_note="Der Kunde möchte wissen, wie das geht.",
        english_summary="Der Kunde hat eine Frage zu der Rechnung.",
    )
    (decision, _), _ = _decide(policy, assessment)
    assert (decision.handoff_note or "").startswith("Customer ticket needs a human")


ENGLISH_TICKET = Ticket(
    id="T7",
    subject="App crashes",
    body="The app crashes immediately on my iPhone 8 (iOS 15). Other staff are fine.",
)


def test_wrong_language_label_is_corrected_from_the_text(policy):
    corrected, note = policy.correct_language(ENGLISH_TICKET, make_assessment(language="nl"))
    assert corrected.language is Language.EN
    assert note == "language corrected from nl to en by text check"


def test_ambiguous_text_keeps_the_model_label(policy):
    ticket = Ticket(id="T8", subject="Onderhoud", body="Onderhoud vanavond?")
    corrected, note = policy.correct_language(ticket, make_assessment(language="nl"))
    assert corrected.language is Language.NL
    assert note == ""


def test_dutch_draft_to_english_customer_is_blocked(policy):
    assessment = make_assessment(language="en", faq_ids=("F12",))  # draft is Dutch
    _, blockers = _decide(policy, assessment, ENGLISH_TICKET)
    assert "draft is in nl, customer wrote in en" in blockers
