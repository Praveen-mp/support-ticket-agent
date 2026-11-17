import pytest

from ticket_agent.domain.language import detect_language, looks_english
from ticket_agent.domain.models import Language


@pytest.mark.parametrize(
    "text",
    [
        "Customer wants a refund for August because they were closed.",
        "Bug in clock-in.",
        "Customer asks where to find Instellingen → Abonnement → Facturen.",
        "Refund request; client rule sends it to a human.",
    ],
)
def test_english_notes_pass(text):
    assert looks_english(text)


@pytest.mark.parametrize(
    "text",
    [
        "De klant wil geld terug voor augustus omdat ze dicht waren.",
        "Klant heeft een vraag over de factuur van vorige maand.",
        "Der Kunde möchte wissen, wie er die Rechnung bekommt.",
    ],
)
def test_dutch_and_german_notes_fail(text):
    assert not looks_english(text)


@pytest.mark.parametrize(
    ("text", "language"),
    [
        (
            "App crashes\nThe app crashes immediately on my iPhone 8 (iOS 15). "
            "Other staff are fine.",
            Language.EN,
        ),
        ("Google Agenda\nGraag een koppeling met Google Agenda voor de diensten.", Language.NL),
        ("Rechnung\nIch möchte bitte wissen, wo ich die Rechnung finde.", Language.DE),
        ("Hoi, ik ben mijn wachtwoord vergeten. Hoe kom ik er weer in?", Language.NL),
    ],
)
def test_detects_clear_language(text, language):
    assert detect_language(text) is language


@pytest.mark.parametrize("text", ["Hallo", "Onderhoud", "iOS 15 crash", ""])
def test_thin_evidence_gives_no_answer(text):
    assert detect_language(text) is None
