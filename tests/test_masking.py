from ticket_agent.domain.masking import (
    contains_payment_details,
    mask_payment_details,
    mask_recursively,
)


def test_masks_unspaced_iban():
    assert mask_payment_details("Afgeschreven van NL91ABNA0417164300.") == (
        "Afgeschreven van [IBAN]."
    )


def test_masks_spaced_iban_without_eating_following_words():
    text = "Rekening NL91 ABNA 0417 1643 00 graag controleren"
    assert mask_payment_details(text) == "Rekening [IBAN] graag controleren"


def test_masks_lowercase_and_german_iban():
    assert mask_payment_details("nl91abna0417164300") == "[IBAN]"
    assert mask_payment_details("DE89 3704 0044 0532 0130 00") == "[IBAN]"


def test_masks_card_numbers_with_spaces_or_dashes():
    assert mask_payment_details("kaart 4111 1111 1111 1111 ok") == "kaart [CARD] ok"
    assert mask_payment_details("4111-1111-1111-1111") == "[CARD]"


def test_leaves_ordinary_numbers_alone():
    text = "Bel 06 12345678, ticket T005, 12 medewerkers, factuur 2026-10"
    assert mask_payment_details(text) == text
    assert not contains_payment_details(text)


def test_masks_nested_structures():
    data = {"note": "IBAN NL91ABNA0417164300", "items": ["4111111111111111", 3], "n": None}
    assert mask_recursively(data) == {"note": "IBAN [IBAN]", "items": ["[CARD]", 3], "n": None}
