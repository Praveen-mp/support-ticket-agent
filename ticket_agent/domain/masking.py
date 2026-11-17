"""Masks payment details (IBANs, card numbers).

Client rule: never repeat payment details in a reply or a log. We mask ticket text before it
reaches the model, and mask every log line again before it is written.
"""

from __future__ import annotations

import re
from typing import Any

IBAN_MASK = "[IBAN]"
CARD_MASK = "[CARD]"

# Country code, 2 check digits, then 11-30 alphanumerics in groups of 4, spaces optional.
_IBAN = re.compile(
    r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b",
    re.IGNORECASE,
)
# 13 to 19 digits, optionally grouped with spaces or dashes. No Luhn check on purpose:
# masking a long number that is not a card costs nothing, missing a card breaks a rule.
_CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")


def mask_payment_details(text: str) -> str:
    masked = _IBAN.sub(IBAN_MASK, text)
    return _CARD.sub(CARD_MASK, masked)


def contains_payment_details(text: str) -> bool:
    return bool(_IBAN.search(text) or _CARD.search(text))


def mask_recursively(value: Any) -> Any:
    """Masks every string inside nested dicts, lists and tuples."""
    if isinstance(value, str):
        return mask_payment_details(value)
    if isinstance(value, dict):
        return {key: mask_recursively(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [mask_recursively(item) for item in value]
    return value
