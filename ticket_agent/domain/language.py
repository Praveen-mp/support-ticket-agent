"""Cheap, deterministic language checks for English, Dutch and German.

They count common function words; this is not a general language detector. It exists so
that a language decision never rests on the model alone: the eval showed the model calling
an English ticket Dutch, which would send an English customer a Dutch reply.

Words shared between the languages ("is", "was", "we", "als", "die", "in") are left out,
so every hit is evidence for one language only. Dutch menu names quoted in an English text
(e.g. Instellingen → Abonnement) do not count, because they are not function words.
"""

from __future__ import annotations

import re

from ticket_agent.domain.models import Language

_WORD = re.compile(r"[a-zäöüßéëï]+")

# Space-separated word lists read better here than one word per line.
_ENGLISH = frozenset(
    "the and are to for with that this it they their them he she his her my i you your "  # noqa: SIM905
    "our us can how what why do does a be but if not would about from please customer "
    "wants asks needs has have get got any there when where which other fine".split()
)
_DUTCH = frozenset(
    "de het een ik je jij jullie u uw wij onze ons niet van voor met dat dit deze zijn "  # noqa: SIM905
    "heeft hebben wil willen graag klant maar ook kan kunnen kunt bij naar wordt hoe "
    "geen mijn op aan te om nog al wat waarom moet wel hoi groet bedankt dank alvast".split()
)
_GERMAN = frozenset(
    "der das und nicht ich ist für kunde möchte bitte auch nach sich ein eine unser kein "  # noqa: SIM905
    "mein sie ihr es zu den dem von auf haben können warum danke guten hallo leider wir "
    "uns".split()
)

_MIN_HITS = 2
_DOMINANCE = 2  # the winner needs at least twice the hits of the runner-up


def _counts(text: str) -> dict[Language, int]:
    words = _WORD.findall(text.lower())
    return {
        Language.EN: sum(word in _ENGLISH for word in words),
        Language.NL: sum(word in _DUTCH for word in words),
        Language.DE: sum(word in _GERMAN for word in words),
    }


def detect_language(text: str) -> Language | None:
    """The language the text is clearly written in, or None when the evidence is thin."""
    ranked = sorted(_counts(text).items(), key=lambda item: item[1], reverse=True)
    (best, best_hits), (_, second_hits) = ranked[0], ranked[1]
    if best_hits >= _MIN_HITS and best_hits >= _DOMINANCE * second_hits:
        return best
    return None


def looks_english(text: str) -> bool:
    """False only when Dutch or German function words outnumber English ones.

    A short note with no function words at all (e.g. "Bug in clock-in.") passes: a real
    Dutch or German sentence almost always contains several of them.
    """
    counts = _counts(text)
    return counts[Language.EN] >= max(counts[Language.NL], counts[Language.DE])
