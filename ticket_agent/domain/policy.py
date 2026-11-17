"""The client's support rules, enforced in code.

The model proposes; this module decides. Every rule here can only move a ticket towards a
human (handoff) or a higher priority, never the other way. A wrong answer reaches a real
customer, an unnecessary handoff costs a few minutes of a support agent's time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ticket_agent.domain.faq import FaqArticle, FaqKnowledgeBase, extract_urls
from ticket_agent.domain.language import detect_language, looks_english
from ticket_agent.domain.masking import contains_payment_details, mask_payment_details
from ticket_agent.domain.models import (
    MAX_REASON_CHARS,
    Action,
    Assessment,
    Category,
    Decision,
    Language,
    Priority,
    SensitiveTopic,
    Ticket,
)

# Backstop for the always-human topics, in Dutch, English and German. The model is the primary
# detector; these patterns only catch phrasings it might miss. They are kept narrow on purpose:
# a false positive here turns an answerable FAQ question into a handoff.
_SENSITIVE_PATTERNS: dict[SensitiveTopic, tuple[str, ...]] = {
    SensitiveTopic.REFUND_OR_DISPUTED_CHARGE: (
        r"terugbetal",
        r"restitu",
        r"geld\s+terug",
        r"\brefund",
        r"r(?:ü|ue)ckerstatt",
        r"dubbel\s+(?:afgeschreven|betaald|gefactureerd)",
        r"onterecht\s+(?:afgeschreven|gefactureerd|in\s+rekening)",
        r"charged\s+twice",
        r"double\s+charge",
        r"chargeback",
        r"storn(?:o|eren|ering)",
        r"terugboek",
    ),
    SensitiveTopic.CANCELLATION_OR_CONTRACT_CHANGE: (
        r"opzeg",
        r"zeg(?:gen)?\s+(?:we|wij|ik)\s+(?:\w+\s+)?op\b",
        r"k(?:ü|ue)ndig",
        r"cancel\s+(?:my|our|the)\s+(?:subscription|account|contract|plan)",
        r"terminate\s+(?:my|our|the)\s+(?:subscription|account|contract)",
        r"contract\s+(?:ontbind|beëindig|beeindig)",
    ),
    SensitiveTopic.PERSONAL_DATA_REQUEST: (
        r"\b(?:avg|gdpr|dsgvo)[- ]?(?:verzoek|request|anfrage)",
        r"inzageverzoek",
        r"recht\s+op\s+(?:inzage|vergetelheid|verwijdering)",
        r"right\s+to\s+(?:be\s+forgotten|erasure|access)",
        r"(?:verwijder|wis)\w*\s+(?:al\s+)?(?:mijn|onze|zijn|haar)\s+(?:persoons)?gegevens",
        r"delete\s+(?:my|our|all)\s+(?:personal\s+)?data",
    ),
    SensitiveTopic.SECURITY_INCIDENT: (
        r"gehack",
        r"\bhacked\b",
        r"\bhacker",
        r"datalek",
        r"data\s*breach",
        r"datenleck",
        r"phishing",
        r"onbevoegd",
        r"unauthori[sz]ed\s+(?:access|login)",
        r"unbefugt",
        r"account\s+(?:is\s+)?overgenomen",
    ),
    SensitiveTopic.LEGAL_THREAT: (
        r"advocaat",
        r"\blawyer",
        r"\battorney",
        r"anwalt",
        r"juridische\s+stappen",
        r"legal\s+action",
        r"rechtliche\s+schritte",
        r"rechtszaak",
        r"aanklagen",
        r"\blawsuit",
        r"ingebrekestelling",
    ),
    SensitiveTopic.CHURN_THREAT: (
        r"overstappen\s+naar",
        r"(?:we|wij|ik)\s+(?:gaan|stappen)\s+over\s+naar",
        r"\bconcurrent(?:en)?\b",
        r"andere\s+(?:aanbieder|leverancier|software)",
        r"switch(?:ing)?\s+to\s+(?:another|a\s+different|a\s+competitor)",
        r"looking\s+for\s+(?:an\s+)?alternative",
        r"leave\s+roosterly",
    ),
}
_COMPILED_PATTERNS = {
    topic: tuple(re.compile(p, re.IGNORECASE) for p in patterns)
    for topic, patterns in _SENSITIVE_PATTERNS.items()
}

# Brief: P1 = security or legal risk; P2 = money or the contract is involved.
_PRIORITY_FLOOR: dict[SensitiveTopic, Priority] = {
    SensitiveTopic.SECURITY_INCIDENT: Priority.P1,
    SensitiveTopic.LEGAL_THREAT: Priority.P1,
    SensitiveTopic.REFUND_OR_DISPUTED_CHARGE: Priority.P2,
    SensitiveTopic.CANCELLATION_OR_CONTRACT_CHANGE: Priority.P2,
    SensitiveTopic.CHURN_THREAT: Priority.P2,
}

_FLOOR_REASON: dict[Priority, str] = {
    Priority.P1: "security or legal risk (client rule)",
    Priority.P2: "money or the contract is involved (client rule)",
}

_GENERIC_HANDOFF_NOTE = (
    "Customer ticket needs a human; the automatic summary was unusable. "
    "Please read the original message."
)

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


@dataclass(frozen=True)
class PolicyConfig:
    confidence_threshold: float = 0.75
    max_reply_chars: int = 2000


def detect_sensitive_topics(text: str) -> frozenset[SensitiveTopic]:
    return frozenset(
        topic
        for topic, patterns in _COMPILED_PATTERNS.items()
        if any(pattern.search(text) for pattern in patterns)
    )


def _normalise_number(raw: str) -> str:
    value = float(raw.replace(",", "."))
    return f"{value:g}"


def _numbers_in(text: str) -> set[str]:
    return {_normalise_number(match) for match in _NUMBER.findall(text)}


class TicketPolicy:
    def __init__(self, knowledge_base: FaqKnowledgeBase, config: PolicyConfig) -> None:
        self._kb = knowledge_base
        self._config = config

    def close_blank_ticket(self, ticket: Ticket) -> Decision:
        return Decision(
            ticket_id=ticket.id,
            action=Action.CLOSE,
            language=Language.NL,
            category=Category.OTHER,
            priority=Priority.P4,
            reason="Empty ticket: no subject and no body. Closed without action.",
            confidence=1.0,
        )

    @staticmethod
    def correct_language(ticket: Ticket, assessment: Assessment) -> tuple[Assessment, str]:
        """Overrides the model's language label when the ticket text clearly says otherwise.

        Returns the (possibly corrected) assessment and a note, empty if nothing changed.
        When the text is ambiguous the model's label stands.
        """
        observed = detect_language(f"{ticket.subject}\n{ticket.body}")
        if observed is None or observed is assessment.language:
            return assessment, ""
        note = f"language corrected from {assessment.language} to {observed} by text check"
        return assessment.model_copy(update={"language": observed}), note

    def reply_blockers(
        self,
        ticket: Ticket,
        assessment: Assessment,
        keyword_topics: frozenset[SensitiveTopic],
    ) -> list[str]:
        """Every reason why the proposed reply must not be sent. Empty list means it may be."""
        blockers: list[str] = []
        topics = set(assessment.sensitive_topics) | keyword_topics
        if topics:
            blockers.append("always-human topic: " + ", ".join(sorted(topics)))
        if assessment.is_spam_or_empty:
            blockers.append("model marked the ticket as spam or empty")
        if not assessment.fully_answered_by_faq:
            blockers.append("the FAQ does not fully answer the question")
        unknown_ids = [i for i in assessment.faq_ids if self._kb.get(i) is None]
        if not assessment.faq_ids:
            blockers.append("no FAQ article cited")
        if unknown_ids:
            blockers.append("unknown FAQ ids cited: " + ", ".join(unknown_ids))
        if assessment.confidence < self._config.confidence_threshold:
            blockers.append(
                f"confidence {assessment.confidence:.2f} below "
                f"{self._config.confidence_threshold:.2f}"
            )
        draft = assessment.draft_reply.strip()
        if not draft:
            blockers.append("empty draft reply")
        elif len(draft) > self._config.max_reply_chars:
            blockers.append(f"draft longer than {self._config.max_reply_chars} characters")
        else:
            draft_language = detect_language(draft)
            if draft_language is not None and draft_language is not assessment.language:
                blockers.append(
                    f"draft is in {draft_language}, customer wrote in {assessment.language}"
                )
            articles = self._kb.articles_for(assessment.faq_ids)
            blockers.extend(self.grounding_problems(draft, ticket, articles))
        return blockers

    def grounding_problems(
        self, draft: str, ticket: Ticket, articles: list[FaqArticle]
    ) -> list[str]:
        """Checks a reply needs no Dutch to verify: payment data, URLs and numbers."""
        problems: list[str] = []
        if contains_payment_details(draft):
            problems.append("draft repeats payment details")
        for ref in extract_urls(draft):
            if not self._kb.is_known_url(ref):
                problems.append(f"draft mentions a URL not in the FAQ: {ref.host}{ref.path}")
        source = " ".join(a.text for a in articles) + f" {ticket.subject} {ticket.body}"
        unsupported = sorted(_numbers_in(draft) - _numbers_in(source))
        if unsupported:
            problems.append("draft has numbers not in the cited FAQ: " + ", ".join(unsupported))
        return problems

    def decide(
        self,
        ticket: Ticket,
        assessment: Assessment,
        keyword_topics: frozenset[SensitiveTopic],
        reply_blockers: list[str],
    ) -> tuple[Decision, list[str]]:
        """Turns the model's proposal into the final decision. Returns it with policy notes."""
        notes: list[str] = []
        topics = set(assessment.sensitive_topics) | keyword_topics
        priority = self._apply_priority_floor(assessment.priority, topics)
        if priority is not assessment.priority:
            notes.append(f"priority raised from {assessment.priority} to {priority}")

        action = self._choose_action(assessment, topics, reply_blockers, notes)
        handoff_note = None
        if action is Action.HANDOFF:
            handoff_note = self._handoff_note(assessment, notes)

        decision = Decision(
            ticket_id=ticket.id,
            action=action,
            language=assessment.language,
            category=assessment.category,
            priority=priority,
            reason=self._reason(assessment, action, priority, topics),
            draft_reply=assessment.draft_reply.strip() if action is Action.REPLY else None,
            handoff_note=handoff_note,
            confidence=assessment.confidence,
        )
        return decision, notes

    @staticmethod
    def _reason(
        assessment: Assessment,
        action: Action,
        priority: Priority,
        topics: set[SensitiveTopic],
    ) -> str:
        """Why this category and action, then the one-line priority reason the brief asks for.

        The priority line is kept whole; only the first part is shortened if space runs out.
        """
        main = assessment.reason.strip() or f"Classified as {assessment.category}."
        if action is not assessment.proposed_action:
            main = f"{main} Sent to a human by policy."

        if priority is not assessment.priority:
            floor_reasons = sorted(
                {_FLOOR_REASON[_PRIORITY_FLOOR[t]] for t in topics if t in _PRIORITY_FLOOR}
            )
            priority_line = f"Priority {priority}: {' and '.join(floor_reasons)}."
        elif assessment.priority_reason.strip():
            priority_line = f"Priority {priority}: {assessment.priority_reason.strip()}"
        else:
            priority_line = ""

        room = MAX_REASON_CHARS - len(priority_line) - 1
        reason = f"{_truncate(main, room)} {priority_line}".strip()
        return _truncate(mask_payment_details(reason), MAX_REASON_CHARS)

    def _choose_action(
        self,
        assessment: Assessment,
        topics: set[SensitiveTopic],
        reply_blockers: list[str],
        notes: list[str],
    ) -> Action:
        if topics:
            notes.append("always-human topic: " + ", ".join(sorted(topics)))
            return Action.HANDOFF
        if assessment.proposed_action is Action.CLOSE:
            if (
                assessment.is_spam_or_empty
                and assessment.confidence >= self._config.confidence_threshold
            ):
                return Action.CLOSE
            notes.append("close not confirmed as spam or empty with enough confidence")
            return Action.HANDOFF
        if assessment.proposed_action is Action.REPLY:
            if not reply_blockers:
                return Action.REPLY
            notes.append("auto-reply blocked: " + "; ".join(reply_blockers))
        return Action.HANDOFF

    @staticmethod
    def _apply_priority_floor(priority: Priority, topics: set[SensitiveTopic]) -> Priority:
        floors = [_PRIORITY_FLOOR[t] for t in topics if t in _PRIORITY_FLOOR]
        most_urgent = min([priority, *floors], key=lambda p: p.rank)
        return most_urgent

    @staticmethod
    def _handoff_note(assessment: Assessment, notes: list[str]) -> str:
        """The note for the support team, which must be in English.

        Takes the model's note if it reads as English, else its English summary, else a fixed
        English line. Any fallback is recorded in the policy notes.
        """
        why_human = "; ".join(notes)
        candidates = (
            ("handoff note", assessment.handoff_note.strip()),
            ("English summary", assessment.english_summary.strip()),
        )
        note = _GENERIC_HANDOFF_NOTE
        for name, text in candidates:
            if text and looks_english(text):
                note = text
                break
            if text:
                notes.append(f"model {name} was not in English, not used")
        if why_human:
            note = f"{note}\n\nWhy a human: {why_human}."
        return mask_payment_details(note)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
