"""Decides one ticket. Pure with respect to the helpdesk: it never posts anything.

The same code path runs in production and in the eval, so the eval measures what ships.
"""

from __future__ import annotations

import hashlib

from ticket_agent.application.ports import ReplyVerifier, TicketAssessor
from ticket_agent.domain.faq import FaqKnowledgeBase
from ticket_agent.domain.masking import mask_payment_details
from ticket_agent.domain.models import Action, DecisionOutcome, Ticket, Usage
from ticket_agent.domain.policy import TicketPolicy, detect_sensitive_topics

# Bump when prompts or rules change, so cached decisions from older code are not reused.
PIPELINE_VERSION = "2026-11-17.3"


class TicketDecider:
    def __init__(
        self,
        assessor: TicketAssessor,
        verifier: ReplyVerifier,
        policy: TicketPolicy,
        knowledge_base: FaqKnowledgeBase,
        model_id: str,
    ) -> None:
        self._assessor = assessor
        self._verifier = verifier
        self._policy = policy
        self._kb = knowledge_base
        self._model_id = model_id

    def fingerprint(self, ticket: Ticket) -> str:
        """Identifies a decision: same ticket text, same pipeline, same model."""
        material = "\x1f".join(
            [PIPELINE_VERSION, self._model_id, ticket.id, ticket.subject, ticket.body]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def decide(self, ticket: Ticket) -> DecisionOutcome:
        masked = ticket.model_copy(
            update={
                "subject": mask_payment_details(ticket.subject),
                "body": mask_payment_details(ticket.body),
            }
        )
        if masked.is_blank:
            return DecisionOutcome(
                decision=self._policy.close_blank_ticket(masked),
                policy_notes=("blank ticket closed without a model call",),
            )

        assessed = self._assessor.assess(masked)
        # The language label drives the reply language and the reply check, so it is
        # cross-checked against the ticket text before anything else uses it.
        assessment, language_note = self._policy.correct_language(masked, assessed.value)
        usage: Usage = assessed.usage
        keyword_topics = detect_sensitive_topics(f"{masked.subject}\n{masked.body}")

        reply_check = None
        blockers: list[str] = []
        if assessment.proposed_action is Action.REPLY:
            blockers = self._policy.reply_blockers(masked, assessment, keyword_topics)
            # Only pay for the second call when the reply could actually be sent.
            if not blockers:
                checked = self._verifier.verify(
                    masked,
                    assessment.draft_reply,
                    assessment.language,
                    self._kb.articles_for(assessment.faq_ids),
                )
                reply_check = checked.value
                usage = usage + checked.usage
                blockers.extend(f"reply check: {p}" for p in reply_check.problems)

        decision, notes = self._policy.decide(masked, assessment, keyword_topics, blockers)
        if language_note:
            notes.insert(0, language_note)
        return DecisionOutcome(
            decision=decision,
            assessment=assessment,
            reply_check=reply_check,
            policy_notes=tuple(notes),
            usage=usage,
        )
