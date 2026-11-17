"""Runs the agent over all open tickets, safely re-runnable.

No-double-answer guarantee, in order of defence:
1. Before deciding, re-read the ticket from the helpdesk and skip it if it is handled.
2. Save the decision locally before posting (write-ahead), so a crash after posting
   re-sends the same decision at most, and step 1 catches it first.
3. If a post times out, ask the helpdesk whether it landed before trying again.
4. A 409 is never retried.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ticket_agent.application.decider import TicketDecider
from ticket_agent.application.errors import (
    AlreadyHandledError,
    HelpdeskAuthError,
    HelpdeskError,
    ModelError,
    OutcomeUnknownError,
    TicketNotFoundError,
)
from ticket_agent.application.ports import DecisionStore, Helpdesk, RunLog
from ticket_agent.domain.models import DecisionOutcome, Ticket, Usage

logger = logging.getLogger(__name__)

MAX_SUBMIT_ATTEMPTS = 3
MAX_CONSECUTIVE_FAILURES = 3


@dataclass(frozen=True)
class RunLimits:
    max_tickets: int = 200
    max_cost_usd: float = 0.50


@dataclass
class RunSummary:
    run_id: str
    dry_run: bool
    seen: int = 0
    submitted: int = 0
    skipped_already_handled: int = 0
    failed: int = 0
    conflicts: int = 0
    reused_decisions: int = 0
    actions: dict[str, int] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    stopped_early: str = ""


class TicketRunner:
    def __init__(
        self,
        *,
        helpdesk: Helpdesk,
        decider: TicketDecider,
        store: DecisionStore,
        run_log: RunLog,
        limits: RunLimits,
        run_id: str,
    ) -> None:
        self._helpdesk = helpdesk
        self._decider = decider
        self._store = store
        self._log = run_log
        self._limits = limits
        self._run_id = run_id

    def run(self, *, dry_run: bool = False) -> RunSummary:
        summary = RunSummary(run_id=self._run_id, dry_run=dry_run)
        tickets = self._helpdesk.list_open_tickets(self._limits.max_tickets)
        logger.info("run started: run_id=%s open_tickets=%s", self._run_id, len(tickets))

        consecutive_failures = 0
        for ticket in tickets:
            if summary.usage.cost_usd >= self._limits.max_cost_usd:
                summary.stopped_early = (
                    f"cost budget reached: ${summary.usage.cost_usd:.4f} "
                    f">= ${self._limits.max_cost_usd:.2f}"
                )
                logger.warning("stopping: %s", summary.stopped_early)
                break
            summary.seen += 1
            succeeded = self._handle_ticket(ticket, summary, dry_run=dry_run)
            consecutive_failures = 0 if succeeded else consecutive_failures + 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                summary.stopped_early = f"{consecutive_failures} tickets failed in a row"
                logger.error("stopping: %s", summary.stopped_early)
                break

        logger.info(
            "run finished: run_id=%s submitted=%s skipped=%s failed=%s cost_usd=%.4f",
            self._run_id,
            summary.submitted,
            summary.skipped_already_handled,
            summary.failed,
            summary.usage.cost_usd,
        )
        return summary

    def _handle_ticket(self, ticket: Ticket, summary: RunSummary, *, dry_run: bool) -> bool:
        """Returns False only for failures that may mean the system is unhealthy."""
        started = time.monotonic()
        retries_before = self._helpdesk.retry_count
        # Overwritten on every normal path; stays only if the run is interrupted or crashes.
        entry: dict[str, Any] = {"ticket_id": ticket.id, "status": "aborted"}
        try:
            if self._helpdesk.get_ticket_state(ticket.id).handled:
                summary.skipped_already_handled += 1
                entry["status"] = "skipped_already_handled"
                return True

            outcome, reused = self._decide(ticket)
            summary.usage = summary.usage + (Usage() if reused else outcome.usage)
            summary.reused_decisions += int(reused)
            entry.update(_describe(outcome, reused=reused))

            if dry_run:
                entry["status"] = "dry_run"
                return True

            entry["status"] = self._submit(outcome)
            if entry["status"] == "conflict_already_handled":
                summary.conflicts += 1
            else:
                summary.submitted += 1
                action = outcome.decision.action.value
                summary.actions[action] = summary.actions.get(action, 0) + 1
            return True
        except HelpdeskAuthError:
            raise
        except TicketNotFoundError as exc:
            entry.update(status="skipped_not_found", error=str(exc))
            return True
        except (HelpdeskError, ModelError) as exc:
            summary.failed += 1
            if isinstance(exc, ModelError):
                summary.usage = summary.usage + Usage(cost_usd=exc.cost_usd)
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            logger.error("ticket failed: ticket_id=%s error=%s", ticket.id, exc, exc_info=True)
            return False
        finally:
            entry["http_retries"] = self._helpdesk.retry_count - retries_before
            entry["duration_ms"] = round((time.monotonic() - started) * 1000)
            self._write_log(entry)

    def _decide(self, ticket: Ticket) -> tuple[DecisionOutcome, bool]:
        fingerprint = self._decider.fingerprint(ticket)
        cached = self._store.load(ticket.id, fingerprint)
        if cached is not None:
            return cached, True
        outcome = self._decider.decide(ticket)
        self._store.save(ticket.id, fingerprint, outcome)
        return outcome, False

    def _submit(self, outcome: DecisionOutcome) -> str:
        decision = outcome.decision
        for attempt in range(1, MAX_SUBMIT_ATTEMPTS + 1):
            try:
                self._helpdesk.submit_decision(decision)
                self._store.mark_submitted(decision.ticket_id)
                return "submitted"
            except AlreadyHandledError:
                # Should not happen given the pre-check; logged loudly because it means a
                # second process, or a bug, raced us.
                logger.warning("409 already handled: ticket_id=%s", decision.ticket_id)
                self._store.mark_submitted(decision.ticket_id)
                return "conflict_already_handled"
            except OutcomeUnknownError:
                logger.warning(
                    "post outcome unknown, reconciling: ticket_id=%s attempt=%s",
                    decision.ticket_id,
                    attempt,
                )
                if self._helpdesk.get_ticket_state(decision.ticket_id).handled:
                    self._store.mark_submitted(decision.ticket_id)
                    return "submitted_reconciled"
        raise OutcomeUnknownError(
            f"could not confirm the decision for {decision.ticket_id} after "
            f"{MAX_SUBMIT_ATTEMPTS} attempts"
        )

    def _write_log(self, entry: dict[str, Any]) -> None:
        line = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "run_id": self._run_id,
            **entry,
        }
        self._log.write(line)


def _describe(outcome: DecisionOutcome, *, reused: bool) -> dict[str, Any]:
    decision = outcome.decision
    described: dict[str, Any] = {
        "action": decision.action.value,
        "language": decision.language.value,
        "category": decision.category.value,
        "priority": decision.priority.value,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "policy_notes": list(outcome.policy_notes),
        "decision_reused": reused,
        "tokens_in": outcome.usage.input_tokens,
        "tokens_out": outcome.usage.output_tokens,
        "cost_usd": 0.0 if reused else round(outcome.usage.cost_usd, 6),
    }
    if outcome.assessment is not None:
        described["model_proposed"] = outcome.assessment.proposed_action.value
        described["faq_ids"] = list(outcome.assessment.faq_ids)
        described["sensitive_topics"] = [t.value for t in outcome.assessment.sensitive_topics]
        described["ticket_summary_en"] = outcome.assessment.english_summary
    if decision.handoff_note:
        described["handoff_note"] = decision.handoff_note
    if decision.draft_reply:
        described["draft_reply"] = decision.draft_reply
    if outcome.reply_check is not None:
        described["reply_check_passed"] = outcome.reply_check.passed
        described["reply_back_translation_en"] = outcome.reply_check.back_translation
    return described
