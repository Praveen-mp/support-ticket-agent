"""Runs the production decider over labelled examples. Nothing is posted to the helpdesk."""

from __future__ import annotations

import logging

from ticket_agent.application.decider import TicketDecider
from ticket_agent.application.errors import ModelError
from ticket_agent.evaluation.scoring import EvalReport, ExampleResult, LabeledExample

logger = logging.getLogger(__name__)


def evaluate(decider: TicketDecider, examples: list[LabeledExample]) -> EvalReport:
    report = EvalReport()
    for example in examples:
        try:
            outcome = decider.decide(example.to_ticket())
            report.results.append(ExampleResult(example.id, example.label, outcome))
        except ModelError as exc:
            logger.error("eval example failed: example_id=%s error=%s", example.id, exc)
            report.results.append(ExampleResult(example.id, example.label, None, error=str(exc)))
    return report
