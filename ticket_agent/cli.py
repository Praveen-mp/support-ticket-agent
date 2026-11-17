"""Command line entry point and composition root (the only place that wires real adapters)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic
from pydantic import ValidationError

from ticket_agent.application.decider import TicketDecider
from ticket_agent.application.errors import AgentError, HelpdeskAuthError
from ticket_agent.application.runner import RunLimits, RunSummary, TicketRunner
from ticket_agent.config import Settings
from ticket_agent.domain.faq import FaqKnowledgeBase
from ticket_agent.domain.masking import mask_recursively
from ticket_agent.domain.policy import PolicyConfig, TicketPolicy
from ticket_agent.evaluation.evaluate import evaluate
from ticket_agent.evaluation.scoring import EvalReport, LabeledExample
from ticket_agent.infrastructure.anthropic_model import AnthropicJsonModel
from ticket_agent.infrastructure.helpdesk_client import HelpdeskClient
from ticket_agent.infrastructure.llm import (
    JsonModel,
    LlmReplyReviser,
    LlmReplyVerifier,
    LlmTicketAssessor,
    Pricing,
)
from ticket_agent.infrastructure.openai_compatible_model import OpenAICompatibleJsonModel
from ticket_agent.infrastructure.resilience import RateLimiter, RetryPolicy
from ticket_agent.infrastructure.storage import JsonDecisionStore, JsonlRunLog
from ticket_agent.prompts import ASSESS_PROMPT, REVISE_PROMPT, VERIFY_PROMPT

logger = logging.getLogger("ticket_agent")

EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_CONFIG = 2


class ConfigError(Exception):
    pass


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # Keep third-party request logs out of the console; our client logs what matters.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        settings = Settings()
        return int(args.handler(args, settings))
    except (ConfigError, ValidationError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except HelpdeskAuthError as exc:
        print(f"Helpdesk rejected the key or email: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except AgentError as exc:
        logger.error("run aborted: %s", exc, exc_info=args.verbose)
        print(f"Aborted: {exc}", file=sys.stderr)
        return EXIT_FAILURES
    except KeyboardInterrupt:
        print("Interrupted. Re-run the same command to continue where it stopped.", file=sys.stderr)
        return EXIT_FAILURES


# --- commands -------------------------------------------------------------------------------


def _cmd_status(_args: argparse.Namespace, settings: Settings) -> int:
    client = _build_helpdesk(settings)
    try:
        print(json.dumps(client.get_status(), indent=2, ensure_ascii=False))
    finally:
        client.close()
    return EXIT_OK


def _cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    helpdesk = _build_helpdesk(settings)
    run_log = JsonlRunLog(settings.log_dir / f"run-{run_id}{'-dry' if args.dry_run else ''}.jsonl")
    runner = TicketRunner(
        helpdesk=helpdesk,
        decider=_build_decider(settings),
        store=JsonDecisionStore(settings.state_path),
        run_log=run_log,
        limits=RunLimits(
            max_tickets=args.limit or settings.max_tickets_per_run,
            max_cost_usd=settings.max_run_cost_usd,
        ),
        run_id=run_id,
    )
    try:
        summary = runner.run(dry_run=args.dry_run)
    finally:
        helpdesk.close()
    _print_run_summary(summary, run_log.path)
    return EXIT_FAILURES if summary.failed or summary.stopped_early else EXIT_OK


def _cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    examples_path = settings.eval_dir / "examples.json"
    if args.refresh_examples or not examples_path.exists():
        _download_examples(settings, examples_path)
    raw = json.loads(examples_path.read_text(encoding="utf-8"))
    examples = [LabeledExample.model_validate(item) for item in raw]
    if args.only:
        wanted = set(args.only.split(","))
        examples = [e for e in examples if e.id in wanted]

    report = evaluate(_build_decider(settings), examples)
    result_path = settings.eval_dir / "results" / f"{args.name}.json"
    _write_eval_result(result_path, report, settings)
    _print_eval_report(report, result_path)
    return EXIT_OK


def _cmd_reset(args: argparse.Namespace, settings: Settings) -> int:
    if not args.yes:
        print("This re-opens all practice tickets and clears the local decision cache.")
        print("Run again with --yes to confirm.")
        return EXIT_FAILURES
    client = _build_helpdesk(settings)
    try:
        result = client.reset_practice()
    finally:
        client.close()
    JsonDecisionStore(settings.state_path).clear()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return EXIT_OK


# --- wiring ---------------------------------------------------------------------------------


def _build_helpdesk(settings: Settings) -> HelpdeskClient:
    if settings.helpdesk_api_key is None or not settings.candidate_email:
        raise ConfigError("HELPDESK_API_KEY and CANDIDATE_EMAIL must be set (see .env.example)")
    return HelpdeskClient(
        base_url=settings.helpdesk_base_url,
        api_key=settings.helpdesk_api_key.get_secret_value(),
        candidate_email=settings.candidate_email,
        candidate_name=settings.candidate_name,
        rate_limiter=RateLimiter(settings.requests_per_second),
        retry_policy=RetryPolicy(),
    )


def _build_decider(settings: Settings) -> TicketDecider:
    knowledge_base = FaqKnowledgeBase.from_file(settings.faq_path)
    model = _build_model(settings)
    return TicketDecider(
        assessor=LlmTicketAssessor(model, knowledge_base),
        verifier=LlmReplyVerifier(model),
        reviser=LlmReplyReviser(model),
        policy=TicketPolicy(knowledge_base, PolicyConfig(settings.confidence_threshold)),
        knowledge_base=knowledge_base,
        model_id=f"{settings.llm_provider}:{settings.model_name}:{settings.llm_effort}",
    )


def _build_model(settings: Settings) -> JsonModel:
    defaults = settings.provider_defaults
    pricing = Pricing(
        input_per_mtok=(
            settings.price_input_per_mtok
            if settings.price_input_per_mtok is not None
            else defaults.price_input_per_mtok
        ),
        output_per_mtok=(
            settings.price_output_per_mtok
            if settings.price_output_per_mtok is not None
            else defaults.price_output_per_mtok
        ),
        cache_read_factor=defaults.cache_read_factor,
        cache_write_factor=defaults.cache_write_factor,
    )
    if settings.llm_provider == "groq":
        if settings.groq_api_key is None:
            raise ConfigError("GROQ_API_KEY must be set (see .env.example)")
        return OpenAICompatibleJsonModel(
            base_url=settings.groq_base_url,
            api_key=settings.groq_api_key.get_secret_value(),
            model=settings.model_name,
            effort=settings.llm_effort,
            pricing=pricing,
            rate_limiter=RateLimiter(settings.model_requests_per_minute / 60),
            # The free tier's per-minute token budget can need up to a minute to refill.
            retry_policy=RetryPolicy(max_attempts=8, max_retry_after_s=60),
        )
    if settings.anthropic_api_key is None:
        raise ConfigError("ANTHROPIC_API_KEY must be set (see .env.example)")
    client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key.get_secret_value(), timeout=60.0, max_retries=3
    )
    return AnthropicJsonModel(client, settings.model_name, settings.llm_effort, pricing)


def _download_examples(settings: Settings, path: Path) -> None:
    client = _build_helpdesk(settings)
    try:
        examples = client.get_examples()
    finally:
        client.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(examples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("saved labelled examples: path=%s count=%s", path, len(examples))


# --- output ---------------------------------------------------------------------------------


def _write_eval_result(path: Path, report: EvalReport, settings: Settings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "provider": settings.llm_provider,
        "model": settings.model_name,
        "prompts": [ASSESS_PROMPT, VERIFY_PROMPT, REVISE_PROMPT],
        "effort": settings.llm_effort,
        "summary": report.summary(),
        "examples": [r.to_row() for r in report.results],
    }
    path.write_text(
        json.dumps(mask_recursively(document), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _print_eval_report(report: EvalReport, path: Path) -> None:
    print(f"\n{'id':<6} {'expected':<34} {'predicted':<34} notes")
    for result in report.results:
        label = result.label
        expected = f"{label.action}/{label.category}/{label.priority}/{label.language}"
        if result.outcome is None:
            predicted, notes = "ERROR", result.error
        else:
            d = result.outcome.decision
            predicted = f"{d.action}/{d.category}/{d.priority}/{d.language}"
            flags = [
                name
                for name, flag in (
                    ("UNSAFE_REPLY", result.unsafe_reply),
                    ("WRONG_CLOSE", result.wrong_close),
                    ("MISSED_REPLY", result.missed_reply),
                )
                if flag
            ]
            notes = " ".join(flags)
        print(f"{result.example_id:<6} {expected:<34} {predicted:<34} {notes}")
    print("\n" + json.dumps(report.summary(), indent=2))
    print(f"\nFull results: {path}")


def _print_run_summary(summary: RunSummary, log_path: Path) -> None:
    print(
        json.dumps(
            {
                "run_id": summary.run_id,
                "dry_run": summary.dry_run,
                "tickets_seen": summary.seen,
                "submitted": summary.submitted,
                "actions": summary.actions,
                "skipped_already_handled": summary.skipped_already_handled,
                "reused_decisions": summary.reused_decisions,
                "conflicts_409": summary.conflicts,
                "failed": summary.failed,
                "stopped_early": summary.stopped_early or None,
                "model_calls": summary.usage.calls,
                "cost_usd": round(summary.usage.cost_usd, 6),
                "log": str(log_path),
            },
            indent=2,
        )
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ticket-agent", description="Triage Roosterly helpdesk tickets."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="show helpdesk status for this account")
    status.set_defaults(handler=_cmd_status)

    run = sub.add_parser("run", help="decide and post every open ticket")
    run.add_argument("--dry-run", action="store_true", help="decide but post nothing")
    run.add_argument("--limit", type=int, default=0, help="handle at most N tickets")
    run.set_defaults(handler=_cmd_run)

    ev = sub.add_parser("eval", help="score the agent on the labelled examples")
    ev.add_argument("--name", default="latest", help="result file name under eval/results/")
    ev.add_argument("--only", default="", help="comma-separated example ids, e.g. E01,E03")
    ev.add_argument("--refresh-examples", action="store_true", help="download examples again")
    ev.set_defaults(handler=_cmd_eval)

    reset = sub.add_parser("reset", help="re-open all practice tickets")
    reset.add_argument("--yes", action="store_true", help="confirm")
    reset.set_defaults(handler=_cmd_reset)

    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
