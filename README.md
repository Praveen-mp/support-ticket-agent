# Roosterly support ticket agent

Reads every open helpdesk ticket and posts one decision per ticket: `reply`, `handoff` or `close`.

## How to run

```bash
uv sync                                   # Python 3.12+, installs everything
cp .env.example .env                      # then fill in the keys and your email
uv run ticket-agent status                # check access
uv run ticket-agent eval --name baseline  # score on the 10 labelled examples, posts nothing
uv run ticket-agent run --dry-run         # decide all open tickets, post nothing
uv run ticket-agent run                   # decide and post
uv run ticket-agent reset --yes           # re-open practice tickets, clear local cache
uv run pytest                             # 106 tests, no network
```

## Decisions

1. **The model proposes, code decides.** One model call per ticket fills a fixed JSON schema (no tools). `domain/policy.py` then enforces the client's rules and can only move a ticket towards a human: always-human topics (model flag *or* a Dutch/English/German keyword backstop), no cited FAQ, partial answers, low confidence, or a draft with a URL, number or payment detail not in the cited article all become a handoff.
2. **Dutch: work in Dutch, verify in English.** The drafting call works on the original text (no translate-first step: cheaper, no nuance lost). A second, independent call back-translates every draft literally and lists claims the cited FAQ does not support; code turns those observations into pass/fail. The back-translation is in the log, so a non-Dutch reader can see what was sent.
3. **Fail towards a human.** Unsure means: no FAQ fully answers, the reply check fails, or confidence < 0.75. Handoff costs minutes; a wrong reply reaches a customer.

Assumptions where the brief is open: a ticket only partly answered by the FAQ is a handoff (one action per ticket). "How do I upgrade" is a self-service question (F05), "please change our plan" is a contract change. Legal or security risk floors priority at P1, money or contract at P2. Model: `openai/gpt-oss-120b` on Groq's free tier, reasoning effort `low`, strict JSON schema output. The provider is one setting (`LLM_PROVIDER=groq|anthropic`); the prompts, rules and eval do not change with it.

AI tools: built with Claude Code from my plan and design. *(Edit this line in your own words.)*

## Reliability

- **429:** a client-side limiter spaces calls at 4/s; on a 429 we wait `Retry-After`.
- **503:** bounded retries (6) with exponential backoff and jitter. Safe for writes, because the API saves nothing on a 503.
- **409, no double answers:** before deciding, the ticket is re-read and skipped if handled. If a post times out we ask the helpdesk whether it landed before retrying. A 409 is never retried.
- **After a crash:** each decision is saved before it is posted, so a re-run continues with the remaining open tickets and reuses saved decisions instead of paying again. Tested by killing a run mid-post (`tests/test_runner.py`).
- **Security:** keys only from env; ticket text is wrapped as data and cannot close its own tag; IBANs and card numbers are masked before the model sees them and again in every log line (`logs/`, one JSON line per ticket). Budgets: max cost per run, stop after 3 failures in a row.

## Eval and fix

`eval/examples.json` holds Nordavo's labels, unchanged. Safety first: unsafe replies, wrong closes, then accuracy per field.

*Pending: baseline, the fix, the result after, and cost per 100 tickets (from measured tokens × $0.10/$0.50 per million).*

## What I would improve

*Pending.*
