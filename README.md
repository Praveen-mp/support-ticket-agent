# Roosterly support ticket agent

Reads every open helpdesk ticket and posts one decision per ticket: `reply`, `handoff` or `close`.

## How to run

```bash
uv sync                          # Python 3.12+
cp .env.example .env             # fill in keys and your email
uv run ticket-agent status       # check access
uv run ticket-agent eval         # score on the 10 labelled examples, posts nothing
uv run ticket-agent run --dry-run
uv run ticket-agent run          # decide and post
uv run ticket-agent reset --yes  # re-open practice tickets
uv run pytest                    # 123 tests, no network
```

## Decisions

1. **The model proposes, code decides.** One model call per ticket fills a fixed JSON schema; the model has no tools. `domain/policy.py` enforces the client's rules and can only move a ticket towards a human. Always-human topics (model flag or keyword backstop), P1, no cited FAQ, a partial answer or low confidence all become a handoff. A draft that fails a check (wrong language, or a URL, number, payment detail or claim not in the cited article) is rewritten once and checked again from scratch; if it still fails, it is a handoff.
2. **Dutch: work in Dutch, verify in English.** The draft is written from the original text, with no translation step (cheaper, no nuance lost). A second, independent call back-translates it and lists claims the FAQ does not support; code turns that into pass or fail. The back-translation is logged, so a non-Dutch reader can see what was sent. The language label itself is cross-checked in code (see the fix).
3. **Unsure means human.** Unsure means: no FAQ fully answers, the reply check fails, or confidence is below 0.75. A handoff costs minutes; a wrong reply reaches a customer.

Assumptions: a ticket only partly answered by the FAQ is a handoff. "How do I upgrade" is self-service (F05); "please change our plan" is a contract change. Security or legal risk sets at least P1, money or contract at least P2. Model: `openai/gpt-oss-120b` on Groq's free tier, low reasoning effort, strict JSON output. The provider is one setting (`LLM_PROVIDER=groq|anthropic`).

AI tools: built with Claude Code from my plan and design. *(Edit this line in your own words.)*

## Reliability

- **429:** calls are spaced at 4 per second; on a 429 we wait `Retry-After`.
- **503:** up to 6 attempts with exponential backoff and jitter. Safe for writes, because nothing is saved on a 503.
- **409:** never retried. Before deciding, each ticket is re-read and skipped if handled. If a post times out, we ask the helpdesk whether it landed before trying again.
- **Crash:** each decision is saved before it is posted, so a re-run continues with the open tickets and reuses saved decisions. Tested by killing a run mid-post.
- **Security:** keys only from env. Ticket text is wrapped as data and cannot close its own tag. IBANs and card numbers are masked before the model sees them and again in every log line (`logs/`, one line per ticket). A run stops at a cost budget or after 3 failures in a row.

## Eval and fix

`ticket-agent eval` runs the production code on Nordavo's 10 labelled examples, unchanged, and posts nothing. Results are in `eval/results/`.

| Run | Unsafe replies | Wrong closes | Missed replies | Language | Category | Priority | Action |
|---|---|---|---|---|---|---|---|
| Baseline | 0 | 0 | 4 | 9/10 | 8/10 | 8/10 | 6/10 |
| Tried: prompt v2 | 0 | 0 | 4 | 9/10 | 7/10 | 9/10 | 6/10 |
| Fix: language check | 0 | 0 | 4 | **10/10** | 8/10 | 8/10 | 6/10 |
| Fix 2: one rewrite + prompt v3 | 0 | 0 | **3** | 10/10 | 7/10 | 9/10 | **7/10** |

- **Wrong:** E01 is in English, but the model labelled it Dutch in both early runs. That label sets the reply language and the reply check compared against it, so an English customer could get a Dutch reply unnoticed.
- **Fix:** `domain/language.py` counts function words. It corrects the label when the ticket is clearly in another language, and blocks drafts in the wrong language. In the last run the model labelled E01 correctly by itself, so I replayed the stored labels of the first two runs through the check: 9/10 → 10/10 in both.
- **Tried first:** prompt v2 (the FAQ, not the category, decides the action). No gain: the model proposed a reply for E01, but the reply check blocked a claim the FAQ doesn't make.
- **Fix 2, from the practice run:** 16 of 19 tickets went to a human; 4 should not have. Two drafts added one sentence the FAQ does not say (correctly blocked), a feature request was not matched to F10, and an English P1 outage got a Dutch draft. Now a failed draft gets one rewrite, prompt v3 sends ideas to the roadmap (F10), and P1 is always a human. Re-deciding those tickets: 3 replies, the outage stays a handoff. In the eval E05 is fixed; the rewrite fired on E07 and E10 but the reply check still (rightly) said they do not answer the question.
- **Cost:** 10 tickets took 13 calls, 27,177 input tokens (16,890 cached) and 3,582 output tokens. At Groq's list price: (10,287 × $0.15 + 16,890 × $0.075 + 3,582 × $0.60) / 1M = $0.0050. With fix 2: 18 calls, $0.0069, so **about $0.07 per 100 tickets**. On the free tier it cost $0.

## What I would improve

- **Next step:** the 3 missed replies. E07 needs simple arithmetic on FAQ prices (25 × €6), which the number check blocks. E10 and E01 need more labelled tickets first: with 10 examples, a ±1 change is noise.
- **Change to the brief:** let one ticket get both a reply and a handoff, so a partly answered ticket can still get its FAQ answer.
