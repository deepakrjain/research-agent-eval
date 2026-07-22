# Decisions Log

This file records non-obvious design decisions made during development,
with reasoning. It exists so that anyone reviewing this project (including
future-me) can understand *why* things were built this way, not just *what*
was built.

---

## 2026-07-21 — Zero-cost architecture choices

**Decision:** Use Groq (Llama 3) as the agent LLM, Google Gemini as the
judge LLM, DuckDuckGo for search, and requests+BeautifulSoup for scraping.

**Why:** The entire stack must run with zero cost and no credit card.
- **Groq over OpenAI/Anthropic:** Groq offers a generous free tier with
  fast inference on open-weight models (Llama 3.1/3.3). OpenAI and
  Anthropic both require a credit card even for free credits.
- **Gemini as judge, not Groq again:** Using a *different* model from a
  *different* vendor as the judge avoids self-preference bias — a known
  problem where LLMs rate their own outputs (or outputs from similar
  models) more favorably. More on this in Phase 6.
- **DuckDuckGo over Tavily/SerpAPI/Google:** `duckduckgo-search` is a
  Python package that requires no API key at all. Tavily has a free tier
  but requires signup and has a request cap. SerpAPI requires a credit
  card. For a portfolio project where anyone should be able to clone and
  run, zero-signup dependencies are ideal.
- **requests+BS4 over Playwright/Selenium:** Most informational pages we
  need to read render fine without JavaScript. A headless browser adds
  complexity, memory overhead, and CI setup pain. If we hit JS-rendered
  pages, we'll add a fallback — but starting simple is the right call.

**Decision:** Use `.env` + `python-dotenv` for secrets, with `.env` in
`.gitignore` and `.env.example` checked in.

**Why:** Even on a solo project, committing an API key to a public GitHub
repo means it gets scraped by bots within minutes. `.env.example` serves
as documentation for anyone cloning the repo ("here's what keys you need
and where to get them"), while `.env` stays local. This is the standard
pattern in production — getting into the habit now avoids a painful lesson
later.

**Decision:** Use Pydantic for *all* structured LLM outputs (no free-text
parsing anywhere).

**Why:** LLMs are stochastic — they don't always format responses the same
way. Regex parsing of free text is brittle: a model might say "Yes" one
time and "Yes, I have enough information" the next. Pydantic models give
us a contract: either the response validates against the schema or it
fails explicitly, which is far easier to debug than silently wrong parsing.

---

## 2026-07-21 — Iteration cap and loop guardrails (Phase 2)

**Decision:** Set the default iteration cap to 5 rounds.

**Why:** This is a balance between thoroughness and cost/time:
- **Too low (1-2):** The agent barely explores — it's just a pipeline with
  extra steps. You lose the benefit of the loop.
- **Too high (10-20):** On the Groq free tier, each iteration costs a
  search + N page fetches + a decision LLM call. 20 iterations could burn
  through daily rate limits fast. Also, diminishing returns: after 5
  rounds, you've likely seen the same information reworded.
- **5 is a starting point.** We can tune it based on eval results in Phase
  8. The cap is a constant in `config.py`, not hardcoded in the loop.

**Decision:** Default to `enough_info=True` when decision parsing fails.

**Why:** This is a fail-safe design. If the LLM returns malformed JSON
that Pydantic can't parse, we have two options: (a) retry the decision
call, or (b) stop the loop. We chose (b) because retrying could also
fail, leading to an infinite retry loop. Stopping with whatever we have
is safer — a partial answer is better than an infinite loop.

**Decision:** Use dependency injection (`_decision_fn`, `_search_fn`, etc.)
for testing the loop, rather than mocking at the library level.

**Why:** Mocking `groq.Groq` or `duckduckgo_search.DDGS` at the library
level is fragile — it couples tests to the internal implementation of those
libraries. Injecting fake functions at the loop level means our tests don't
care whether we use Groq, OpenAI, or a local model. If we swap providers,
the tests still work unchanged.

---

## 2026-07-22 — Query decomposition strategy (Phase 3)

**Decision:** Run the planner BEFORE the loop, not inside it.

**Why:** Planning is a one-time upfront cost. If we planned inside the
loop, we'd re-decompose the question every iteration (same input → same
output → wasted LLM call). By planning once and feeding sub-queries into
the loop as a queue, we get focused searches without repeated planning.

**Decision:** Target 2-4 sub-queries, soft cap at 6 via Pydantic validator.

**Why:** With a 5-iteration loop cap, each sub-query consumes one
iteration. If the planner generates 4 sub-queries, that leaves only 1
iteration for the decide-LLM to request follow-up searches. The 2-4
target balances thoroughness (multiple angles) with flexibility (room for
the agent to course-correct). The validator truncates at 6 rather than
raising an error, because a slightly-over plan is better than a crashed
agent.

**Decision:** On planner failure, fall back to using the original question.

**Why:** The planner is an optimization, not a requirement. If the LLM
returns malformed JSON or the API call fails, we don't want the entire
agent to crash. Falling back to the original question means we're
effectively running the Phase 2 behavior (no decomposition), which still
produces answers — just less well-targeted ones.

---

## 2026-07-22 — Cited Synthesis and Self-Correction (Phase 4)

**Decision:** Force JSON output for synthesis to extract both the answer and the citations used.

**Why:** It is extremely difficult to parse a plain-text answer to reliably figure out which sources were cited, especially if the LLM hallucinated a citation like `[5]` when we only have 3 sources. Forcing JSON `{"answer_text": "...", "citations_used": [...]}` makes parsing deterministic and easy.

**Decision:** Add a retry loop to self-correct hallucinated citations.

**Why:** LLMs hallucinate citations frequently, blending sources or citing source `[N]` out of habit when it doesn't exist. Instead of silently outputting a broken citation or failing, we catch the `ValidationError` and feed the error back to the LLM. Doing this "self-correction" inside `synthesizer.py` isolates the complexity from the main agent loop.
