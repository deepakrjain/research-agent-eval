"""
loop.py — The search-read-decide loop: the core of agentic behavior.

THIS IS WHERE THE PROJECT STOPS BEING A PIPELINE AND BECOMES AN AGENT.

A pipeline executes a fixed sequence: search → read → answer.
An agent executes a LOOP: search → read → DECIDE → (maybe search again).

The key difference is the DECIDE step: an explicit LLM call that looks
at what's been gathered so far and returns a structured decision:
    {enough_info: bool, next_query: str | None, reasoning: str}

This is a simplified version of the ReAct (Reason + Act) pattern:
- Reason: "I've found info about X but nothing about Y"
- Act: "Search for Y specifically"
- Observe: Read the new results
- Repeat

GUARDRAILS:
1. Hard iteration cap (default 5) — prevents infinite loops even if
   the model keeps saying "not enough." In production, infinite loops
   are the #1 cost and reliability failure for agents.
2. Query deduplication — never run the same search query twice
3. Source deduplication — never fetch the same URL twice
4. Graceful degradation — if search returns nothing or extraction
   fails on all pages, the loop doesn't crash; it works with what
   it has.
"""

import json
from groq import Groq
from rich.console import Console

from agent.config import get_groq_api_key, GROQ_MODEL, MAX_LOOP_ITERATIONS
from agent.searcher import search
from agent.extractor import extract_content
from agent.synthesizer import synthesize
from agent.models import SearchDecision, SourceDocument, AgentResult
from agent.planner import decompose_question

console = Console()

# ---- Decision prompt ----
# This prompt asks the LLM to evaluate gathered information and decide
# whether to continue searching. It MUST return valid JSON matching
# the SearchDecision schema.
DECISION_SYSTEM_PROMPT = """You are a research planning assistant. Your job is to evaluate whether enough information has been gathered to answer a question, or whether more searching is needed.

You will receive:
1. The original question
2. The search queries already run
3. Summaries of information gathered so far

You must respond with a JSON object with exactly these fields:
{
    "enough_info": true/false,
    "reasoning": "brief explanation of what you know and what's missing",
    "next_query": "the next search query to try" or null
}

Rules:
- Set enough_info to true if the gathered information can answer the question well
- Set enough_info to false if critical aspects of the question are unanswered
- If enough_info is false, next_query MUST be a NEW query different from all previous queries
- If enough_info is true, set next_query to null
- Be efficient: don't ask for more searches if the core question is answered
- Respond ONLY with the JSON object, no other text"""


def make_decision(
    question: str,
    queries_used: list[str],
    gathered_texts: list[str],
    groq_client: Groq,
) -> SearchDecision:
    """
    Ask the LLM whether we have enough information to answer the question.

    Returns a validated SearchDecision object.

    WHY THIS IS A SEPARATE FUNCTION:
    Isolating the decision logic makes it independently testable and
    swappable. We can mock this function in tests to simulate "always
    says not enough" or "always says enough" scenarios.

    WHY JSON MODE + PYDANTIC (not free-text parsing):
    Groq supports response_format={"type": "json_object"}, which
    constrains the model to output valid JSON. We then validate that
    JSON against our Pydantic schema. This two-layer approach means:
    1. The model CAN'T return non-JSON (Groq enforces this)
    2. The JSON CAN'T have wrong field names/types (Pydantic enforces this)
    Compare to regex: model says "Yes I have enough" — does "Yes" match
    your regex? What about "Yes, I believe so"? Pydantic doesn't care
    about phrasing; it checks structure.
    """
    # Build a summary of what we've gathered
    gathered_summary = ""
    if gathered_texts:
        for i, text in enumerate(gathered_texts, 1):
            # Truncate each source summary to keep the decision prompt manageable
            truncated = text[:500] + "..." if len(text) > 500 else text
            gathered_summary += f"\n--- Source {i} ---\n{truncated}\n"
    else:
        gathered_summary = "\n(No information gathered yet)\n"

    user_message = f"""Original question: {question}

Queries already searched: {json.dumps(queries_used)}

Information gathered so far:
{gathered_summary}

Evaluate: Do we have enough information to comprehensively answer the original question? Respond with the JSON object."""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": DECISION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,  # Very low — we want deterministic decisions
        max_tokens=256,
        response_format={"type": "json_object"},
    )

    raw_json = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw_json)
        decision = SearchDecision(**parsed)
    except (json.JSONDecodeError, Exception) as e:
        # If parsing fails, default to "enough info" to avoid infinite loops
        # This is a SAFETY MEASURE: if the model returns garbage, we stop
        # rather than continuing forever.
        console.print(f"[yellow]⚠ Decision parsing failed: {e}. Defaulting to enough_info=True[/yellow]")
        decision = SearchDecision(
            enough_info=True,
            reasoning=f"Decision parsing failed ({e}), stopping to be safe.",
            next_query=None,
        )

    return decision


def run_agent_loop(
    question: str,
    max_iterations: int = MAX_LOOP_ITERATIONS,
    _decision_fn=None,
    _search_fn=None,
    _extract_fn=None,
    _synthesize_fn=None,
    _planner_fn=None,
) -> AgentResult:
    """
    The main search-read-decide loop.

    Args:
        question: The user's question.
        max_iterations: Hard cap on loop iterations (safety guardrail).
        _decision_fn: Injectable decision function (for testing).
        _search_fn: Injectable search function (for testing).
        _extract_fn: Injectable extraction function (for testing).
        _synthesize_fn: Injectable synthesis function (for testing).
        _planner_fn: Injectable planner function (for testing).

    Returns:
        AgentResult with the answer, sources, and loop metadata.

    WHY INJECTABLE FUNCTIONS (_decision_fn, etc.):
    We need to test that the loop cap works and dedup works WITHOUT
    making real API calls. By accepting optional function overrides,
    tests can inject fake searchers, fake extractors, and fake decision
    makers. This is dependency injection — the same pattern used in
    production agent systems.
    """
    # Use injected functions or real implementations
    decision_fn = _decision_fn
    search_fn = _search_fn or search
    extract_fn = _extract_fn or extract_content
    synthesize_fn = _synthesize_fn or synthesize
    planner_fn = _planner_fn or decompose_question

    # Initialize the Groq client once (reused across all decision calls)
    groq_client = None
    if decision_fn is None:
        groq_client = Groq(api_key=get_groq_api_key())
        decision_fn = lambda q, queries, texts: make_decision(q, queries, texts, groq_client)

    # ---- Phase 3: Query Decomposition ----
    # Before entering the loop, decompose the question into focused
    # sub-queries. These become the initial queries for the loop.
    # The decide step only kicks in AFTER all sub-queries are processed.
    console.print(f"\n[bold magenta]📋 Decomposing question into sub-queries...[/bold magenta]")
    plan = planner_fn(question)
    pending_queries = list(plan.sub_queries)  # Queue of queries to process

    # ---- State tracking ----
    all_sources: list[SourceDocument] = []       # All successfully extracted sources
    all_source_texts: list[str] = []             # Just the text, for the decision LLM
    queries_used: list[str] = []                 # Track queries for dedup
    urls_seen: set[str] = set()                  # Track URLs for dedup
    hit_cap = False

    # Pull the first query from the planner's queue
    current_query = pending_queries.pop(0) if pending_queries else question

    for iteration in range(1, max_iterations + 1):
        console.print(f"\n[bold cyan]━━━ Iteration {iteration}/{max_iterations} ━━━[/bold cyan]")

        # ---- DEDUP CHECK: Skip if we've already run this exact query ----
        if current_query in queries_used:
            console.print(f"[yellow]⚠ Query already used: '{current_query}'. Skipping.[/yellow]")
            # If the model suggested a duplicate query, we're probably
            # going in circles. Stop gracefully.
            break

        # ---- ACT: Search ----
        console.print(f"[blue]🔍 Searching:[/blue] {current_query}")
        queries_used.append(current_query)
        results = search_fn(current_query)

        if not results:
            console.print("[yellow]⚠ No results found for this query.[/yellow]")
        else:
            console.print(f"[green]✓ Found {len(results)} results[/green]")

        # ---- ACT: Extract (with URL dedup) ----
        new_sources_this_round = 0
        for r in results:
            url = r.url if hasattr(r, 'url') else r.get('url', '')
            title = r.title if hasattr(r, 'title') else r.get('title', '')

            # DEDUP: Skip already-fetched URLs
            if url in urls_seen:
                console.print(f"  [dim]↩ Already read: {title}[/dim]")
                continue

            urls_seen.add(url)
            content = extract_fn(url)

            if hasattr(content, 'success'):
                # Real ExtractedContent object
                if content.success:
                    source = SourceDocument(url=url, text=content.text)
                    all_sources.append(source)
                    all_source_texts.append(content.text)
                    new_sources_this_round += 1
                    console.print(f"  [green]✓[/green] {title} ({len(content.text)} chars)")
                else:
                    console.print(f"  [red]✗[/red] {title}: {content.error}")
            else:
                # Mock/test content (might be a string or dict)
                text = content if isinstance(content, str) else str(content)
                if text:
                    source = SourceDocument(url=url, text=text)
                    all_sources.append(source)
                    all_source_texts.append(text)
                    new_sources_this_round += 1

        console.print(f"[green]  → {new_sources_this_round} new sources extracted[/green]")

        # ---- DECIDE: Do we have enough? ----
        # If we still have pending sub-queries from the planner, use those
        # instead of asking the LLM. The planner already determined these
        # are needed, so we process them first.
        if pending_queries:
            current_query = pending_queries.pop(0)
            console.print(f"[magenta]📋 Next planned sub-query: {current_query}[/magenta]")
            continue

        # All planned sub-queries processed — now ask the LLM
        console.print(f"[blue]🤔 Evaluating gathered information...[/blue]")
        decision = decision_fn(question, queries_used, all_source_texts)

        console.print(f"  [dim]Reasoning: {decision.reasoning}[/dim]")

        if decision.enough_info:
            console.print(f"[bold green]✓ Agent decided: enough information gathered.[/bold green]")
            break

        if decision.next_query:
            current_query = decision.next_query
            console.print(f"[blue]→ Next query: {current_query}[/blue]")
        else:
            console.print("[yellow]⚠ Agent says not enough info but has no next query. Stopping.[/yellow]")
            break

        # ---- CAP CHECK ----
        if iteration == max_iterations:
            hit_cap = True
            console.print(f"[bold yellow]⚠ Hit iteration cap ({max_iterations}). Stopping loop.[/bold yellow]")

    # ---- SYNTHESIZE ----
    console.print(f"\n[bold blue]🧠 Synthesizing final answer from {len(all_sources)} sources...[/bold blue]\n")

    if all_source_texts:
        answer = synthesize_fn(question, all_source_texts)
    else:
        answer = "Unable to find sufficient information to answer this question."

    return AgentResult(
        question=question,
        answer=answer,
        sources=all_sources,
        queries_used=queries_used,
        iterations=len(queries_used),  # actual iterations completed
        hit_cap=hit_cap,
    )
