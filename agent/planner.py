"""
planner.py — Decomposes a broad question into focused sub-queries.

WHY PLANNING IS A SEPARATE AGENTIC CAPABILITY:
Consider the question: "How does Python's GIL affect web scraping
performance compared to Go's goroutines?"

Without planning, the agent searches this exact question — a long,
multi-faceted query that search engines handle poorly. It gets back
generic results that touch on some aspects but miss others.

With planning, the agent FIRST decomposes this into focused sub-queries:
  1. "Python GIL impact on I/O bound tasks"
  2. "Go goroutines concurrency model web scraping"
  3. "Python vs Go web scraping performance benchmarks"

Each sub-query is specific enough to return highly relevant results.
Together, they cover the full scope of the question.

THIS IS THE SAME IDEA BEHIND "DEEP AGENT" PATTERNS:
In larger agent architectures (AutoGPT, Devin, etc.), a "planner"
component breaks fuzzy goals into concrete sub-tasks. Our version is
simpler — just query decomposition — but the principle is identical:
breaking a vague objective into specific, actionable steps before
executing any of them.

WHY PLANNING HAPPENS BEFORE THE LOOP, NOT INSIDE IT:
Planning is a one-time upfront cost. If we planned inside the loop,
we'd re-decompose the question every iteration, getting the same
sub-queries each time. By planning once and feeding the sub-queries
into the loop as initial queries, we get the benefit of decomposition
without the waste of repeated planning.
"""

import json
from groq import Groq
from rich.console import Console

from agent.config import get_groq_api_key, GROQ_MODEL
from agent.models import QueryPlan

console = Console()


PLANNER_SYSTEM_PROMPT = """You are a research planning assistant. Given a question, decompose it into 2-4 focused search queries that together would gather enough information to answer the original question comprehensively.

You must respond with a JSON object with exactly these fields:
{
    "sub_queries": ["query 1", "query 2", "query 3"],
    "reasoning": "brief explanation of the decomposition strategy"
}

Rules:
1. Generate exactly 2-4 sub-queries (no more, no less)
2. Each sub-query should target a DIFFERENT aspect of the question
3. Sub-queries should be concise search-engine queries (5-10 words), not full sentences
4. Together, the sub-queries should cover the full scope of the original question
5. Avoid redundant queries that would return the same information
6. For simple factual questions, 2 queries is fine. For complex comparative questions, use 3-4
7. Respond ONLY with the JSON object, no other text"""


def decompose_question(
    question: str,
    groq_client: Groq | None = None,
) -> QueryPlan:
    """
    Decompose a question into 2-4 focused sub-queries.

    Args:
        question: The original user question.
        groq_client: Optional pre-initialized Groq client (for reuse).

    Returns:
        A QueryPlan with sub_queries and reasoning.

    FALLBACK BEHAVIOR:
    If the LLM fails to produce a valid plan, we fall back to using
    the original question as-is. This ensures the agent ALWAYS makes
    progress — a failed planner shouldn't prevent the agent from
    searching at all.
    """
    if groq_client is None:
        groq_client = Groq(api_key=get_groq_api_key())

    user_message = f"""Decompose this question into 2-4 focused search queries:

Question: {question}

Respond with the JSON object."""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,  # Low temp for consistent decomposition
            max_tokens=512,
            response_format={"type": "json_object"},
        )

        raw_json = response.choices[0].message.content.strip()
        parsed = json.loads(raw_json)
        plan = QueryPlan(**parsed)

        console.print(f"[bold green]📋 Query plan ({len(plan.sub_queries)} sub-queries):[/bold green]")
        for i, q in enumerate(plan.sub_queries, 1):
            console.print(f"  {i}. {q}")
        console.print(f"  [dim]Reasoning: {plan.reasoning}[/dim]")

        return plan

    except Exception as e:
        # Fallback: use the original question as the only "sub-query"
        console.print(f"[yellow]⚠ Planning failed ({e}). Using original question as search query.[/yellow]")
        return QueryPlan(
            sub_queries=[question],
            reasoning=f"Planning failed ({e}), falling back to original question.",
        )
