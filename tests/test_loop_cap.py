"""
test_loop_cap.py — Tests that the agent loop respects its iteration cap.

WHY THIS TEST EXISTS:
An agent loop without a working cap is a ticking time bomb. If the LLM
keeps saying "not enough info," the loop runs forever, burning through
API quota (and money, in a paid setup). This test PROVES the cap works
by injecting a decision function that ALWAYS says "not enough" and
verifying the loop stops anyway.

TESTING STRATEGY:
We use dependency injection (_decision_fn, _search_fn, etc.) to replace
all external dependencies with deterministic fakes. This means:
- No API calls (Groq, DuckDuckGo)
- No network requests (no page fetching)
- Deterministic behavior (same result every time)
- Fast execution (milliseconds, not seconds)
"""

import pytest
from agent.loop import run_agent_loop
from agent.models import SearchDecision, AgentResult
from agent.searcher import SearchResult
from agent.extractor import ExtractedContent


class TestLoopCap:
    """Verify the loop respects its hard iteration cap."""

    def test_loop_stops_at_cap_when_never_enough(self):
        """
        If the decision function ALWAYS says 'not enough info,' the loop
        must still stop at max_iterations. This is the primary safety
        guardrail against infinite loops.
        """
        call_count = 0

        def fake_decision(question, queries, texts):
            nonlocal call_count
            call_count += 1
            return SearchDecision(
                enough_info=False,
                reasoning="Still need more information.",
                next_query=f"query attempt {call_count + 1}",
            )

        def fake_search(query, max_results=5):
            return [SearchResult(
                title=f"Result for {query}",
                url=f"https://example.com/{query.replace(' ', '-')}",
                snippet="Some snippet",
            )]

        def fake_extract(url):
            return ExtractedContent(
                url=url,
                text="This is some extracted content about the topic that is long enough to pass validation checks.",
                success=True,
            )

        def fake_synthesize(question, texts):
            return f"Synthesized answer from {len(texts)} sources."

        cap = 3  # Small cap for fast testing
        result = run_agent_loop(
            question="Test question?",
            max_iterations=cap,
            _decision_fn=fake_decision,
            _search_fn=fake_search,
            _extract_fn=fake_extract,
            _synthesize_fn=fake_synthesize,
        )

        assert isinstance(result, AgentResult)
        # The loop should have run exactly `cap` iterations
        assert result.iterations <= cap
        assert result.hit_cap is True

    def test_loop_stops_early_when_enough(self):
        """
        If the decision function says 'enough' on the first iteration,
        the loop should stop immediately — no unnecessary extra searches.
        """
        def fake_decision(question, queries, texts):
            return SearchDecision(
                enough_info=True,
                reasoning="Got everything we need.",
                next_query=None,
            )

        def fake_search(query, max_results=5):
            return [SearchResult(
                title="Good result",
                url="https://example.com/good",
                snippet="Useful snippet",
            )]

        def fake_extract(url):
            return ExtractedContent(
                url=url,
                text="Comprehensive content that fully answers the question with enough detail for synthesis.",
                success=True,
            )

        def fake_synthesize(question, texts):
            return "Complete answer."

        result = run_agent_loop(
            question="Simple question?",
            max_iterations=5,
            _decision_fn=fake_decision,
            _search_fn=fake_search,
            _extract_fn=fake_extract,
            _synthesize_fn=fake_synthesize,
        )

        assert result.iterations == 1
        assert result.hit_cap is False

    def test_loop_cap_of_one_runs_exactly_once(self):
        """Edge case: cap of 1 should still work (search once, synthesize)."""
        def fake_decision(question, queries, texts):
            return SearchDecision(
                enough_info=False,
                reasoning="Need more.",
                next_query="another query",
            )

        def fake_search(query, max_results=5):
            return [SearchResult(
                title="Result",
                url=f"https://example.com/{hash(query)}",
                snippet="Snippet",
            )]

        def fake_extract(url):
            return ExtractedContent(
                url=url,
                text="Content that is extracted from the page and should be long enough for testing purposes.",
                success=True,
            )

        def fake_synthesize(question, texts):
            return "Answer from limited search."

        result = run_agent_loop(
            question="Edge case?",
            max_iterations=1,
            _decision_fn=fake_decision,
            _search_fn=fake_search,
            _extract_fn=fake_extract,
            _synthesize_fn=fake_synthesize,
        )

        assert result.iterations == 1
        assert result.hit_cap is True
