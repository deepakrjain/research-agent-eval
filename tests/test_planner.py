"""
test_planner.py — Tests for query decomposition.

WHY THESE TESTS MATTER:
The planner is the first step in the agent loop. If it produces 0
sub-queries, the loop has nothing to search. If it produces 20,
we've burned most of our iteration budget on planned queries before
the decide loop can even run.

WHAT WE TEST:
1. Decomposition produces 1-6 sub-queries (within validator bounds)
2. Sub-queries are non-empty strings
3. The planner integrates correctly into the loop (sub-queries are
   actually searched before the decide step kicks in)
4. Planner failure falls back gracefully

NOTE: We test the Pydantic model's validator directly, and test the
full planner-to-loop integration with injected fakes.
"""

import pytest
from agent.models import QueryPlan
from agent.loop import run_agent_loop
from agent.models import SearchDecision
from agent.searcher import SearchResult
from agent.extractor import ExtractedContent
from tests.conftest import make_fake_planner


class TestQueryPlanModel:
    """Test the QueryPlan Pydantic model's validation."""

    def test_valid_plan_with_two_queries(self):
        """Minimum valid plan: 2 sub-queries."""
        plan = QueryPlan(
            sub_queries=["query one", "query two"],
            reasoning="Splitting into two aspects.",
        )
        assert len(plan.sub_queries) == 2

    def test_valid_plan_with_four_queries(self):
        """Maximum target: 4 sub-queries."""
        plan = QueryPlan(
            sub_queries=["q1", "q2", "q3", "q4"],
            reasoning="Four aspects.",
        )
        assert len(plan.sub_queries) == 4

    def test_rejects_empty_sub_queries(self):
        """0 sub-queries should raise a validation error."""
        with pytest.raises(Exception):
            QueryPlan(
                sub_queries=[],
                reasoning="No queries.",
            )

    def test_truncates_excessive_sub_queries(self):
        """More than 6 sub-queries should be truncated to 6."""
        plan = QueryPlan(
            sub_queries=[f"query {i}" for i in range(10)],
            reasoning="Too many.",
        )
        assert len(plan.sub_queries) <= 6

    def test_single_query_is_valid(self):
        """1 sub-query should be valid (fallback case)."""
        plan = QueryPlan(
            sub_queries=["single query"],
            reasoning="Simple question.",
        )
        assert len(plan.sub_queries) == 1


class TestPlannerLoopIntegration:
    """Test that the planner integrates correctly into the loop."""

    def test_planner_sub_queries_are_searched(self):
        """
        When the planner returns 3 sub-queries, the loop should search
        all 3 before asking the decide LLM.
        """
        searched_queries = []

        def fake_search(query, max_results=5):
            searched_queries.append(query)
            return [SearchResult(
                title=f"Result for {query}",
                url=f"https://example.com/{query.replace(' ', '-')}",
                snippet="Snippet",
            )]

        def fake_extract(url):
            return ExtractedContent(
                url=url,
                text="Substantial extracted content that is long enough for testing and validation purposes throughout.",
                success=True,
            )

        def fake_decision(question, queries, texts):
            # After processing all sub-queries, say "enough"
            return SearchDecision(
                enough_info=True,
                reasoning="Got info from all sub-queries.",
                next_query=None,
            )

        def fake_synthesize(question, sources):
            return "Synthesized answer."

        planned_queries = ["aspect one", "aspect two", "aspect three"]

        result = run_agent_loop(
            question="Complex question?",
            max_iterations=5,
            _decision_fn=fake_decision,
            _search_fn=fake_search,
            _extract_fn=fake_extract,
            _synthesize_fn=fake_synthesize,
            _planner_fn=make_fake_planner(planned_queries),
        )

        # All 3 planned sub-queries should have been searched
        assert "aspect one" in searched_queries
        assert "aspect two" in searched_queries
        assert "aspect three" in searched_queries
        # Total queries used should include all planned queries
        assert len(result.queries_used) >= 3

    def test_planner_fallback_uses_original_question(self):
        """
        When the planner returns just the original question (fallback),
        the loop should still work normally.
        """
        def fake_search(query, max_results=5):
            return [SearchResult(
                title="Result",
                url="https://example.com/result",
                snippet="Snippet",
            )]

        def fake_extract(url):
            return ExtractedContent(
                url=url,
                text="Content that answers the question with enough detail for the agent to produce a good answer.",
                success=True,
            )

        def fake_decision(question, queries, texts):
            return SearchDecision(
                enough_info=True,
                reasoning="Single query was sufficient.",
                next_query=None,
            )

        def fake_synthesize(question, sources):
            return "Answer."

        # Planner returns just the original question (fallback behavior)
        result = run_agent_loop(
            question="Simple factual question?",
            max_iterations=5,
            _decision_fn=fake_decision,
            _search_fn=fake_search,
            _extract_fn=fake_extract,
            _synthesize_fn=fake_synthesize,
            _planner_fn=make_fake_planner(),  # defaults to [question]
        )

        assert result.iterations >= 1
        assert "Simple factual question?" in result.queries_used
