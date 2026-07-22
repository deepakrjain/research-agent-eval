"""
test_dedup.py — Tests for query and URL deduplication in the agent loop.

WHY DEDUP MATTERS:
Without deduplication, the agent can:
1. Search the same query repeatedly (wasting iterations, getting the
   same results each time)
2. Fetch the same URL repeatedly (wasting time and bandwidth, adding
   duplicate text to the context which confuses synthesis)

In production, duplicate work is wasted compute = wasted money. In our
free-tier setup, it wastes limited API quota. Either way, dedup is a
basic guardrail that every production agent needs.

WHAT WE TEST:
- Query dedup: if the decision function suggests a query we've already
  run, the loop should skip it (not search again)
- URL dedup: if a search returns a URL we've already fetched, we should
  not fetch it again
"""

import pytest
from agent.loop import run_agent_loop
from agent.models import SearchDecision
from agent.searcher import SearchResult
from agent.extractor import ExtractedContent
from tests.conftest import make_fake_planner


class TestQueryDedup:
    """Verify the loop never runs the same search query twice."""

    def test_duplicate_query_stops_loop(self):
        """
        If the decision function suggests the SAME query we already ran,
        the loop should detect the duplicate and stop rather than
        searching again with identical terms.
        """
        decision_count = 0

        def fake_decision(question, queries, texts):
            nonlocal decision_count
            decision_count += 1
            # Always suggest the original question as the next query
            # (which was already used as the first query)
            return SearchDecision(
                enough_info=False,
                reasoning="Need more info.",
                next_query=question,  # This is a DUPLICATE of the first query!
            )

        def fake_search(query, max_results=5):
            return [SearchResult(
                title=f"Result for {query}",
                url=f"https://example.com/{hash(query)}",
                snippet="Some snippet",
            )]

        def fake_extract(url):
            return ExtractedContent(
                url=url,
                text="Extracted content that is long enough to pass the minimum length validation check easily.",
                success=True,
            )

        def fake_synthesize(question, sources):
            return "Answer."

        result = run_agent_loop(
            question="Test question?",
            max_iterations=5,
            _decision_fn=fake_decision,
            _search_fn=fake_search,
            _extract_fn=fake_extract,
            _synthesize_fn=fake_synthesize,
            _planner_fn=make_fake_planner(),
        )

        # The query "Test question?" should only appear ONCE in queries_used
        assert result.queries_used.count("Test question?") == 1
        # Loop should have stopped after 1 real iteration (2nd was a dup)
        assert result.iterations <= 2

    def test_all_queries_are_unique(self):
        """Every query in queries_used should be unique — no repeats."""
        call_count = 0

        def fake_decision(question, queries, texts):
            nonlocal call_count
            call_count += 1
            return SearchDecision(
                enough_info=False,
                reasoning="Need more.",
                next_query=f"unique query {call_count + 1}",
            )

        def fake_search(query, max_results=5):
            return [SearchResult(
                title=f"Result",
                url=f"https://example.com/{query.replace(' ', '-')}",
                snippet="Snippet",
            )]

        def fake_extract(url):
            return ExtractedContent(
                url=url,
                text="Extracted page content that contains useful information for answering the research question at hand.",
                success=True,
            )

        def fake_synthesize(question, sources):
            return "Answer."

        result = run_agent_loop(
            question="Original question",
            max_iterations=4,
            _decision_fn=fake_decision,
            _search_fn=fake_search,
            _extract_fn=fake_extract,
            _synthesize_fn=fake_synthesize,
            _planner_fn=make_fake_planner(),
        )

        # All queries should be unique
        assert len(result.queries_used) == len(set(result.queries_used))


class TestURLDedup:
    """Verify the loop never fetches the same URL twice."""

    def test_same_url_from_different_queries_fetched_once(self):
        """
        If two different search queries return the same URL, we should
        only fetch that URL once. The second encounter should be skipped.
        """
        extract_call_urls = []

        def fake_decision(question, queries, texts):
            if len(queries) < 2:
                return SearchDecision(
                    enough_info=False,
                    reasoning="Need more.",
                    next_query="second query",
                )
            return SearchDecision(
                enough_info=True,
                reasoning="Got enough.",
                next_query=None,
            )

        # Both queries return the SAME URL
        def fake_search(query, max_results=5):
            return [SearchResult(
                title="Same Article",
                url="https://example.com/same-article",  # Same URL every time!
                snippet="Snippet",
            )]

        def fake_extract(url):
            extract_call_urls.append(url)
            return ExtractedContent(
                url=url,
                text="Extracted content that is substantial enough to pass the minimum character length validation check.",
                success=True,
            )

        def fake_synthesize(question, sources):
            return "Answer."

        result = run_agent_loop(
            question="Test dedup?",
            max_iterations=5,
            _decision_fn=fake_decision,
            _search_fn=fake_search,
            _extract_fn=fake_extract,
            _synthesize_fn=fake_synthesize,
            _planner_fn=make_fake_planner(),
        )

        # The extract function should only have been called ONCE for this URL
        assert extract_call_urls.count("https://example.com/same-article") == 1
        # Only one unique source in the result
        assert len(result.sources) == 1
