"""
conftest.py — Shared test fixtures and helpers.

WHY THIS FILE:
pytest automatically discovers conftest.py and makes its fixtures
available to all test files in the same directory. By putting the
fake planner here, we avoid duplicating it across every test file.
"""

import pytest
from agent.models import QueryPlan


def make_fake_planner(sub_queries=None):
    """
    Create a fake planner that returns a fixed set of sub-queries.

    By default, returns just the original question as a single
    sub-query — effectively bypassing planning, which is what we
    want when we're testing OTHER parts of the loop (cap, dedup).
    """
    def fake_planner(question, groq_client=None):
        queries = sub_queries if sub_queries is not None else [question]
        return QueryPlan(
            sub_queries=queries,
            reasoning="Test planner — returning fixed sub-queries.",
        )
    return fake_planner
