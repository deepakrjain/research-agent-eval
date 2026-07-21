"""
searcher.py — DuckDuckGo search wrapper.

WHY DUCKDUCKGO:
- No API key required (truly zero-cost, zero-signup)
- The `duckduckgo-search` package provides a clean Python interface
- Returns titles, URLs, and snippets — enough to decide which pages to read

WHY THIS IS A SEPARATE MODULE:
Wrapping the search library in our own function gives us:
1. A stable interface — if we swap search providers later, only this file changes
2. A place to add error handling, retries, result filtering
3. Easy to mock in tests (mock one function, not a third-party library)
"""

from dataclasses import dataclass
from duckduckgo_search import DDGS
from agent.config import MAX_SEARCH_RESULTS


@dataclass
class SearchResult:
    """
    A single search result.

    WHY A DATACLASS (not a dict):
    Dicts are bags of "maybe this key exists" — SearchResult gives us a
    contract: every result WILL have a title, url, and snippet. If the
    search library changes its output format, we catch it here at
    construction time, not deep in the synthesizer when we try to access
    result["snippet"] and get a KeyError.
    """
    title: str
    url: str
    snippet: str


def search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[SearchResult]:
    """
    Run a DuckDuckGo text search and return structured results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of SearchResult objects. May be empty if the search fails
        or returns nothing — callers must handle the empty case.

    WHY WE CATCH EXCEPTIONS:
    DuckDuckGo's API is unofficial and can intermittently fail (rate
    limiting, network issues, HTML changes). A crash here would kill
    the entire agent loop. Instead, we return an empty list and let
    the loop logic decide what to do (retry with a different query,
    or give up gracefully).
    """
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))

        results = []
        for r in raw_results:
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=r.get("body", ""),
            ))
        return results

    except Exception as e:
        # Log the error but don't crash — the agent loop can recover
        print(f"[searcher] Search failed for '{query}': {e}")
        return []
