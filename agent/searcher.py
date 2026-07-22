"""
searcher.py — DuckDuckGo search wrapper with retry and backoff.

WHY DUCKDUCKGO:
- No API key required (truly zero-cost, zero-signup)
- The `ddgs` package provides a clean Python interface
- Returns titles, URLs, and snippets — enough to decide which pages to read

WHY THIS IS A SEPARATE MODULE:
Wrapping the search library in our own function gives us:
1. A stable interface — if we swap search providers later, only this file changes
2. A place to add error handling, retries, result filtering
3. Easy to mock in tests (mock one function, not a third-party library)

WHY RETRY WITH EXPONENTIAL BACKOFF:
DuckDuckGo rate-limits automated queries. When you fire 20 questions in rapid
succession (like our benchmark runner does), the API silently returns empty
results. Exponential backoff is the industry-standard solution:
- Attempt 1: fail → wait 1s
- Attempt 2: fail → wait 2s
- Attempt 3: fail → wait 4s
The wait time doubles on each failure, giving the rate-limiter time to reset.
"""

import time
from dataclasses import dataclass
from ddgs import DDGS
from agent.config import MAX_SEARCH_RESULTS

MAX_SEARCH_RETRIES = 3        # max attempts before giving up on a query
BACKOFF_BASE_SECONDS = 1.5    # initial wait time; doubles on each retry


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
    Retries up to MAX_SEARCH_RETRIES times with exponential backoff if the
    result list is empty (a sign of rate-limiting) or an exception occurs.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of SearchResult objects. May be empty if all retries fail.
        Callers must handle the empty case.

    WHY WE CATCH EXCEPTIONS:
    DuckDuckGo's API is unofficial and can intermittently fail (rate
    limiting, network issues, HTML changes). A crash here would kill
    the entire agent loop. Instead, we return an empty list and let
    the loop logic decide what to do (retry with a different query,
    or give up gracefully).
    """
    wait = BACKOFF_BASE_SECONDS
    last_error = ""

    for attempt in range(1, MAX_SEARCH_RETRIES + 1):
        try:
            raw_results = list(DDGS().text(query, max_results=max_results))

            if raw_results:
                # Success — parse and return
                results = []
                for r in raw_results:
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                    ))
                return results

            # Empty result list — likely rate-limited; backoff and retry
            last_error = "empty result set (possible rate limit)"

        except Exception as e:
            last_error = str(e)[:200]

        if attempt < MAX_SEARCH_RETRIES:
            print(f"[searcher] Attempt {attempt}/{MAX_SEARCH_RETRIES} failed ({last_error}). Retrying in {wait:.1f}s...")
            time.sleep(wait)
            wait *= 2  # exponential backoff

    # All retries exhausted
    print(f"[searcher] Search failed for '{query}' after {MAX_SEARCH_RETRIES} attempts: {last_error}")
    return []
