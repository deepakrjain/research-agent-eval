"""
extractor.py — Fetch a URL and extract clean, readable text.

WHY THIS IS THE HARDEST PART OF WEB SCRAPING:
A web page is 90% boilerplate (nav bars, ads, footers, cookie banners,
JavaScript). The 10% that matters is the article/content body. We need
to extract that signal from the noise, AND handle all the ways a page
can fail to load (timeout, 404, SSL error, non-HTML content like PDFs).

APPROACH:
1. Fetch with requests (fast, no JS rendering needed for most info pages)
2. Parse with BeautifulSoup
3. Strip known junk elements (script, style, nav, footer, ads)
4. Extract text from what remains
5. Truncate to a max length (LLMs have context limits, and most useful
   info is in the first few paragraphs anyway)

WHY NOT PLAYWRIGHT/SELENIUM:
JS-rendered pages (SPAs) won't work with this approach. But for our use
case — informational/reference pages — most render fine as static HTML.
Adding a headless browser would mean: heavier dependencies, slower
fetches, harder CI setup, and more memory. We start simple and would
add a JS fallback only if we see many extraction failures in eval.
"""

import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from agent.config import PAGE_FETCH_TIMEOUT, MAX_CONTENT_LENGTH


@dataclass
class ExtractedContent:
    """
    The result of attempting to extract content from a URL.

    WHY success EXISTS:
    Instead of returning None on failure (which callers might forget to
    check) or raising exceptions (which callers might not catch), we
    return a result object that ALWAYS exists but clearly indicates
    whether extraction worked. The caller checks content.success, not
    "is content None?"
    """
    url: str
    text: str
    success: bool
    error: str = ""


# HTML elements that almost never contain useful article content
JUNK_TAGS = [
    "script", "style", "nav", "footer", "header",
    "aside", "form", "iframe", "noscript",
]

# CSS classes/ids commonly used for non-content elements
JUNK_CLASSES = [
    "sidebar", "menu", "nav", "footer", "header", "ad",
    "advertisement", "cookie", "popup", "modal", "banner",
    "social", "share", "comment", "comments",
]


def extract_content(url: str) -> ExtractedContent:
    """
    Fetch a URL and extract clean text content.

    Returns an ExtractedContent with success=True and extracted text,
    or success=False with an error message explaining what went wrong.
    Never raises exceptions — all failures are captured in the result.

    WHY NEVER RAISE:
    This function is called inside a loop that processes multiple URLs.
    If one page fails, we want to skip it and try the next one, not
    crash the entire agent. Returning a failure result makes this
    "skip and continue" pattern trivial for callers.
    """
    # --- Step 1: Fetch the page ---
    try:
        response = requests.get(
            url,
            timeout=PAGE_FETCH_TIMEOUT,
            headers={
                # WHY FULL BROWSER HEADERS:
                # Many sites use bot-detection middleware that looks at
                # more than just User-Agent. If Accept, Accept-Language,
                # or DNT are missing, the 403 rate goes up dramatically.
                # These headers mimic what Chrome 120 actually sends.
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            },
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return ExtractedContent(
            url=url, text="", success=False,
            error=f"Timeout after {PAGE_FETCH_TIMEOUT}s"
        )
    except requests.exceptions.RequestException as e:
        return ExtractedContent(
            url=url, text="", success=False,
            error=f"Fetch failed: {str(e)[:200]}"
        )
    except Exception as e:
        # Catch-all for unexpected errors (e.g., from mocked raise_for_status)
        return ExtractedContent(
            url=url, text="", success=False,
            error=f"Fetch failed: {str(e)[:200]}"
        )

    # --- Step 2: Check content type ---
    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type and "text/plain" not in content_type:
        return ExtractedContent(
            url=url, text="", success=False,
            error=f"Non-HTML content type: {content_type[:100]}"
        )

    # --- Step 3: Parse and clean ---
    try:
        text = _extract_text_from_html(response.text)
    except Exception as e:
        return ExtractedContent(
            url=url, text="", success=False,
            error=f"Parse error: {str(e)[:200]}"
        )

    # --- Step 4: Validate we got something useful ---
    if len(text.strip()) < 50:
        return ExtractedContent(
            url=url, text="", success=False,
            error="Extracted text too short (< 50 chars) — likely junk or empty page"
        )

    # --- Step 5: Truncate to max length ---
    if len(text) > MAX_CONTENT_LENGTH:
        text = text[:MAX_CONTENT_LENGTH] + "\n\n[... content truncated ...]"

    return ExtractedContent(url=url, text=text, success=True)


def _extract_text_from_html(html: str) -> str:
    """
    Parse HTML and extract readable text, stripping boilerplate.

    WHY THIS IS IMPERFECT (and that's okay):
    No heuristic approach will perfectly separate "article body" from
    "page chrome" on every website. Libraries like readability-lxml or
    trafilatura do better but add dependencies. Our approach is good
    enough for informational pages, and if eval reveals systematic
    extraction failures, we'll swap in a better library — guided by
    data, not premature optimization.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove known junk tags entirely
    for tag_name in JUNK_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove elements with junk-indicating class names or IDs
    # WHY THE hasattr GUARD:
    # soup.find_all(True) can return NavigableString nodes (raw text nodes
    # that have no .get() method) alongside Tag nodes. After calling
    # decompose() on parent elements, their children may become detached
    # NavigableStrings. Without the guard, this causes:
    #   AttributeError: 'NoneType' object has no attribute 'get'
    for element in soup.find_all(True):
        if not hasattr(element, "get"):
            continue
        classes = " ".join(element.get("class", []) or [])
        element_id = element.get("id", "") or ""
        combined = f"{classes} {element_id}".lower()

        if any(junk in combined for junk in JUNK_CLASSES):
            element.decompose()

    # Extract text with newlines between block elements
    text = soup.get_text(separator="\n", strip=True)

    # Clean up excessive whitespace (multiple blank lines → single)
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # remove empty lines
    text = "\n".join(lines)

    return text
