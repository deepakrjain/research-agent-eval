"""
test_extraction.py — Tests for extractor.py's failure handling.

WHY THESE TESTS MATTER:
The extractor is the most failure-prone component — it talks to the
real internet, where anything can go wrong (timeouts, 404s, non-HTML
content, pages that are 99% JavaScript, empty pages). We need to verify
that NONE of these failure modes crash the agent. Every failure should
be captured in the ExtractedContent.success field, never as an
unhandled exception.

TESTING STRATEGY:
We mock `requests.get` to simulate various failure conditions without
actually hitting the network. This makes tests:
- Fast (no HTTP calls)
- Deterministic (same result every time)
- Independent (no network dependency in CI)
"""

import pytest
from unittest.mock import patch, Mock
from requests.exceptions import Timeout, ConnectionError

from agent.extractor import extract_content, _extract_text_from_html, ExtractedContent


class TestExtractContentFailures:
    """Test that extract_content handles all failure modes gracefully."""

    @patch("agent.extractor.requests.get")
    def test_timeout_returns_failure(self, mock_get):
        """A page that times out should return success=False, not crash."""
        mock_get.side_effect = Timeout("Connection timed out")

        result = extract_content("https://example.com/slow-page")

        assert isinstance(result, ExtractedContent)
        assert result.success is False
        assert "Timeout" in result.error
        assert result.text == ""

    @patch("agent.extractor.requests.get")
    def test_connection_error_returns_failure(self, mock_get):
        """A page that can't be reached should return success=False."""
        mock_get.side_effect = ConnectionError("DNS resolution failed")

        result = extract_content("https://nonexistent.example.com")

        assert result.success is False
        assert "Fetch failed" in result.error

    @patch("agent.extractor.requests.get")
    def test_404_returns_failure(self, mock_get):
        """A 404 page should return success=False."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")

        mock_get.return_value = mock_response

        result = extract_content("https://example.com/not-found")

        assert result.success is False

    @patch("agent.extractor.requests.get")
    def test_non_html_content_returns_failure(self, mock_get):
        """A PDF or image should return success=False (we can't parse it)."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {"Content-Type": "application/pdf"}

        mock_get.return_value = mock_response

        result = extract_content("https://example.com/paper.pdf")

        assert result.success is False
        assert "Non-HTML" in result.error

    @patch("agent.extractor.requests.get")
    def test_empty_page_returns_failure(self, mock_get):
        """A page with almost no text content should return success=False."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {"Content-Type": "text/html"}
        # Page with only a tiny bit of text — not useful
        mock_response.text = "<html><body><p>OK</p></body></html>"

        mock_get.return_value = mock_response

        result = extract_content("https://example.com/empty")

        assert result.success is False
        assert "too short" in result.error

    @patch("agent.extractor.requests.get")
    def test_successful_extraction(self, mock_get):
        """A normal HTML page should return success=True with extracted text."""
        article_text = "This is a substantial article about Python programming. " * 10
        html = f"""
        <html>
        <head><title>Test Page</title></head>
        <body>
            <nav><a href="/">Home</a></nav>
            <article><p>{article_text}</p></article>
            <footer>Copyright 2024</footer>
        </body>
        </html>
        """
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_response.text = html

        mock_get.return_value = mock_response

        result = extract_content("https://example.com/article")

        assert result.success is True
        assert len(result.text) > 50
        assert result.url == "https://example.com/article"
        assert result.error == ""

    @patch("agent.extractor.requests.get")
    def test_junk_tags_are_stripped(self, mock_get):
        """Script, style, nav, and footer content should be removed."""
        html = """
        <html>
        <body>
            <script>var x = "malicious";</script>
            <style>.hidden { display: none; }</style>
            <nav><a href="/">Home</a><a href="/about">About</a></nav>
            <article>
                <p>This is the real content that should be extracted from the page and kept intact for analysis.</p>
            </article>
            <footer>Copyright 2024 All Rights Reserved</footer>
        </body>
        </html>
        """
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text = html

        mock_get.return_value = mock_response

        result = extract_content("https://example.com/article")

        assert result.success is True
        assert "malicious" not in result.text
        assert "display: none" not in result.text
        assert "real content" in result.text


class TestExtractTextFromHtml:
    """Test the internal HTML parsing function directly."""

    def test_removes_script_tags(self):
        html = "<html><body><script>alert('xss')</script><p>Real content here for testing purposes and extraction.</p></body></html>"
        text = _extract_text_from_html(html)
        assert "alert" not in text
        assert "Real content" in text

    def test_removes_navigation(self):
        html = """
        <html><body>
            <nav><ul><li>Home</li><li>About</li></ul></nav>
            <main><p>Article content that should survive extraction and be returned to the caller.</p></main>
        </body></html>
        """
        text = _extract_text_from_html(html)
        assert "Article content" in text
        # Nav content should be stripped
        assert "Home" not in text

    def test_handles_empty_html(self):
        """Empty/minimal HTML should return empty or near-empty string."""
        text = _extract_text_from_html("<html><body></body></html>")
        assert len(text.strip()) == 0

    def test_preserves_paragraph_separation(self):
        html = "<html><body><p>First paragraph.</p><p>Second paragraph.</p></body></html>"
        text = _extract_text_from_html(html)
        # Both paragraphs should be present
        assert "First paragraph" in text
        assert "Second paragraph" in text
