"""
test_synthesizer.py — Tests for the cited synthesis module.

WHY THESE TESTS MATTER:
Cited synthesis is prone to two major LLM failure modes:
1. Returning raw text instead of JSON
2. Hallucinating citations (e.g. citing [3] when there are only 2 sources)

We test that the SynthesizedAnswer model catches these issues and that
the synthesize() function's retry loop works as expected.
"""

import pytest
from pydantic import ValidationError
from agent.models import SynthesizedAnswer, SourceDocument
from agent.synthesizer import synthesize


class TestSynthesizedAnswerModel:
    """Test the Pydantic model for the synthesizer output."""

    def test_valid_synthesized_answer(self):
        answer = SynthesizedAnswer(
            answer_text="The sky is blue [1].",
            citations_used=[1]
        )
        assert answer.answer_text == "The sky is blue [1]."
        assert answer.citations_used == [1]

    def test_missing_fields_raises_validation_error(self):
        with pytest.raises(ValidationError):
            SynthesizedAnswer(
                answer_text="No citations array here."
            )


class TestSynthesizeIntegration:
    """Test the synthesize loop and retry behavior using a fake Groq client."""

    def test_synthesize_with_no_sources(self):
        """Should return a fallback string immediately."""
        result = synthesize("Why is the sky blue?", [])
        assert "No source information" in result

    def test_successful_synthesis(self):
        """
        If the model returns valid JSON and valid citations on the first try,
        it should return the formatted string with the source list.
        """
        class FakeChoice:
            class FakeMessage:
                content = '{"answer_text": "Python is fast [1].", "citations_used": [1]}'
            message = FakeMessage()

        class FakeCompletions:
            def create(self, **kwargs):
                class FakeResponse:
                    choices = [FakeChoice()]
                return FakeResponse()

        class FakeGroq:
            class FakeChat:
                completions = FakeCompletions()
            chat = FakeChat()

        sources = [SourceDocument(url="https://python.org", text="Python is fast.")]
        result = synthesize("Is Python fast?", sources, groq_client=FakeGroq())

        # The answer text should be present
        assert "Python is fast [1]." in result
        # The source URL should be appended
        assert "1. https://python.org" in result

    def test_retry_on_invalid_citations(self):
        """
        If the model hallucinates a citation (e.g. [2] when there's only 1 source),
        the retry loop should catch it and prompt again.
        """
        call_count = 0

        class FakeCompletions:
            def create(self, **kwargs):
                nonlocal call_count
                call_count += 1
                class FakeChoice:
                    class FakeMessage:
                        pass
                    message = FakeMessage()
                choice = FakeChoice()
                if call_count == 1:
                    # First attempt: cites source [2] which doesn't exist
                    choice.message.content = '{"answer_text": "Python [2].", "citations_used": [2]}'
                else:
                    # Second attempt: fixes the citation to [1]
                    choice.message.content = '{"answer_text": "Python [1].", "citations_used": [1]}'
                
                class FakeResponse:
                    choices = [choice]
                return FakeResponse()

        class FakeGroq:
            class FakeChat:
                completions = FakeCompletions()
            chat = FakeChat()

        sources = [SourceDocument(url="https://python.org", text="Python is fast.")]
        result = synthesize("Is Python fast?", sources, groq_client=FakeGroq())

        # The client should have been called twice due to the retry
        assert call_count == 2
        # The final result should contain the corrected text
        assert "Python [1]." in result
        assert "1. https://python.org" in result

    def test_proceeds_after_max_retries_fail(self):
        """
        If the model fails to fix citations after max_retries, it should
        just proceed with whatever it has, filtering out the invalid citations
        from the reference list.
        """
        call_count = 0

        class FakeCompletions:
            def create(self, **kwargs):
                nonlocal call_count
                call_count += 1
                class FakeChoice:
                    class FakeMessage:
                        content = '{"answer_text": "Always wrong [2].", "citations_used": [2]}'
                    message = FakeMessage()
                
                class FakeResponse:
                    choices = [FakeChoice()]
                return FakeResponse()

        class FakeGroq:
            class FakeChat:
                completions = FakeCompletions()
            chat = FakeChat()

        sources = [SourceDocument(url="https://python.org", text="Python is fast.")]
        result = synthesize("Is Python fast?", sources, groq_client=FakeGroq())

        # 1 initial call + 2 retries = 3 calls
        assert call_count == 3
        # It should still output the text
        assert "Always wrong [2]." in result
        # But it should say no specific sources were cited (since 2 was invalid)
        assert "(No specific sources were cited in the answer.)" in result
