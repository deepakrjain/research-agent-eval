"""
test_judge.py — Sanity checks to ensure our LLM judge is trustworthy.

WHY WE TEST THE JUDGE:
"Who watches the watchmen?" If our judge LLM is lazy and just scores
everything a 5/5, our benchmark results are meaningless. We must prove
that the judge can reliably distinguish between a perfect answer and a
completely wrong/hallucinated answer.
"""

import pytest
from eval.judge import evaluate_answer


def make_fake_groq(score, is_hallucinated, reasoning):
    class FakeCompletions:
        def create(self, **kwargs):
            class FakeChoice:
                class FakeMessage:
                    content = f'{{"score": {score}, "is_hallucinated": {"true" if is_hallucinated else "false"}, "reasoning": "{reasoning}"}}'
                message = FakeMessage()
            class FakeResponse:
                choices = [FakeChoice()]
            return FakeResponse()
            
    class FakeGroq:
        class FakeChat:
            completions = FakeCompletions()
        chat = FakeChat()
        
    return FakeGroq()


class TestJudgeTrustworthiness:
    def test_judge_scores_perfect_answer_highly(self):
        question = "What is the capital of France?"
        reference = "The capital of France is Paris."
        agent = "Paris is the capital of France, as it has been for centuries."
        
        # In a real environment, we'd use the real API to prove the prompt works.
        # Here we use a fake to ensure the pipeline and parsing logic is sound.
        fake_client = make_fake_groq(5, False, "Perfect match.")
        
        score = evaluate_answer(
            question=question,
            reference_answer=reference,
            agent_answer=agent,
            groq_client=fake_client
        )
        
        assert score.score == 5
        assert score.is_hallucinated is False

    def test_judge_catches_hallucinations_and_wrong_answers(self):
        question = "What is the capital of France?"
        reference = "The capital of France is Paris."
        agent = "The capital of France is London, which is famous for the Eiffel Tower."
        
        fake_client = make_fake_groq(1, True, "Completely wrong and contradicts reference.")
        
        score = evaluate_answer(
            question=question,
            reference_answer=reference,
            agent_answer=agent,
            groq_client=fake_client
        )
        
        assert score.score <= 2
        assert score.is_hallucinated is True

    def test_judge_scores_partial_answers_in_the_middle(self):
        question = "Who wrote Hamlet and when was it published?"
        reference = "Hamlet was written by William Shakespeare and published in 1603."
        agent = "William Shakespeare wrote the famous play Hamlet."
        
        fake_client = make_fake_groq(3, False, "Missing the date.")
        
        score = evaluate_answer(
            question=question,
            reference_answer=reference,
            agent_answer=agent,
            groq_client=fake_client
        )
        
        assert 2 <= score.score <= 4
        assert score.is_hallucinated is False
