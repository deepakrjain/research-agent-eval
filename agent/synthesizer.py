"""
synthesizer.py — Takes a question + extracted source texts and produces
a plain-text answer using Groq (Llama 3).

PHASE 1 VERSION (minimal):
- No citations yet (that's Phase 4)
- No structured output validation yet (no Pydantic model for the answer)
- Just: question + context → LLM → plain text answer

WHY THIS IS INTENTIONALLY SIMPLE:
We're building bottom-up. This synthesizer "works" but has no way to
verify its claims link back to sources. Phase 4 will upgrade it to
produce cited answers with validated source references. Starting simple
lets us get the pipeline running end-to-end before adding complexity.

WHY GROQ / LLAMA 3:
Groq's free tier gives us access to Llama 3.3 70B — a strong open-weight
model with fast inference. We're not paying for this, and the quality is
competitive with proprietary models for straightforward synthesis tasks.
"""

from groq import Groq
from agent.config import get_groq_api_key, GROQ_MODEL


# System prompt that frames the model's role.
# WHY A SYSTEM PROMPT:
# Without explicit framing, the model might:
# - Answer from its training data instead of the provided sources
# - Add disclaimers like "As an AI, I can't browse the web..."
# - Produce overly verbose, padded responses
# The system prompt constrains it to act as a research synthesizer.
SYSTEM_PROMPT = """You are a research assistant that synthesizes information from provided sources to answer questions.

Rules:
1. Answer ONLY based on the provided source texts. Do not use your training knowledge.
2. If the sources don't contain enough information to fully answer the question, say so explicitly.
3. Be concise and direct. Aim for 2-4 paragraphs.
4. Do not include phrases like "Based on the sources" or "According to the provided text" — just answer naturally.
5. If sources contradict each other, note the disagreement."""


def synthesize(question: str, source_texts: list[str]) -> str:
    """
    Given a question and a list of extracted source texts, produce a
    synthesized answer using Groq's LLM.

    Args:
        question: The user's original question.
        source_texts: List of extracted text strings from web pages.

    Returns:
        A plain text answer string.

    WHY WE JOIN SOURCES WITH SEPARATORS:
    If we just concatenated all source texts, the model couldn't tell
    where one source ends and another begins. Numbered separators make
    it clear, and in Phase 4 we'll use these numbers for citations.
    """
    if not source_texts:
        return "No source information was available to answer this question."

    # Format sources with clear separators
    formatted_sources = []
    for i, text in enumerate(source_texts, 1):
        formatted_sources.append(f"--- Source {i} ---\n{text}")
    sources_block = "\n\n".join(formatted_sources)

    user_message = f"""Question: {question}

Sources:
{sources_block}

Please synthesize a clear, accurate answer to the question using only the information from these sources."""

    client = Groq(api_key=get_groq_api_key())

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,  # Low temperature for factual synthesis
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()
