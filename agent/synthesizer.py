"""
synthesizer.py — Takes a question + extracted source texts and produces
a cited answer using Groq (Llama 3).

PHASE 4 VERSION (Cited Synthesis):
- The model outputs a structured JSON response (SynthesizedAnswer).
- Every claim must map to a specific numbered source ([1], [2]).
- We validate that all citation numbers used actually exist.
- If the model hallucinated a citation, we catch it and regenerate.
- The final output is a markdown string with the source list appended.

WHY CITED SYNTHESIS IS HARD:
LLMs naturally want to write fluent text, often blending information
from multiple sources seamlessly. Forcing them to attribute specific
claims to specific sources is an unnatural constraint. Furthermore,
they are prone to "hallucinating" citations (e.g., citing [5] when
only 3 sources were provided). By using structured output and explicit
validation, we force the model to ground its claims and catch it
when it fails.
"""

import json
from groq import Groq
from rich.console import Console
from agent.config import get_groq_api_key, GROQ_MODEL
from agent.models import SynthesizedAnswer, SourceDocument

console = Console()

# We use JSON mode to ensure the model separates its answer text from
# the list of citations it used.
SYSTEM_PROMPT = """You are a meticulous research assistant that synthesizes information to answer questions.

Rules:
1. Answer ONLY based on the provided source texts. Do not use your training knowledge.
2. Every factual claim MUST be followed by an inline citation to the source it came from, formatted as [1], [2], etc.
3. If multiple sources support a claim, cite them all: [1][2].
4. If the sources don't contain enough information to fully answer the question, say so explicitly.
5. Be concise and direct. Aim for 2-4 paragraphs.
6. Do not include phrases like "Based on the sources" — just answer naturally.
7. Only use citation numbers that correspond to the provided sources.

You MUST respond with a JSON object containing exactly two fields:
{
    "answer_text": "Your complete answer here, with inline citations like [1].",
    "citations_used": [1] // A list of integers representing the source numbers you actually cited
}
"""

def synthesize(
    question: str, 
    sources: list[SourceDocument],
    groq_client: Groq | None = None
) -> str:
    """
    Produce a cited answer from the given sources.

    Includes a retry loop: if the model cites a non-existent source,
    we prompt it to try again (up to 2 retries).

    Args:
        question: The user's original question.
        sources: List of SourceDocument objects.
        groq_client: Optional pre-initialized Groq client.

    Returns:
        A formatted markdown string containing the answer and source list.
    """
    if not sources:
        return "No source information was available to answer this question."

    if groq_client is None:
        groq_client = Groq(api_key=get_groq_api_key())

    # Format sources with clear separators
    formatted_sources = []
    for i, source in enumerate(sources, 1):
        # We pass only the text to the model to save tokens, but we use
        # the URL later for the reference list.
        formatted_sources.append(f"--- Source {i} ---\n{source.text}")
    sources_block = "\n\n".join(formatted_sources)

    user_message = f"""Question: {question}

Sources:
{sources_block}

Please synthesize a clear, accurate, and cited answer to the question."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    max_retries = 2
    valid_citation_nums = set(range(1, len(sources) + 1))
    
    for attempt in range(max_retries + 1):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.2,  # Low temperature for factual synthesis
                max_tokens=1024,
                response_format={"type": "json_object"},
            )

            raw_json = response.choices[0].message.content.strip()
            parsed = json.loads(raw_json)
            synthesized = SynthesizedAnswer(**parsed)

            # Validate citations
            invalid_cites = set(synthesized.citations_used) - valid_citation_nums
            if invalid_cites:
                if attempt < max_retries:
                    console.print(f"[yellow]⚠ Model cited invalid sources {invalid_cites}. Retrying...[/yellow]")
                    # Append the failure to the message history so the model learns
                    messages.append({"role": "assistant", "content": raw_json})
                    messages.append({
                        "role": "user", 
                        "content": f"You cited invalid source numbers: {list(invalid_cites)}. You may only cite sources 1 through {len(sources)}. Please try again."
                    })
                    continue
                else:
                    console.print("[red]⚠ Model failed to fix citations after retries. Proceeding anyway.[/red]")
            
            # Formatting the final output
            final_text = synthesized.answer_text + "\n\n### Sources Used\n"
            
            # Only include sources that were actually cited
            used_nums = sorted(list(set(synthesized.citations_used) & valid_citation_nums))
            if not used_nums:
                final_text += "(No specific sources were cited in the answer.)\n"
            else:
                for num in used_nums:
                    # 1-indexed to 0-indexed mapping
                    source_url = sources[num - 1].url
                    final_text += f"{num}. {source_url}\n"

            return final_text

        except Exception as e:
            if attempt < max_retries:
                console.print(f"[yellow]⚠ Synthesis parsing failed ({e}). Retrying...[/yellow]")
                messages.append({"role": "user", "content": f"Your JSON was invalid or failed to parse: {e}. Please return valid JSON."})
            else:
                console.print(f"[red]⚠ Synthesis completely failed ({e}).[/red]")
                return "An error occurred during synthesis."

    return "Failed to synthesize an answer after multiple attempts."
