"""
models.py — Pydantic models for all structured data in the agent.

WHY PYDANTIC, NOT DICTS OR REGEX:
LLMs are stochastic — even with JSON mode enabled, the SHAPE of the
response can vary between calls. Consider asking an LLM "do you have
enough information?" It might respond:

    {"enough": true}        ← correct key name?
    {"enough_info": "yes"}  ← string instead of bool?
    {"has_enough": true}    ← different key name?

With a Pydantic model, we define the contract ONCE:
    class SearchDecision(BaseModel):
        enough_info: bool

If the LLM returns {"has_enough": true}, Pydantic raises a
ValidationError with a clear message: "field 'enough_info' is
missing." Compare that to regex parsing, where "yes" matches but
"Yes, I have enough" doesn't, and the failure is silent.

WHY ALL MODELS LIVE HERE:
Centralizing models in one file means:
1. Any developer can see all the "contracts" at a glance
2. No circular imports between modules that share models
3. Easy to audit what the LLM is expected to return
"""

from pydantic import BaseModel, Field, field_validator


class QueryPlan(BaseModel):
    """
    The planner's decomposition of a broad question into sub-queries.

    WHY DECOMPOSE:
    A broad question like "Compare React and Vue for enterprise apps"
    is hard to answer with a single search. The planner breaks it into
    focused sub-queries like:
      - "React enterprise features scalability"
      - "Vue enterprise features scalability"
      - "React vs Vue performance benchmarks"

    Each sub-query targets a specific ASPECT of the question, producing
    more relevant search results than one vague query.

    WHY CONSTRAINED TO 2-4:
    - Less than 2: You're not decomposing at all — defeats the purpose
    - More than 4: Diminishing returns + wastes loop iterations. Each
      sub-query costs a search + extraction round. 4 sub-queries out
      of a 5-iteration cap leaves only 1 iteration for follow-up.
    """
    sub_queries: list[str] = Field(
        description="List of 2-4 focused search queries that together cover all aspects of the original question."
    )
    reasoning: str = Field(
        description="Brief explanation of how the question was decomposed and what each sub-query targets."
    )

    @field_validator("sub_queries")
    @classmethod
    def validate_sub_query_count(cls, v):
        """Enforce the 2-4 sub-query constraint."""
        if len(v) < 1:
            raise ValueError("Must have at least 1 sub-query")
        if len(v) > 6:
            # Hard cap — if the model returns 20, truncate to 6
            # (we use 6 instead of 4 as a soft ceiling to avoid
            # throwing away genuinely useful decompositions that
            # are slightly over the target)
            v = v[:6]
        return v


class SearchDecision(BaseModel):
    """
    The LLM's decision about whether to continue searching.

    Returned after each search-read cycle. This is the core of
    agentic behavior — the model explicitly reasons about its own
    state and decides its next action.

    WHY EACH FIELD EXISTS:
    - enough_info: The binary decision. Must be a bool, not "yes"/"no"
    - reasoning: Forces the model to explain WHY it thinks it has
      enough (or not). This serves two purposes:
      1. Better decisions (chain-of-thought improves LLM reasoning)
      2. Debuggability (we can read the log and see WHY it stopped)
    - next_query: If not enough, what to search next. None means
      the model thinks more searching won't help (dead end).
    """
    enough_info: bool = Field(
        description="Whether the gathered information is sufficient to answer the original question comprehensively."
    )
    reasoning: str = Field(
        description="Brief explanation of why the information is or isn't sufficient, and what's missing if not."
    )
    next_query: str | None = Field(
        default=None,
        description="If enough_info is False, the next search query to run. Should be different from previous queries. None if enough_info is True or if further searching seems futile."
    )


class SynthesizedAnswer(BaseModel):
    """
    The structured answer from the synthesizer.

    WHY THIS EXISTS:
    We need the LLM to provide BOTH the text of the answer AND a structured
    list of the citations it used, so we can validate that the citations
    actually exist in our source list before returning the final result.
    """
    answer_text: str = Field(
        description="The synthesized answer with inline numbered citations like [1] or [1][2]."
    )
    citations_used: list[int] = Field(
        description="A list of source numbers (1-indexed) that were actually cited in the answer_text."
    )


class SourceDocument(BaseModel):
    """
    A single source that the agent has read and extracted content from.

    WHY THIS EXISTS (vs just passing raw strings):
    The synthesizer needs both the text AND the URL (for citations in
    Phase 4). Passing (url, text) tuples is fragile — Pydantic gives
    us a named, validated container.
    """
    url: str = Field(description="The URL this content was extracted from.")
    text: str = Field(description="The extracted text content from the page.")


class AgentResult(BaseModel):
    """
    The complete result of an agent run — returned by the loop.

    Captures not just the answer but the metadata about HOW the agent
    arrived at it: how many iterations, what sources, what queries.
    This metadata is essential for:
    1. Debugging (why was the answer bad?)
    2. Evaluation (did the agent use enough sources?)
    3. Display (show the user what the agent did)
    """
    question: str = Field(description="The original question asked.")
    answer: str = Field(description="The synthesized answer.")
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="All sources read during the research process."
    )
    queries_used: list[str] = Field(
        default_factory=list,
        description="All search queries executed during the research."
    )
    iterations: int = Field(
        default=0,
        description="Number of search-read-decide iterations completed."
    )
    hit_cap: bool = Field(
        default=False,
        description="Whether the loop was stopped by the iteration cap (True) or by the LLM deciding it had enough info (False)."
    )


class EvaluationScore(BaseModel):
    """
    The structured output of the independent LLM judge.
    
    WHY STRUCTURED OUTPUT FOR EVALS:
    If a judge just returns "This is a good answer, 4/5", we have to
    parse that text to extract the 4. By using a Pydantic model, we
    guarantee we get an integer score, a boolean for hallucination,
    and a clear reasoning string, making aggregate statistics trivial
    to compute.
    """
    score: int = Field(
        ge=1, le=5,
        description="Score from 1 (completely wrong/irrelevant) to 5 (perfectly accurate and comprehensive) compared to the reference answer."
    )
    is_hallucinated: bool = Field(
        description="True if the agent made factual claims that contradict or are entirely unsupported by the reference answer."
    )
    reasoning: str = Field(
        description="A brief explanation of why this score was given."
    )
