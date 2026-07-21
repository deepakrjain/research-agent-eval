"""
Shared configuration — loads environment variables and provides
constants used across the agent.

WHY THIS EXISTS:
Rather than scattering dotenv.load_dotenv() and os.getenv() calls
across every module, we centralize config here. This means:
1. One place to change default values (model names, timeouts, caps)
2. One place where missing keys fail loudly at startup, not halfway
   through a 5-minute benchmark run
3. Easy to swap models for testing (change one line, not ten)
"""

import os
from dotenv import load_dotenv

# Load .env file from project root (no-op if .env doesn't exist)
load_dotenv()


def get_groq_api_key() -> str:
    """Return the Groq API key, failing loudly if not set."""
    key = os.getenv("GROQ_API_KEY")
    if not key or key == "your_groq_api_key_here":
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key.\n"
            "Get a free key at: https://console.groq.com/keys"
        )
    return key


def get_gemini_api_key() -> str:
    """Return the Gemini API key, failing loudly if not set."""
    key = os.getenv("GEMINI_API_KEY")
    if not key or key == "your_gemini_api_key_here":
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )
    return key


# ---- Model defaults ----
# Groq free tier supports these open-weight models.
# We use llama-3.3-70b-versatile for best quality on free tier.
GROQ_MODEL = "llama-3.3-70b-versatile"

# ---- Agent defaults ----
MAX_SEARCH_RESULTS = 5       # results per DuckDuckGo query
MAX_LOOP_ITERATIONS = 5      # hard cap on search-read-decide cycles
PAGE_FETCH_TIMEOUT = 10      # seconds before giving up on a page
MAX_CONTENT_LENGTH = 3000    # chars of extracted text to keep per page
