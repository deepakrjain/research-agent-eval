# Research Agent: Autonomous AI with Self-Evaluation

A fully autonomous, self-correcting AI research agent built from scratch in Python to demonstrate core Agentic AI principles. Rather than being a thin wrapper around a single LLM API call, this agent manages its own state, decomposes complex problems, browses the live internet, detects its own hallucinations, and decides when it has enough information to synthesize a final cited answer.

This project also includes a complete **Evaluation Harness**, using an independent "LLM-as-a-Judge" to scientifically benchmark the agent's accuracy and tool reliability.

---

## 🧠 Core Agentic Architecture

This project was built phase-by-phase to demonstrate how modern AI agents actually work under the hood.

### 1. The ReAct (Reason + Act) Loop
Unlike standard chatbots, this agent uses a dynamic control flow. When given a question:
1. It searches DuckDuckGo for relevant information.
2. It fetches and extracts the text from the top web pages.
3. It passes the gathered knowledge to a **Decision Engine**.
4. The LLM evaluates its own knowledge state: *"Do I have enough facts to answer this?"*
5. If **Yes**, it moves to synthesis. If **No**, it generates a *new* search query to fill the gaps and loops back to step 1.

### 2. Task Decomposition (The Planner)
To prevent the LLM from getting confused by multi-part questions, a **Planner** intercepts the user's prompt before the loop starts. It breaks complex questions (e.g., *"Compare the battery life of iPhone 15 and Galaxy S24"*) into a queue of atomic sub-queries that the loop executes sequentially.

### 3. Cited Synthesis & Self-Correction
Hallucinations are the biggest flaw in LLMs. To combat this, the **Synthesizer** is forced to output structured JSON using Pydantic, containing both the answer text and a strict array of source citation integers `[1, 3]`.
If the LLM hallucinates a citation that doesn't exist, a Python validation layer catches the error, feeds it back to the LLM as a system prompt, and forces it to retry and fix its own mistake.

---

## ⚖️ The Evaluation Harness (LLM-as-a-Judge)

You cannot build a reliable agent on "vibes." Changing a prompt to fix one bug might break three edge cases. This project includes a scientific evaluation pipeline.

*   **The Benchmark (`eval/dataset.json`):** A static set of 20 diverse factual and comparative questions with known reference answers.
*   **The Runner (`eval/runner.py`):** Executes the agent against all 20 questions, logging execution time, iterations, and search queries to a `.jsonl` file.
*   **The Judge (`eval/judge.py`):** Because traditional code tests (`assert answer == "Paris"`) fail for semantic text, we use a separate LLM to grade the agent's answers on a scale of 1-5.
*   **Preventing Self-Preference Bias:** The agent is powered by Meta's **Llama 3** (`llama-3.3-70b-versatile`), but the Judge is powered by Google's **Gemma** or Mistral's **Mixtral** architectures. By using a different model family for the judge, we prevent the AI from artificially giving high scores to text that matches its own stylistic weights.

---

## 🛠️ Engineering for Tool Reliability

During evaluation, we discovered that the agent's intelligence was bottlenecked by its tools. We implemented robust production engineering patterns:
*   **Exponential Backoff Retries:** The `searcher.py` gracefully handles rate limits by backing off and retrying failed searches, preventing catastrophic failures during bulk evaluation runs.
*   **Browser Fingerprinting:** The `extractor.py` sends a complete Chrome 120 HTTP header set (Accept, Accept-Language, DNT) to bypass 403 Forbidden bot-detection middlewares on modern websites.
*   **Resilient HTML Parsing:** BeautifulSoup extraction includes fallback guards for detached DOM nodes to prevent `NoneType` crashes on complex Wikipedia structures.

---

## 🚀 Setup & Execution

### Prerequisites
*   Python 3.10+
*   A free API key from [Groq](https://console.groq.com/)

### Installation
```bash
git clone https://github.com/deepakrjain/research-agent-eval.git
cd research-agent-eval
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=gsk_your_api_key_here
```

### Usage

**1. Run the Interactive Agent**
```bash
python -m agent.run_agent "What is the capital of Australia?"
```

**2. Run the Benchmark Evaluation**
```bash
# Execute the agent against the 20-question dataset (takes ~10 mins)
python -m eval.runner

# Score the results using the LLM Judge
python -m eval.score results/run_YOUR_TIMESTAMP.jsonl
```

---
*Built as a comprehensive demonstration of autonomous agent architecture and evaluation-driven development.*
