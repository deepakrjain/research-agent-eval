"""
app.py — Interactive Web UI & Dashboard for the Agentic Research System.

Provides:
1. Live Interactive Research Playground (Search-Read-Decide Agent)
2. Benchmark Evaluation & Results Dashboard (Judge Scores, Metrics & Analytics)
4. System Architecture & Technical Breakdown

Run via:
    streamlit run app.py
"""

import os
import json
import time
from datetime import datetime
import pandas as pd
import streamlit as st

from agent.config import get_groq_api_key, GROQ_MODEL, MAX_LOOP_ITERATIONS
from agent.loop import run_agent_loop
from agent.planner import decompose_question
from agent.searcher import search
from agent.extractor import extract_content
from agent.synthesizer import synthesize
from agent.models import SourceDocument, SearchDecision

# Page Configuration
st.set_page_config(
    page_title="Agentic Research System & Benchmarking UI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism & Rich Aesthetics)
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Gradient Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4338ca 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin: 0;
        color: #ffffff;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #c7d2fe;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background: rgba(99, 102, 241, 0.25);
        color: #a5b4fc;
        border: 1px solid rgba(165, 180, 252, 0.3);
        margin-right: 0.5rem;
    }

    /* Metric Cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Code snippet box */
    .code-box {
        background: #0f172a;
        border-radius: 8px;
        padding: 1rem;
        font-family: monospace;
        color: #38bdf8;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# Helper functions
def check_api_keys():
    try:
        key = get_groq_api_key()
        return True, "Groq API Key active"
    except Exception as e:
        return False, str(e)


def load_results_files():
    results_dir = "results"
    if not os.path.exists(results_dir):
        return []
    files = [f for f in os.listdir(results_dir) if f.endswith(".jsonl")]
    files.sort(reverse=True)
    return files


def parse_jsonl_file(filepath):
    records = []
    if not os.path.exists(filepath):
        return records
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


# Render Header
st.markdown("""
<div class="hero-banner">
    <div>
        <span class="badge">Groq Llama 3.3 70B</span>
        <span class="badge">DuckDuckGo Search</span>
        <span class="badge">LLM-as-a-Judge Evaluation</span>
        <span class="badge">Zero Cost Stack</span>
    </div>
    <h1 class="hero-title">Autonomous Research Agent & Eval Benchmark</h1>
    <p class="hero-subtitle">Iterative Search-Read-Decide Loop with Ground-Truth LLM Evaluation</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("⚙️ System Status & Controls")
api_ok, api_msg = check_api_keys()
if api_ok:
    st.sidebar.success(f"✅ {api_msg}")
else:
    st.sidebar.error(f"❌ {api_msg}")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Agent Settings")
max_iters = st.sidebar.slider("Max Loop Iterations", min_value=1, max_value=10, value=MAX_LOOP_ITERATIONS)
st.sidebar.info(f"**Model:** {GROQ_MODEL}\n\n**Search Engine:** DuckDuckGo (Zero key)\n\n**Parser:** Pydantic Guardrails")

st.sidebar.markdown("---")
st.sidebar.caption("Built with Groq, Streamlit & Pydantic")


# Navigation Tabs
tab_playground, tab_eval, tab_arch = st.tabs([
    "🔍 Live Research Playground",
    "📊 Benchmark Eval Dashboard",
    "🏗️ Architecture & Specs"
])


# ==========================================
# TAB 1: LIVE RESEARCH PLAYGROUND
# ==========================================
with tab_playground:
    st.subheader("Interactive Agentic Research")
    st.write("Submit a question to see the research agent plan sub-queries, execute web searches, read source pages, self-evaluate info sufficiency, and synthesize cited answers.")

    sample_questions = [
        "Select a sample question...",
        "What are the main differences between Python's asyncio and threading?",
        "Who won the Nobel Prize in Physics in 2023, and what was it awarded for?",
        "What is the capital of Australia, and what year was it chosen as the capital?",
        "How does DeepSeek-V3 architecture differ from Llama 3?",
        "What is self-preference bias in LLM-as-a-judge evaluations?"
    ]

    selected_sample = st.selectbox("Quick Presets:", sample_questions)
    
    default_q = ""
    if selected_sample != "Select a sample question...":
        default_q = selected_sample

    user_query = st.text_input("Enter your research question:", value=default_q, placeholder="e.g. Compare Rust vs Go for high-concurrency microservices")

    col_btn, col_blank = st.columns([1, 4])
    run_clicked = col_btn.button("🚀 Run Agent", type="primary", use_container_width=True)

    if run_clicked:
        if not user_query.strip():
            st.warning("Please enter a question or select a preset.")
        elif not api_ok:
            st.error("GROQ_API_KEY is missing! Check your .env file.")
        else:
            st.markdown("### 🔄 Execution Progress")
            progress_container = st.container()
            
            with progress_container:
                status_box = st.status("Initializing Agent Loop...", expanded=True)
                
                start_time = time.time()
                
                # Step 1: Decomposition
                status_box.update(label="📋 Step 1: Decomposing question into sub-queries...", state="running")
                plan = decompose_question(user_query)
                st.write("**Planned Sub-Queries:**")
                for sq in plan.sub_queries:
                    st.write(f"- 🔹 {sq}")
                
                # Execute agent loop while displaying logs
                status_box.update(label="🔍 Step 2: Executing Search-Read-Decide Loop...", state="running")
                result = run_agent_loop(question=user_query, max_iterations=max_iters)
                elapsed = round(time.time() - start_time, 2)

                status_box.update(label=f"✅ Research Complete in {elapsed}s!", state="complete", expanded=False)

            st.markdown("---")
            st.markdown("### 📝 Synthesized Answer")
            st.markdown(result.answer)

            st.markdown("### 📊 Metadata & Sources")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Iterations Used", f"{result.iterations} / {max_iters}")
            col_m2.metric("Queries Searched", len(result.queries_used))
            col_m3.metric("Sources Analyzed", len(result.sources))
            col_m4.metric("Time Taken", f"{elapsed}s")

            with st.expander("🔗 View Searched Queries & Source URLs"):
                st.write("**Searched Queries:**")
                for q in result.queries_used:
                    st.write(f"- `{q}`")
                
                st.write("**Sources Extracted:**")
                for i, src in enumerate(result.sources, 1):
                    st.write(f"**[{i}]** [{src.url}]({src.url}) ({len(src.text)} characters extracted)")

            st.success("🎉 Research completed!")


# ==========================================
# TAB 2: BENCHMARK EVAL DASHBOARD
# ==========================================
with tab_eval:
    st.subheader("LLM-as-a-Judge Benchmark Results")
    st.write("Analyze recorded benchmark runs scored by our independent LLM judge (comparing agent answers against ground-truth references).")

    files = load_results_files()
    scored_files = [f for f in files if f.endswith("_scored.jsonl")]

    if not files:
        st.warning("No benchmark result files found in `results/` folder.")
    else:
        selected_file = st.selectbox("Select Benchmark Run File:", scored_files if scored_files else files)
        
        filepath = os.path.join("results", selected_file)
        records = parse_jsonl_file(filepath)

        if not records:
            st.error("Selected file is empty or unparseable.")
        else:
            df = pd.DataFrame(records)
            
            # Compute KPIs
            has_scores = "judge_score" in df.columns
            total_q = len(df)
            avg_time = round(df["time_seconds"].mean(), 2) if "time_seconds" in df.columns else 0
            
            if has_scores:
                avg_score = round(df["judge_score"].mean(), 2)
                perfect_count = (df["judge_score"] == 5).sum()
                accuracy = round((perfect_count / total_q) * 100, 1)
                hallucination_count = df["is_hallucinated"].sum() if "is_hallucinated" in df.columns else 0
                hallucination_rate = round((hallucination_count / total_q) * 100, 1)
            else:
                avg_score, accuracy, hallucination_rate = "N/A", "N/A", "N/A"

            st.markdown("<br>", unsafe_allow_html=True)
            kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
            kpi_col1.metric("Avg Judge Score", f"{avg_score} / 5.0")
            kpi_col2.metric("Accuracy Rate", f"{accuracy}%")
            kpi_col3.metric("Hallucination Rate", f"{hallucination_rate}%")
            kpi_col4.metric("Avg Response Time", f"{avg_time}s")
            kpi_col5.metric("Total Benchmark Qs", total_q)

            st.markdown("---")

            # Score breakdown charts & filters
            if has_scores:
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.markdown("#### Score Distribution (1-5)")
                    score_counts = df["judge_score"].value_counts().sort_index()
                    st.bar_chart(score_counts)
                
                with chart_col2:
                    st.markdown("#### Accuracy by Question Type")
                    if "type" in df.columns:
                        type_avg = df.groupby("type")["judge_score"].mean()
                        st.bar_chart(type_avg)

            st.markdown("---")
            st.markdown("### 📋 Detailed Benchmark Results Table")

            # Filtering options
            col_f1, col_f2 = st.columns(2)
            if "type" in df.columns:
                q_types = ["All"] + list(df["type"].unique())
                sel_type = col_f1.selectbox("Filter by Question Type:", q_types)
                if sel_type != "All":
                    df = df[df["type"] == sel_type]

            if has_scores:
                scores_opt = ["All"] + sorted(list(df["judge_score"].unique()), reverse=True)
                sel_score = col_f2.selectbox("Filter by Judge Score:", scores_opt)
                if sel_score != "All":
                    df = df[df["judge_score"] == sel_score]

            for idx, row in df.iterrows():
                score_badge = f"⭐ {row.get('judge_score', 'N/A')}/5" if has_scores else "Unscored"
                is_hall = "⚠️ Hallucinated" if row.get("is_hallucinated", False) else "✅ Factually Grounded"
                
                q_id = row.get('id', idx)
                q_text = row.get('question', 'Question')
                with st.expander(f"**[{q_id}] {q_text}** — {score_badge} | {is_hall}"):
                    st.write(f"**Question Type:** `{row.get('type', 'general')}` | **Time:** {row.get('time_seconds', 0)}s | **Iterations:** {row.get('iterations', 0)}")
                    
                    st.markdown("#### Agent Answer")
                    st.info(str(row.get("agent_answer", "")))
                    
                    st.markdown("#### Reference Answer (Ground Truth)")
                    st.success(str(row.get("reference_answer", "")))
                    
                    if has_scores and pd.notna(row.get("judge_reasoning")):
                        st.markdown("#### Judge Reasoning")
                        st.write(f"*{row.get('judge_reasoning')}*")
                    
                    queries = row.get("queries_used")
                    if isinstance(queries, (list, tuple)) and len(queries) > 0:
                        st.markdown("**Queries Used:**")
                        st.code("\n".join(f"- {q}" for q in queries))



# ==========================================
# TAB 3: SYSTEM ARCHITECTURE & SPECS
# ==========================================
with tab_arch:
    st.subheader("🏗️ System Architecture & Design Decisions")
    st.write("Understand the design choices that make this agent robust, deterministic, and zero-cost.")

    st.markdown("""
    ### 🔄 6-Phase Agent Pipeline Architecture

    ```
    ┌─────────────────────────┐
    │  User Research Question │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  Phase 3: Query Planner │  <-- Decomposes question into 2-4 sub-queries
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │ Phase 2: Loop (Cap: 5)  │
    │  ┌───────────────────┐  │
    │  │ DuckDuckGo Search │  │
    │  └─────────┬─────────┘  │
    │            ▼            │
    │  ┌───────────────────┐  │
    │  │  BS4 Web Scraper  │  │
    │  └─────────┬─────────┘  │
    │            ▼            │
    │  ┌───────────────────┐  │
    │  │ Decision LLM Call │  <-- ReAct Loop: "Do we have enough information?"
    │  └───────────────────┘  │
    └────────────┬────────────┘
                 │ (enough_info = True OR max iterations)
                 ▼
    ┌─────────────────────────┐
    │ Phase 4: Synthesizer    │  <-- Generates cited Markdown answer with self-correction
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │ Phase 6: LLM Judge Eval │  <-- Scores accuracy (1-5) against ground-truth reference
    └─────────────────────────┘
    ```
    """)

    st.markdown("### 💡 Core Engineering Decisions")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("""
        #### 1. Zero-Cost Infrastructure
        - **Groq Free Tier:** Ultrafast inference on open-weights (Llama 3.3 70B).
        - **DuckDuckGo API:** Requires zero API keys or credit card registration.
        - **Requests + BS4:** Lightweight web extraction without headless browser overhead.
        """)

        st.markdown("""
        #### 2. Deterministic Structured Outputs
        - **Pydantic Validation:** Every LLM response (decision, plan, synthesis, score) validates against a Pydantic schema.
        - **No Regex Parsing:** Eliminates brittleness when LLM outputs vary slightly in phrasing.
        """)

    with col_d2:
        st.markdown("""
        #### 3. Loop Guardrails & Safety
        - **Hard Iteration Cap (5 rounds):** Prevents infinite loop costs.
        - **URL & Query Deduplication:** Never searches or fetches the same target twice.
        - **Safe Fallback:** If decision parsing fails, defaults gracefully to synthesis instead of crashing.
        """)

        st.markdown("""
        #### 4. Objective LLM-as-a-Judge
        - **Independent Judge Model:** Uses a separate model family (Mixtral/Gemini) to eliminate self-preference bias.
        - **Ground-Truth Benchmark:** Evaluates against verified reference answers using quantitative scoring (1-5).
        """)

