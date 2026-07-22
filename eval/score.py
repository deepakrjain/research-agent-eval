"""
score.py — Runs the independent LLM judge over a benchmark run file.

WHY THIS IS SEPARATE FROM RUNNER.PY:
We keep generation (runner.py) and evaluation (score.py) separate so that:
1. If the judge API rate limits, we don't lose the agent's work.
2. We can re-evaluate the same run with a different judge prompt without
   having to re-run the agent (saving time and API calls).
3. We can manually inspect the agent's raw output before grading it.

Usage:
    python -m eval.score path/to/run_TIMESTAMP.jsonl
"""

import sys
import os
import json
from rich.console import Console

from eval.judge import evaluate_answer
from agent.models import EvaluationScore

console = Console()

def score_run(run_file_path: str):
    """
    Score a JSONL results file using the independent LLM judge.
    """
    if not os.path.exists(run_file_path):
        console.print(f"[red]Error: File not found: {run_file_path}[/red]")
        sys.exit(1)

    console.print(f"[bold magenta]⚖️ Scoring run: {run_file_path}[/bold magenta]")

    records = []
    with open(run_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        console.print("[yellow]File is empty.[/yellow]")
        return

    # Statistics
    total_score = 0
    hallucinations = 0
    errors = 0
    scored_records = []

    for i, record in enumerate(records, 1):
        qid = record.get("id", f"Q{i}")
        console.print(f"\n[bold cyan]━━━ Evaluating Question {i}/{len(records)} ({qid}) ━━━[/bold cyan]")
        
        # If the agent crashed on this question, score it as a 1 automatically
        if record.get("error", False):
            console.print("[red]⚠ Agent errored on this question. Scoring 1/5.[/red]")
            score = EvaluationScore(
                score=1, 
                is_hallucinated=False, 
                reasoning="Agent execution error."
            )
        else:
            # Call the LLM judge
            score = evaluate_answer(
                question=record["question"],
                reference_answer=record["reference_answer"],
                agent_answer=record["agent_answer"]
            )
            
            console.print(f"[bold]Score:[/bold] {score.score}/5")
            console.print(f"[bold]Hallucinated:[/bold] {score.is_hallucinated}")
            console.print(f"[dim]Reasoning: {score.reasoning}[/dim]")
            
        total_score += score.score
        if score.is_hallucinated:
            hallucinations += 1
        if score.score == 1 and record.get("error"):
            errors += 1
            
        # Attach the score to the record for saving
        record["judge_score"] = score.score
        record["judge_reasoning"] = score.reasoning
        record["is_hallucinated"] = score.is_hallucinated
        scored_records.append(record)
        
    # Print summary report
    average_score = total_score / len(records)
    perfect_scores = sum(1 for r in scored_records if r["judge_score"] == 5)
    
    console.print("\n[bold green]✅ Evaluation Complete![/bold green]")
    console.print("=" * 40)
    console.print(f"[bold]Average Score:[/bold]    {average_score:.2f} / 5.00")
    console.print(f"[bold]Perfect Scores:[/bold]   {perfect_scores}/{len(records)}")
    console.print(f"[bold]Hallucinations:[/bold]   {hallucinations}/{len(records)}")
    if errors > 0:
        console.print(f"[bold red]Agent Errors:[/bold red]     {errors}/{len(records)}")
    console.print("=" * 40)
    
    # Save the scored results to a new file
    output_path = run_file_path.replace(".jsonl", "_scored.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in scored_records:
            f.write(json.dumps(r) + "\n")
            
    console.print(f"\n[dim]Scored results saved to {output_path}[/dim]")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[red]Usage: python -m eval.score path/to/run_file.jsonl[/red]")
        sys.exit(1)
        
    score_run(sys.argv[1])
