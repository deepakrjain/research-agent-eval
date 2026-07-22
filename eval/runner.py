"""
runner.py — Runs the research agent over the benchmark dataset and saves the results.

WHY WE NEED A BATCH RUNNER:
To evaluate an agent, we can't just test it manually one question at a time.
We need to run it against a standardized set of questions (the benchmark)
and record exactly what it did (iterations, queries, sources, final answer).
These results will later be evaluated by our LLM judge in Phase 6.

We save the results as JSON lines (.jsonl) in the `results/` folder, ensuring
we never overwrite past runs, allowing us to compare performance over time.
"""

import os
import json
import time
from datetime import datetime
from rich.console import Console

from agent.loop import run_agent_loop

console = Console()

def run_benchmark(dataset_path: str = "eval/dataset.json", results_dir: str = "results"):
    """
    Run the agent against all questions in the dataset and save the output.
    """
    # Load the benchmark dataset
    if not os.path.exists(dataset_path):
        console.print(f"[red]Dataset not found at {dataset_path}[/red]")
        return
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a unique filename for this run based on timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_dir, f"run_{timestamp}.jsonl")
    
    console.print(f"[bold magenta]🚀 Starting benchmark run on {len(dataset)} questions...[/bold magenta]")
    console.print(f"Results will be saved to: {output_path}")

    # We open in append mode so we write each result as it finishes.
    # This ensures that if the script crashes halfway through, we don't lose the results.
    with open(output_path, "a", encoding="utf-8") as out_file:
        for i, item in enumerate(dataset, 1):
            qid = item["id"]
            question = item["question"]
            ref_answer = item["reference_answer"]
            q_type = item["type"]
            
            console.print(f"\n[bold cyan]━━━ Question {i}/{len(dataset)} ({qid}) ━━━[/bold cyan]")
            console.print(f"[bold]Q:[/bold] {question}")
            
            start_time = time.time()
            
            try:
                # Run the agent (cap at 5 iterations)
                result = run_agent_loop(question=question, max_iterations=5)
                
                # Format the output record
                record = {
                    "id": qid,
                    "type": q_type,
                    "question": question,
                    "reference_answer": ref_answer,
                    "agent_answer": result.answer,
                    "queries_used": result.queries_used,
                    "iterations": result.iterations,
                    "hit_cap": result.hit_cap,
                    "sources": [{"url": s.url, "length": len(s.text)} for s in result.sources],
                    "time_seconds": round(time.time() - start_time, 2)
                }
                
                # Write to jsonl
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()
                
                console.print(f"[green]✓ Answered in {record['time_seconds']}s using {record['iterations']} iterations.[/green]")
                
            except Exception as e:
                console.print(f"[red]✗ Error processing {qid}: {e}[/red]")
                
                # Record the failure
                record = {
                    "id": qid,
                    "type": q_type,
                    "question": question,
                    "reference_answer": ref_answer,
                    "agent_answer": f"ERROR: {str(e)}",
                    "error": True
                }
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()
                
    console.print(f"\n[bold green]✅ Benchmark run complete! Results saved to {output_path}[/bold green]")

if __name__ == "__main__":
    # If run directly, execute the benchmark
    run_benchmark()
