"""
run_agent.py — Run the research agent.

Phase 2: Now uses the search-read-decide loop instead of the fixed
single-pass pipeline from Phase 1. The agent decides for itself when
it has enough information.

Usage:
    python run_agent.py "What is the capital of France?"
    python run_agent.py  (uses a default question for quick testing)
"""

import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from agent.loop import run_agent_loop

console = Console()


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What are the main differences between Python's asyncio and threading?"

    console.print(Panel(
        f"[bold]{question}[/bold]",
        title="[bold blue]Research Question[/bold blue]",
        border_style="blue",
    ))

    # Run the agent loop
    result = run_agent_loop(question)

    # Display the answer
    console.print(Panel(
        Markdown(result.answer),
        title="[bold green]Answer[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    # Show metadata
    console.print(f"\n[bold]Loop metadata:[/bold]")
    console.print(f"  Iterations: {result.iterations}")
    console.print(f"  Hit cap: {result.hit_cap}")
    console.print(f"  Queries used: {result.queries_used}")
    console.print(f"  Sources read: {len(result.sources)}")

    console.print(f"\n[bold]Sources:[/bold]")
    for i, source in enumerate(result.sources, 1):
        console.print(f"  [{i}] {source.url}")

    return result


if __name__ == "__main__":
    main()
