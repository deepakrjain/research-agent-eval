"""
run_agent.py — Simple script to run the Phase 1 pipeline.

This is the "not-yet-agentic" version: question in → one search →
fetch top results → synthesize one answer → done. No decisions,
no loops, no autonomy. It's a fixed pipeline.

Usage:
    python run_agent.py "What is the capital of France?"
    python run_agent.py  (uses a default question for quick testing)
"""

import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from agent.searcher import search
from agent.extractor import extract_content
from agent.synthesizer import synthesize

console = Console()


def run_simple_pipeline(question: str) -> str:
    """
    Run the Phase 1 pipeline: search → extract → synthesize.
    Returns the synthesized answer.
    """
    # Step 1: Search
    console.print(f"\n[bold blue]🔍 Searching for:[/bold blue] {question}\n")
    results = search(question)

    if not results:
        console.print("[bold red]❌ No search results found.[/bold red]")
        return "No search results were found for this question."

    console.print(f"[green]✓ Found {len(results)} results[/green]")
    for i, r in enumerate(results, 1):
        console.print(f"  {i}. {r.title}")
        console.print(f"     [dim]{r.url}[/dim]")

    # Step 2: Extract content from each result
    console.print(f"\n[bold blue]📄 Extracting content from pages...[/bold blue]\n")
    source_texts = []
    source_urls = []

    for r in results:
        content = extract_content(r.url)
        if content.success:
            source_texts.append(content.text)
            source_urls.append(content.url)
            console.print(f"  [green]✓[/green] {r.title} ({len(content.text)} chars)")
        else:
            console.print(f"  [red]✗[/red] {r.title}: {content.error}")

    if not source_texts:
        console.print("[bold red]❌ No content could be extracted.[/bold red]")
        return "Failed to extract content from any search results."

    console.print(f"\n[green]✓ Extracted content from {len(source_texts)}/{len(results)} pages[/green]")

    # Step 3: Synthesize answer
    console.print(f"\n[bold blue]🧠 Synthesizing answer...[/bold blue]\n")
    answer = synthesize(question, source_texts)

    # Display the answer
    console.print(Panel(
        Markdown(answer),
        title="[bold green]Answer[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    # Show sources used
    console.print("\n[bold]Sources used:[/bold]")
    for i, url in enumerate(source_urls, 1):
        console.print(f"  [{i}] {url}")

    return answer


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What are the main differences between Python's asyncio and threading?"

    run_simple_pipeline(question)
