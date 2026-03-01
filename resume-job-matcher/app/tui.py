"""
tui.py - Rich Terminal User Interface for Resume-Job Matcher

A beautiful dark-themed terminal UI built with Rich library.
Features:
  - Styled panels and tables for results
  - Progress bars during embedding/search
  - Interactive menus with keyboard navigation
  - Color-coded similarity scores
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.columns import Columns
from rich.rule import Rule
from rich.align import Align
from rich.style import Style
from rich.theme import Theme
from rich import box

# -- Custom dark theme --
CUSTOM_THEME = Theme({
    "title": "bold bright_white",
    "subtitle": "dim white",
    "header": "bold cyan",
    "match_high": "bold green",
    "match_mid": "bold yellow",
    "match_low": "bold red",
    "info": "dim cyan",
    "accent": "bold magenta",
    "muted": "dim white",
    "warning": "bold yellow",
    "error": "bold red",
    "success": "bold green",
})

console = Console(theme=CUSTOM_THEME)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"


# ─── Helpers ────────────────────────────────────────────────────────

def _load_lookup(subdir: str) -> Dict[str, Dict[str, Any]]:
    """Load all JSON files from a data subdirectory into a dict keyed by ID."""
    lookup = {}
    folder = DATA_DIR / subdir
    if folder.exists():
        for fp in sorted(folder.glob("*.json")):
            with open(fp, "r", encoding="utf-8") as f:
                doc = json.load(f)
                doc_id = doc.get("id", "")
                if doc_id:
                    lookup[doc_id] = doc
    return lookup


def _similarity_style(score: float) -> str:
    """Return a Rich style string based on similarity score."""
    if score >= 0.70:
        return "match_high"
    elif score >= 0.50:
        return "match_mid"
    else:
        return "match_low"


def _similarity_bar(score: float, width: int = 20) -> Text:
    """Create a visual bar for similarity score."""
    filled = int(score * width)
    empty = width - filled
    style = _similarity_style(score)
    bar = Text()
    bar.append("█" * filled, style=style)
    bar.append("░" * empty, style="muted")
    bar.append(f" {score*100:.1f}%", style=style)
    return bar


# ─── Banner ─────────────────────────────────────────────────────────

BANNER = r"""
 ╔══════════════════════════════════════════════════════════════╗
 ║   ____                                     _       _       ║
 ║  |  _ \ ___  ___ _   _ _ __ ___   ___     | | ___ | |__    ║
 ║  | |_) / _ \/ __| | | | '_ ` _ \ / _ \    | |/ _ \| '_ \   ║
 ║  |  _ <  __/\__ \ |_| | | | | | |  __/ _  | | (_) | |_) |  ║
 ║  |_| \_\___||___/\__,_|_| |_| |_|\___|| |_| |\___/|_.__/   ║
 ║                                         \___/               ║
 ║           M A T C H E R   ─   P o w e r e d   b y          ║
 ║              E n d e e   V e c t o r   D B                  ║
 ╚══════════════════════════════════════════════════════════════╝
"""


def show_banner():
    """Display the application banner."""
    console.print(Text(BANNER, style="bold cyan"))
    console.print(
        Align.center(
            Text("Semantic AI-Powered Resume ↔ Job Matching", style="title")
        )
    )
    console.print()


# ─── Data browsing panels ───────────────────────────────────────────

def show_resumes_panel():
    """Display all loaded resumes in a table."""
    lookup = _load_lookup("resumes")
    if not lookup:
        console.print("[warning]No resumes found in data/resumes/[/warning]")
        return

    table = Table(
        title="📄 Loaded Resumes",
        box=box.ROUNDED,
        border_style="cyan",
        title_style="header",
        show_lines=True,
    )
    table.add_column("#", style="muted", width=4)
    table.add_column("ID", style="accent", width=12)
    table.add_column("Name", style="title", width=20)
    table.add_column("Title", width=30)
    table.add_column("Location", style="info", width=16)
    table.add_column("Exp", justify="center", width=5)
    table.add_column("Open?", justify="center", width=6)

    for i, (rid, r) in enumerate(sorted(lookup.items()), 1):
        open_icon = "✅" if r.get("is_open_to_work") else "❌"
        table.add_row(
            str(i),
            rid,
            r.get("name", ""),
            r.get("title", ""),
            r.get("location", ""),
            str(r.get("years_experience", "")),
            open_icon,
        )

    console.print(table)


def show_jobs_panel():
    """Display all loaded jobs in a table."""
    lookup = _load_lookup("jobs")
    if not lookup:
        console.print("[warning]No jobs found in data/jobs/[/warning]")
        return

    table = Table(
        title="💼 Loaded Job Postings",
        box=box.ROUNDED,
        border_style="cyan",
        title_style="header",
        show_lines=True,
    )
    table.add_column("#", style="muted", width=4)
    table.add_column("ID", style="accent", width=10)
    table.add_column("Title", style="title", width=30)
    table.add_column("Company", width=22)
    table.add_column("Location", style="info", width=16)
    table.add_column("Min Exp", justify="center", width=8)
    table.add_column("Remote?", justify="center", width=8)
    table.add_column("Salary", style="success", width=22)

    for i, (jid, j) in enumerate(sorted(lookup.items()), 1):
        remote_icon = "✅" if j.get("remote_friendly") else "❌"
        table.add_row(
            str(i),
            jid,
            j.get("title", ""),
            j.get("company", ""),
            j.get("location", ""),
            str(j.get("min_experience", "")),
            remote_icon,
            j.get("salary_range", ""),
        )

    console.print(table)


# ─── Results display ───────────────────────────────────────────────

def show_job_results(results: list, query_label: str = "Query"):
    """
    Display job search results as a Rich table.

    Args:
        results: Endee search results (list of [sim, id, meta, filter, norm, vec])
        query_label: Label describing the query
    """
    lookup = _load_lookup("jobs")

    console.print()
    console.print(Rule(f"[header]Jobs matching: {query_label}[/header]"))
    console.print()

    if not results:
        console.print("[warning]No matching jobs found.[/warning]")
        return

    table = Table(box=box.HEAVY_EDGE, border_style="bright_black", show_lines=True)
    table.add_column("Rank", justify="center", style="muted", width=5)
    table.add_column("Match", width=30)
    table.add_column("Job Title", style="title", width=28)
    table.add_column("Company", width=20)
    table.add_column("Location", style="info", width=14)
    table.add_column("Salary", style="success", width=22)

    for rank, result in enumerate(results, 1):
        sim = result[0]
        rid = result[1]
        doc = lookup.get(rid, {})

        table.add_row(
            f"#{rank}",
            _similarity_bar(sim),
            doc.get("title", rid),
            doc.get("company", "—"),
            doc.get("location", "—"),
            doc.get("salary_range", "—"),
        )

    console.print(table)
    console.print()


def show_candidate_results(results: list, query_label: str = "Query"):
    """
    Display candidate search results as a Rich table.

    Args:
        results: Endee search results
        query_label: Label describing the query
    """
    lookup = _load_lookup("resumes")

    console.print()
    console.print(Rule(f"[header]Candidates matching: {query_label}[/header]"))
    console.print()

    if not results:
        console.print("[warning]No matching candidates found.[/warning]")
        return

    table = Table(box=box.HEAVY_EDGE, border_style="bright_black", show_lines=True)
    table.add_column("Rank", justify="center", style="muted", width=5)
    table.add_column("Match", width=30)
    table.add_column("Name", style="title", width=20)
    table.add_column("Title", width=28)
    table.add_column("Location", style="info", width=14)
    table.add_column("Exp", justify="center", width=5)
    table.add_column("Open?", justify="center", width=6)

    for rank, result in enumerate(results, 1):
        sim = result[0]
        rid = result[1]
        doc = lookup.get(rid, {})
        open_icon = "✅" if doc.get("is_open_to_work") else "❌"

        table.add_row(
            f"#{rank}",
            _similarity_bar(sim),
            doc.get("name", rid),
            doc.get("title", "—"),
            doc.get("location", "—"),
            str(doc.get("years_experience", "—")),
            open_icon,
        )

    console.print(table)
    console.print()


# ─── Search with progress ──────────────────────────────────────────

def search_with_progress(search_fn, description: str = "Searching", **kwargs):
    """
    Run a search function while showing a Rich progress spinner.

    Args:
        search_fn: Callable that performs the search
        description: Text to show during search
        **kwargs: Arguments forwarded to search_fn

    Returns:
        Search results
    """
    with Progress(
        SpinnerColumn("dots", style="cyan"),
        TextColumn("[info]{task.description}[/info]"),
        BarColumn(bar_width=30, complete_style="cyan", finished_style="green"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task(description, total=None)
        results = search_fn(**kwargs)
        progress.update(task, completed=True)

    return results


# ─── Interactive menus ──────────────────────────────────────────────

def pick_resume_interactive() -> Optional[Dict[str, Any]]:
    """Let the user pick a resume from the loaded data."""
    lookup = _load_lookup("resumes")
    if not lookup:
        console.print("[error]No resumes loaded.[/error]")
        return None

    items = sorted(lookup.items())
    for i, (rid, r) in enumerate(items, 1):
        console.print(f"  [accent]{i}[/accent]. {r.get('name', rid)} — {r.get('title', '')}")

    choice = IntPrompt.ask(
        "\n[header]Select a resume number[/header]",
        choices=[str(i) for i in range(1, len(items) + 1)],
    )
    return items[choice - 1][1]


def pick_job_interactive() -> Optional[Dict[str, Any]]:
    """Let the user pick a job from the loaded data."""
    lookup = _load_lookup("jobs")
    if not lookup:
        console.print("[error]No jobs loaded.[/error]")
        return None

    items = sorted(lookup.items())
    for i, (jid, j) in enumerate(items, 1):
        console.print(f"  [accent]{i}[/accent]. {j.get('title', jid)} @ {j.get('company', '')}")

    choice = IntPrompt.ask(
        "\n[header]Select a job number[/header]",
        choices=[str(i) for i in range(1, len(items) + 1)],
    )
    return items[choice - 1][1]


def get_filter_options_job() -> dict:
    """Prompt user for job-search filters."""
    filters = {}
    if Confirm.ask("[info]Apply filters?[/info]", default=False):
        loc = Prompt.ask("[info]Location (Enter to skip)[/info]", default="")
        if loc:
            filters["location"] = loc
        remote = Confirm.ask("[info]Remote-only?[/info]", default=False)
        if remote:
            filters["remote_only"] = True
    return filters


def get_filter_options_candidate() -> dict:
    """Prompt user for candidate-search filters."""
    filters = {}
    if Confirm.ask("[info]Apply filters?[/info]", default=False):
        exp = Prompt.ask("[info]Min years experience (Enter to skip)[/info]", default="")
        if exp.isdigit():
            filters["min_experience"] = int(exp)
        open_only = Confirm.ask("[info]Open-to-work only?[/info]", default=False)
        if open_only:
            filters["open_to_work_only"] = True
    return filters


# ─── Main menu ──────────────────────────────────────────────────────

def main_menu():
    """Run the interactive TUI main menu loop."""
    # Lazy-import heavy modules so the TUI starts fast
    from embedder import embed_text
    from endee_client import EndeeClient
    from match import find_jobs_for_resume, find_candidates_for_job

    show_banner()

    client = EndeeClient()

    # Quick health check
    try:
        indexes = client.list_indexes()
        idx_names = list(indexes.keys()) if isinstance(indexes, dict) else indexes
        console.print(
            Panel(
                f"[success]Connected to Endee[/success]   Indexes: [accent]{', '.join(str(n) for n in idx_names)}[/accent]",
                border_style="green",
                box=box.ROUNDED,
            )
        )
    except Exception as e:
        console.print(
            Panel(
                f"[error]Cannot connect to Endee:[/error] {e}\n\nMake sure the Endee container is running on port 8080.",
                border_style="red",
                box=box.ROUNDED,
            )
        )
        return

    while True:
        console.print()
        console.print(Rule("[header]Main Menu[/header]"))
        console.print()

        menu_items = [
            ("1", "Find jobs for a resume", "Match a candidate to open roles"),
            ("2", "Find candidates for a job", "Match a role to candidates"),
            ("3", "Browse resumes", "View all loaded resumes"),
            ("4", "Browse jobs", "View all loaded job postings"),
            ("5", "Custom text search", "Paste free-form text to search"),
            ("6", "RAG analysis", "Get AI-generated match explanation (requires Ollama)"),
            ("7", "Exit", "Quit the application"),
        ]

        for key, label, desc in menu_items:
            console.print(f"  [accent]{key}[/accent]  {label}  [muted]— {desc}[/muted]")

        console.print()
        choice = Prompt.ask("[header]Choose an option[/header]", choices=[str(i) for i in range(1, 8)])

        # ───────────────────── 1: Jobs for resume ─────────────────────
        if choice == "1":
            console.print(Rule("[header]Find Jobs for a Resume[/header]"))
            resume = pick_resume_interactive()
            if not resume:
                continue
            console.print(
                Panel(
                    f"[title]{resume['name']}[/title] — {resume.get('title', '')}\n"
                    f"[info]{resume.get('location', '')} · {resume.get('years_experience', '?')} yrs[/info]",
                    border_style="cyan",
                    title="Selected Resume",
                )
            )

            filters = get_filter_options_job()

            # Build resume text for embedding
            from ingest_resumes import create_resume_text
            resume_text = create_resume_text(resume)

            try:
                results = search_with_progress(
                    find_jobs_for_resume,
                    description="Embedding & searching jobs...",
                    resume_text=resume_text,
                    k=5,
                    **filters,
                )
                show_job_results(results, query_label=resume.get("name", "resume"))
            except Exception as e:
                console.print(f"[error]Search failed: {e}[/error]")
                console.print("[warning]Try again without filters, or re-ingest data with: python pipeline.py --force[/warning]")

        # ───────────────────── 2: Candidates for job ──────────────────
        elif choice == "2":
            console.print(Rule("[header]Find Candidates for a Job[/header]"))
            job = pick_job_interactive()
            if not job:
                continue
            console.print(
                Panel(
                    f"[title]{job['title']}[/title] @ {job.get('company', '')}\n"
                    f"[info]{job.get('location', '')} · {job.get('min_experience', '?')}+ yrs · "
                    f"{'Remote ✅' if job.get('remote_friendly') else 'On-site'}[/info]",
                    border_style="cyan",
                    title="Selected Job",
                )
            )

            filters = get_filter_options_candidate()

            from ingest_jobs import create_job_text
            job_text = create_job_text(job)

            try:
                results = search_with_progress(
                    find_candidates_for_job,
                    description="Embedding & searching candidates...",
                    job_text=job_text,
                    k=5,
                    **filters,
                )
                show_candidate_results(results, query_label=job.get("title", "job"))
            except Exception as e:
                console.print(f"[error]Search failed: {e}[/error]")
                console.print("[warning]Try again without filters, or re-ingest data with: python pipeline.py --force[/warning]")

        # ───────────────────── 3: Browse resumes ──────────────────────
        elif choice == "3":
            show_resumes_panel()

        # ───────────────────── 4: Browse jobs ─────────────────────────
        elif choice == "4":
            show_jobs_panel()

        # ───────────────────── 5: Free-text search ────────────────────
        elif choice == "5":
            console.print(Rule("[header]Custom Text Search[/header]"))
            search_in = Prompt.ask(
                "[info]Search in[/info]",
                choices=["jobs", "candidates"],
                default="jobs",
            )
            text = Prompt.ask("[info]Enter search text[/info]")
            if not text.strip():
                console.print("[warning]Empty text, skipping.[/warning]")
                continue

            try:
                if search_in == "jobs":
                    results = search_with_progress(
                        find_jobs_for_resume,
                        description="Searching jobs...",
                        resume_text=text.strip(),
                        k=5,
                    )
                    show_job_results(results, query_label=text[:50])
                else:
                    results = search_with_progress(
                        find_candidates_for_job,
                        description="Searching candidates...",
                        job_text=text.strip(),
                        k=5,
                    )
                    show_candidate_results(results, query_label=text[:50])
            except Exception as e:
                console.print(f"[error]Search failed: {e}[/error]")
                console.print("[warning]Make sure Endee is running and data has been ingested.[/warning]")

        # ───────────────────── 6: RAG analysis ────────────────────────
        elif choice == "6":
            try:
                from rag import rag_menu
                rag_menu(console)
            except ImportError:
                console.print("[error]RAG module not found. Make sure rag.py exists.[/error]")
            except Exception as e:
                console.print(f"[error]RAG error: {e}[/error]")

        # ───────────────────── 7: Exit ────────────────────────────────
        elif choice == "7":
            console.print()
            console.print(
                Panel(
                    "[title]Thanks for using Resume-Job Matcher![/title]\n"
                    "[muted]Powered by Endee Vector DB & sentence-transformers[/muted]",
                    border_style="cyan",
                    box=box.DOUBLE,
                )
            )
            break


# ─── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\n[warning]Interrupted. Goodbye![/warning]")
        sys.exit(0)
