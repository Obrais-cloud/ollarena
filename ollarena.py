"""ollarena — Terminal-based model comparison arena for Ollama.

Compare output quality across local models. Run the same prompt against
multiple models in parallel, view results side-by-side, vote on the best,
and track ELO ratings over time.
"""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import requests
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

OLLAMA_BASE = "http://localhost:11434"
DATA_DIR = Path.home() / ".ollarena"
RATINGS_FILE = DATA_DIR / "ratings.json"
HISTORY_FILE = DATA_DIR / "history.json"

console = Console()

# ── ELO ─────────────────────────────────────────────────────────────

DEFAULT_ELO = 1500
K_FACTOR = 32


def expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + math.pow(10, (rb - ra) / 400))


def update_elo(ra: float, rb: float, winner: str) -> tuple[float, float]:
    """Return (new_ra, new_rb). winner is 'a', 'b', or 'tie'."""
    ea = expected_score(ra, rb)
    eb = expected_score(rb, ra)
    if winner == "a":
        sa, sb = 1.0, 0.0
    elif winner == "b":
        sa, sb = 0.0, 1.0
    else:
        sa, sb = 0.5, 0.5
    return ra + K_FACTOR * (sa - ea), rb + K_FACTOR * (sb - eb)


# ── Data persistence ────────────────────────────────────────────────


def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_ratings() -> dict:
    if RATINGS_FILE.exists():
        return json.loads(RATINGS_FILE.read_text())
    return {}


def save_ratings(ratings: dict):
    _ensure_data_dir()
    RATINGS_FILE.write_text(json.dumps(ratings, indent=2))


def load_history() -> list:
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return []


def save_history(history: list):
    _ensure_data_dir()
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


def record_battle(prompt: str, results: list[dict], winner_model: str | None, tie: bool = False):
    ratings = load_ratings()
    history = load_history()

    # Update ELO for each pair
    if len(results) == 2 and not tie and winner_model:
        a, b = results[0], results[1]
        ra = ratings.get(a["model"], DEFAULT_ELO)
        rb = ratings.get(b["model"], DEFAULT_ELO)
        side = "a" if a["model"] == winner_model else "b"
        ra_new, rb_new = update_elo(ra, rb, side)
        ratings[a["model"]] = round(ra_new, 1)
        ratings[b["model"]] = round(rb_new, 1)
    elif tie and len(results) == 2:
        a, b = results[0], results[1]
        ra = ratings.get(a["model"], DEFAULT_ELO)
        rb = ratings.get(b["model"], DEFAULT_ELO)
        ra_new, rb_new = update_elo(ra, rb, "tie")
        ratings[a["model"]] = round(ra_new, 1)
        ratings[b["model"]] = round(rb_new, 1)
    elif winner_model:
        # Multi-model: boost winner, penalize others
        for r in results:
            if r["model"] not in ratings:
                ratings[r["model"]] = DEFAULT_ELO
        winner_elo = ratings[winner_model]
        for r in results:
            if r["model"] != winner_model:
                w, l = update_elo(winner_elo, ratings[r["model"]], "a")
                ratings[winner_model] = round(w, 1)
                ratings[r["model"]] = round(l, 1)
                winner_elo = ratings[winner_model]

    # Ensure all models appear in ratings
    for r in results:
        if r["model"] not in ratings:
            ratings[r["model"]] = DEFAULT_ELO

    save_ratings(ratings)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "models": [r["model"] for r in results],
        "winner": winner_model if not tie else "tie",
        "results": [
            {
                "model": r["model"],
                "tokens": r["gen_tokens"],
                "tok_per_sec": round(r["gen_tok_per_sec"], 1),
                "ttft": round(r["ttft"], 3),
                "total_time": round(r["total_time"], 2),
            }
            for r in results
        ],
    }
    history.append(entry)
    save_history(history)


# ── Ollama API ──────────────────────────────────────────────────────


@dataclass
class GenerationResult:
    model: str
    output: str
    ttft: float
    total_time: float
    prompt_tokens: int
    gen_tokens: int
    gen_tok_per_sec: float
    prompt_tok_per_sec: float
    error: str | None = None


def list_models() -> list[str]:
    r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
    r.raise_for_status()
    models = r.json().get("models", [])
    local = [m["name"] for m in models if m.get("size", 0) > 1_000_000]
    return local or [m["name"] for m in models]


def generate(model: str, prompt: str, system: str = "", max_tokens: int = 1024) -> GenerationResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": max_tokens},
    }
    if system:
        payload["system"] = system

    start = time.perf_counter()
    first_token_time = None
    chunks = []

    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json=payload,
            stream=True,
            timeout=300,
        )
        r.raise_for_status()

        prompt_tokens = 0
        gen_tokens = 0
        prompt_eval_duration = 0
        eval_duration = 0

        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            text = chunk.get("response", "")
            if text:
                chunks.append(text)
                if first_token_time is None:
                    first_token_time = time.perf_counter()

            if chunk.get("done"):
                prompt_tokens = chunk.get("prompt_eval_count", 0)
                gen_tokens = chunk.get("eval_count", 0)
                prompt_eval_duration = chunk.get("prompt_eval_duration", 0)
                eval_duration = chunk.get("eval_duration", 0)
                break

        total_time = time.perf_counter() - start

        if prompt_eval_duration > 0:
            ttft = prompt_eval_duration / 1e9
        elif first_token_time:
            ttft = first_token_time - start
        else:
            ttft = total_time

        gen_tok_per_sec = (gen_tokens / (eval_duration / 1e9)) if eval_duration > 0 else 0
        prompt_tok_per_sec = (prompt_tokens / (prompt_eval_duration / 1e9)) if prompt_eval_duration > 0 else 0

        return GenerationResult(
            model=model,
            output="".join(chunks),
            ttft=ttft,
            total_time=total_time,
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            gen_tok_per_sec=gen_tok_per_sec,
            prompt_tok_per_sec=prompt_tok_per_sec,
        )
    except Exception as e:
        return GenerationResult(
            model=model,
            output="",
            ttft=0,
            total_time=time.perf_counter() - start,
            prompt_tokens=0,
            gen_tokens=0,
            gen_tok_per_sec=0,
            prompt_tok_per_sec=0,
            error=str(e),
        )


# ── Display ─────────────────────────────────────────────────────────


def display_results(results: list[GenerationResult], blind: bool = False):
    panels = []
    for i, r in enumerate(results):
        if r.error:
            title = f"[red]{'Contestant ' + chr(65 + i) if blind else r.model} — ERROR[/red]"
            content = f"[red]{r.error}[/red]"
        else:
            title = f"[bold cyan]{'Contestant ' + chr(65 + i) if blind else r.model}[/bold cyan]"
            stats = (
                f"[dim]{r.gen_tokens} tokens · "
                f"{r.gen_tok_per_sec:.1f} tok/s · "
                f"TTFT {r.ttft:.2f}s · "
                f"Total {r.total_time:.1f}s[/dim]"
            )
            # Truncate very long outputs for display
            output = r.output.strip()
            if len(output) > 2000:
                output = output[:2000] + "\n\n[dim]... (truncated)[/dim]"
            content = f"{output}\n\n{stats}"

        panels.append(
            Panel(
                content,
                title=title,
                border_style="blue",
                expand=True,
                width=console.width // len(results) - 1 if len(results) <= 3 else None,
            )
        )

    if len(results) <= 3:
        console.print(Columns(panels, equal=True, expand=True))
    else:
        for p in panels:
            console.print(p)


def prompt_vote(results: list[GenerationResult], blind: bool = False) -> tuple[str | None, bool]:
    console.print()
    options = []
    for i, r in enumerate(results):
        if not r.error:
            label = chr(65 + i)
            display_name = f"Contestant {label}" if blind else r.model
            options.append((label, r.model, display_name))

    if len(options) < 2:
        console.print("[yellow]Not enough successful responses to vote.[/yellow]")
        return None, False

    vote_str = " / ".join(f"[bold]{label}[/bold]={display}" for label, _, display in options)
    console.print(f"  Vote: {vote_str} / [bold]T[/bold]=tie / [bold]S[/bold]=skip")

    valid = {label.lower() for label, _, _ in options} | {"t", "s"}
    while True:
        try:
            choice = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None, False
        if choice in valid:
            break
        console.print(f"  [red]Invalid. Choose one of: {', '.join(sorted(valid))}[/red]")

    if choice == "s":
        return None, False
    if choice == "t":
        return None, True

    for label, model, _ in options:
        if label.lower() == choice:
            if blind:
                console.print(f"  [green]Winner: {model}[/green]")
            return model, False


# ── Commands ────────────────────────────────────────────────────────


def cmd_battle(args):
    prompt = args.prompt
    system = args.system or ""
    max_tokens = args.max_tokens
    blind = args.blind

    # Read prompt from stdin if "-"
    if prompt == "-":
        prompt = sys.stdin.read().strip()
        if not prompt:
            console.print("[red]No prompt provided via stdin.[/red]")
            sys.exit(1)

    # Resolve models
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        try:
            available = list_models()
        except Exception as e:
            console.print(f"[red]Cannot reach Ollama: {e}[/red]")
            sys.exit(1)
        if len(available) < 2:
            console.print("[red]Need at least 2 models for a battle. Pull more models first.[/red]")
            sys.exit(1)
        # Pick first 2 by default, or let user pick
        if len(available) == 2:
            models = available
        else:
            console.print("[bold]Available models:[/bold]")
            for i, m in enumerate(available):
                console.print(f"  [cyan]{i + 1}[/cyan]. {m}")
            console.print()
            console.print("Enter model numbers (comma-separated, min 2) or [bold]all[/bold]:")
            try:
                sel = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                sys.exit(0)
            if sel.lower() == "all":
                models = available
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in sel.split(",")]
                    models = [available[i] for i in indices]
                except (ValueError, IndexError):
                    console.print("[red]Invalid selection.[/red]")
                    sys.exit(1)
            if len(models) < 2:
                console.print("[red]Need at least 2 models.[/red]")
                sys.exit(1)

    console.print()
    console.print(
        Panel(
            f"[bold]Prompt:[/bold] {prompt[:200]}{'...' if len(prompt) > 200 else ''}\n"
            f"[bold]Models:[/bold] {len(models)}\n"
            f"[bold]Max tokens:[/bold] {max_tokens}\n"
            f"[bold]Blind mode:[/bold] {'yes' if blind else 'no'}",
            title="[bold magenta]⚔ Arena Battle[/bold magenta]",
            border_style="magenta",
        )
    )

    # Run generations in parallel
    results: list[GenerationResult] = [None] * len(models)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        tasks = {}
        with ThreadPoolExecutor(max_workers=len(models)) as pool:
            for i, model in enumerate(models):
                task_id = progress.add_task(f"[cyan]{model}[/cyan]", total=None)
                future = pool.submit(generate, model, prompt, system, max_tokens)
                tasks[future] = (i, model, task_id)

            for future in as_completed(tasks):
                i, model, task_id = tasks[future]
                results[i] = future.result()
                if results[i].error:
                    progress.update(task_id, description=f"[red]{model} — failed[/red]")
                else:
                    progress.update(
                        task_id,
                        description=f"[green]{model}[/green] — {results[i].gen_tok_per_sec:.1f} tok/s",
                    )

    console.print()
    display_results(results, blind=blind)

    # Vote
    winner, tie = prompt_vote(results, blind=blind)
    result_dicts = [
        {
            "model": r.model,
            "gen_tokens": r.gen_tokens,
            "gen_tok_per_sec": r.gen_tok_per_sec,
            "ttft": r.ttft,
            "total_time": r.total_time,
        }
        for r in results
        if not r.error
    ]
    if winner or tie:
        record_battle(prompt, result_dicts, winner, tie)
        if tie:
            console.print("[yellow]Recorded as tie.[/yellow]")
        elif winner:
            console.print(f"[green]Recorded: {winner} wins![/green]")
    else:
        console.print("[dim]Skipped — no rating change.[/dim]")


def cmd_leaderboard(args):
    ratings = load_ratings()
    if not ratings:
        console.print("[yellow]No ratings yet. Run some battles first![/yellow]")
        return

    history = load_history()
    # Count wins/losses/ties per model
    stats: dict[str, dict] = {}
    for model in ratings:
        stats[model] = {"wins": 0, "losses": 0, "ties": 0, "battles": 0}

    for entry in history:
        for m in entry["models"]:
            if m not in stats:
                stats[m] = {"wins": 0, "losses": 0, "ties": 0, "battles": 0}
            stats[m]["battles"] += 1
        winner = entry.get("winner")
        if winner == "tie":
            for m in entry["models"]:
                stats[m]["ties"] += 1
        elif winner:
            stats[winner]["wins"] += 1
            for m in entry["models"]:
                if m != winner:
                    stats[m]["losses"] += 1

    table = Table(
        title="[bold magenta]Leaderboard[/bold magenta]",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Model", style="cyan")
    table.add_column("ELO", style="bold", justify="right")
    table.add_column("W", style="green", justify="right")
    table.add_column("L", style="red", justify="right")
    table.add_column("T", style="yellow", justify="right")
    table.add_column("Battles", justify="right")

    sorted_models = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    for rank, (model, elo) in enumerate(sorted_models, 1):
        s = stats.get(model, {"wins": 0, "losses": 0, "ties": 0, "battles": 0})
        table.add_row(
            str(rank),
            model,
            f"{elo:.0f}",
            str(s["wins"]),
            str(s["losses"]),
            str(s["ties"]),
            str(s["battles"]),
        )

    console.print(table)


def cmd_history(args):
    history = load_history()
    if not history:
        console.print("[yellow]No battle history yet.[/yellow]")
        return

    limit = args.limit or 20
    entries = history[-limit:]

    table = Table(
        title="[bold magenta]Battle History[/bold magenta]",
        box=box.ROUNDED,
    )
    table.add_column("Date", style="dim")
    table.add_column("Prompt", max_width=50)
    table.add_column("Models", style="cyan")
    table.add_column("Winner", style="green")

    for entry in reversed(entries):
        dt = datetime.fromisoformat(entry["timestamp"]).strftime("%Y-%m-%d %H:%M")
        prompt_preview = entry["prompt"][:47] + "..." if len(entry["prompt"]) > 50 else entry["prompt"]
        models = ", ".join(entry["models"])
        winner = entry.get("winner", "skip")
        style = "yellow" if winner == "tie" else "green" if winner != "skip" else "dim"
        table.add_row(dt, prompt_preview, models, f"[{style}]{winner}[/{style}]")

    console.print(table)


def cmd_list(args):
    try:
        models = list_models()
    except Exception as e:
        console.print(f"[red]Cannot reach Ollama: {e}[/red]")
        sys.exit(1)

    ratings = load_ratings()
    table = Table(title="[bold]Available Models[/bold]", box=box.ROUNDED)
    table.add_column("Model", style="cyan")
    table.add_column("ELO", justify="right")
    table.add_column("Status")

    for m in models:
        elo = ratings.get(m)
        elo_str = f"{elo:.0f}" if elo else "[dim]unrated[/dim]"
        table.add_row(m, elo_str, "[green]ready[/green]")

    console.print(table)


def cmd_reset(args):
    if args.confirm:
        if RATINGS_FILE.exists():
            RATINGS_FILE.unlink()
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
        console.print("[green]Ratings and history cleared.[/green]")
    else:
        console.print("Pass [bold]--confirm[/bold] to reset all ratings and history.")


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        prog="ollarena",
        description="Terminal-based model comparison arena for Ollama",
    )
    sub = parser.add_subparsers(dest="command")

    # battle
    p_battle = sub.add_parser("battle", help="Run a prompt against multiple models and vote")
    p_battle.add_argument("prompt", help='The prompt to test (use "-" to read from stdin)')
    p_battle.add_argument("-m", "--models", help="Comma-separated model names (default: interactive pick)")
    p_battle.add_argument("-s", "--system", help="System prompt")
    p_battle.add_argument("-t", "--max-tokens", type=int, default=1024, help="Max tokens (default: 1024)")
    p_battle.add_argument("--blind", action="store_true", help="Hide model names until after voting")

    # leaderboard
    sub.add_parser("leaderboard", aliases=["lb"], help="Show ELO leaderboard")

    # history
    p_hist = sub.add_parser("history", help="Show battle history")
    p_hist.add_argument("-n", "--limit", type=int, default=20, help="Number of entries")

    # list
    sub.add_parser("list", help="List available models with ratings")

    # reset
    p_reset = sub.add_parser("reset", help="Clear all ratings and history")
    p_reset.add_argument("--confirm", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "battle": cmd_battle,
        "leaderboard": cmd_leaderboard,
        "lb": cmd_leaderboard,
        "history": cmd_history,
        "list": cmd_list,
        "reset": cmd_reset,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
