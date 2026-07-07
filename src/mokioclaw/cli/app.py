"""Typer CLI — the ``mokioclaw`` command."""

from pathlib import Path
from typing import Annotated, Literal, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from typer import Option

from mokioclaw.core.agent import stream_agent_events
from mokioclaw.core.paths import ensure_workspace

app = typer.Typer(
    name="mokioclaw",
    help="MokioClaw — A CLI AI agent framework with tool-based architecture.",
)

console = Console()

# Node display labels
NODE_LABELS = {
    "planner": "📋 Planner",
    "verifier": "✅ Verifier",
    "context_monitor": "📈 Context Monitor",
    "context_compressor": "🗜️ Context Compressor",
    "final": "📝 Final",
}

STATUS_ICONS = {
    "completed": "[x]",
    "blocked": "[!]",
    "in_progress": "[-]",
    "pending": "[ ]",
}


@app.command()
def run(
    task: Annotated[str, typer.Argument(help="The task description to execute.")],
    workspace: Annotated[
        Optional[Path],
        Option("--workspace", "-w", help="Workspace directory."),
    ] = None,
    model: Annotated[
        str,
        Option("--model", "-m", help="OpenAI model name to use."),
    ] = "gpt-5.5",
    max_attempts: Annotated[
        int,
        Option("--max-attempts", "-a", help="Maximum plan→execute→verify cycles (default 3)."),
    ] = 3,
    approval_mode: Annotated[
        Literal["inline", "auto", "deny"],
        Option("--approval-mode", help="Approval mode: inline, auto, or deny (default inline)."),
    ] = "inline",
    checkpoint_mode: Annotated[
        Literal["light", "strict", "off"],
        Option("--checkpoint-mode", help="Checkpoint mode: light, strict, or off (default light)."),
    ] = "light",
    trace_mode: Annotated[
        Literal["on", "off"],
        Option("--trace-mode", help="Trace mode: on or off (default on)."),
    ] = "on",
    resume: Annotated[
        Optional[Path],
        Option("--resume", help="Resume from a previous workspace."),
    ] = None,
):
    """Run an AI agent to accomplish *task* within the given workspace."""
    ws = ensure_workspace(str(workspace)) if workspace else ensure_workspace("./workspace")

    console.print()
    console.print(
        Panel.fit(
            f"[bold]Workspace:[/bold] {ws}\n"
            f"[bold]Task:[/bold] {task}\n"
            f"[bold]Model:[/bold] {model}\n"
            f"[bold]Max attempts:[/bold] {max_attempts}",
            title="🚀 MokioClaw",
            border_style="blue",
        )
    )
    console.print()

    for event in stream_agent_events(
        task,
        workspace=ws,
        model_name=model,
        max_attempts=max_attempts,
        approval_mode=approval_mode,
        checkpoint_mode=checkpoint_mode,
        trace_mode=trace_mode,
        resume_workspace=str(resume) if resume else "",
    ):
        etype = event["type"]

        if etype == "node_start":
            node = event["node"]
            label = NODE_LABELS.get(node, node)
            console.print(Panel(f"[bold]{label}[/bold]", border_style="blue"))

        elif etype == "plan":
            _render_plan(event)

        elif etype == "ai_message":
            content = event["content"]
            if content:
                console.print(Panel(Markdown(content), border_style="green"))

        elif etype == "tool_call":
            name = event["name"]
            args = event["args"]
            args_text = _format_args(args)
            console.print(
                Panel(
                    f"[bold cyan]{name}[/bold cyan]\n{args_text}",
                    border_style="yellow",
                    title="🔧 Tool Call",
                )
            )

        elif etype == "tool_result":
            name = event["name"]
            result = event["result"]
            display = _truncate(result, 2000)
            console.print(
                Panel(
                    Syntax(display, "text", theme="ansi_dark"),
                    border_style="dim",
                    title=f"📋 {name}",
                )
            )

        elif etype == "verification":
            _render_verification(event)

        elif etype == "handoff":
            console.print(
                Panel(
                    f"[bold]🔄 {event.get('from', '?')} → {event.get('to', '?')}[/bold]\n"
                    f"[dim]{event.get('instruction', '')[:200]}[/dim]",
                    border_style="magenta",
                    title="Delegation",
                )
            )

        elif etype == "context_monitor":
            _render_context_monitor(event)

        elif etype == "context_compression":
            _render_context_compression(event)

        elif etype == "final_answer":
            console.print()
            console.print(
                Panel(
                    Markdown(event["content"]),
                    border_style="bold green",
                    title="📝 Final",
                )
            )

        elif etype == "node_end":
            pass  # node_end is handled implicitly by node_start of the next node

    console.print()


# ---------------------------------------------------------------------------
# Rich render helpers
# ---------------------------------------------------------------------------

def _render_plan(event: dict) -> None:
    """Render the planner's output: summary, todos, criteria, commands."""
    summary = event.get("summary", "")
    console.print(Panel(Markdown(summary), border_style="cyan", title="📋 Plan"))

    todos = event.get("todos") or []
    if todos:
        todo_lines: list[str] = []
        for t in todos:
            icon = STATUS_ICONS.get(t.get("status", "pending"), "[ ]")
            note = f" — {t['note']}" if t.get("note") else ""
            todo_lines.append(f"  {icon} {t['content']}{note}")
        console.print(Panel("\n".join(todo_lines), border_style="dim", title="📋 Todos"))

    criteria = event.get("acceptance_criteria") or []
    if criteria:
        lines = [f"  • {c}" for c in criteria]
        console.print(Panel("\n".join(lines), border_style="dim", title="📋 Criteria"))

    commands = event.get("verification_commands") or []
    if commands:
        lines = [f"  $ {c}" for c in commands]
        console.print(Panel("\n".join(lines), border_style="dim", title="📋 Commands"))


def _render_verification(event: dict) -> None:
    """Render the verifier's output: passed/failed, checks, command results."""
    passed = event.get("passed", False)
    title = "✅ Verifier — PASSED" if passed else "❌ Verifier — FAILED"
    border = "green" if passed else "red"

    checks = event.get("checks") or []
    results = event.get("results") or []

    parts: list[str] = []

    if checks:
        parts.append("[bold]Checks:[/bold]")
        for c in checks:
            icon = "✅" if c.get("passed") else "❌"
            parts.append(f"  {icon} {c.get('name', '?')}: {c.get('detail', '')}")

    if results:
        parts.append("[bold]Command results:[/bold]")
        for r in results:
            icon = "✅" if r.get("ok") else "❌"
            cmd = r.get("command", "")
            exit_code = r.get("exit_code", "")
            parts.append(f"  {icon} `{cmd}` (exit={exit_code})")

    console.print(Panel("\n".join(parts), border_style=border, title=title))


def _format_args(args: dict) -> str:
    """Format tool call arguments for display."""
    lines: list[str] = []
    for key, value in args.items():
        val_str = str(value)
        if len(val_str) > 200:
            val_str = val_str[:200] + "…"
        lines.append(f"  [dim]{key}:[/dim] {val_str}")
    return "\n".join(lines)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to *max_len* characters, appending an ellipsis note."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n… (truncated, total {len(text)} chars)"


def _render_context_monitor(event: dict) -> None:
    """Render context monitor: token count, limit, and compression flag."""
    token_count = event.get("token_count", 0)
    token_limit = event.get("token_limit", 400_000)
    pct = token_count / max(token_limit, 1) * 100
    should_compress = event.get("should_compress", False)
    next_node = event.get("next_node", "")

    bar = _make_bar(pct, 30)
    flag = "⚠️ 压缩触发!" if should_compress else ""

    console.print(
        Panel(
            f"[bold]Tokens:[/bold] {token_count:,} / {token_limit:,}  ({pct:.1f}%)\n"
            f"[dim]{bar}[/dim]\n"
            f"[bold]Next:[/bold] {next_node}  {flag}",
            border_style="yellow" if should_compress else "dim",
            title="📈 Context Monitor",
        )
    )


def _render_context_compression(event: dict) -> None:
    """Render compression result."""
    before = event.get("tokens_before", 0)
    after = event.get("tokens_after", 0)
    saved = before - after
    pct = saved / max(before, 1) * 100

    console.print(
        Panel(
            f"[bold]Tokens:[/bold] {before:,} → {after:,}  "
            f"([green]saved {saved:,} ({pct:.1f}%)[/green])",
            border_style="green",
            title="🗜️ Context Compression",
        )
    )


def _make_bar(pct: float, width: int) -> str:
    """Draw a simple ASCII progress bar."""
    filled = int(width * pct / 100)
    if filled > width:
        filled = width
    bar = "█" * filled + "░" * (width - filled)
    if pct > 80:
        return f"[red]{bar}[/red]"
    elif pct > 50:
        return f"[yellow]{bar}[/yellow]"
    return f"[green]{bar}[/green]"


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """MokioClaw entry point.  Use ``run`` to execute a task, or ``mokioclaw --help``."""
    if ctx.invoked_subcommand is None:
        typer.echo(app.get_help(ctx))


if __name__ == "__main__":
    app()