"""Typer CLI — the ``mokioclaw`` command."""

from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

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
    "actor": "🔧 Actor",
    "verifier": "✅ Verifier",
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
    task: str = typer.Argument(..., help="The task description to execute."),
    workspace: str = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory. Defaults to an auto-created temp directory.",
    ),
    model: str = typer.Option(
        "gpt-5.5",
        "--model",
        "-m",
        help="OpenAI model name to use.",
    ),
    max_attempts: int = typer.Option(
        3,
        "--max-attempts",
        "-a",
        help="Maximum plan→execute→verify cycles (default 3).",
    ),
):
    """Run an AI agent to accomplish *task* within the given workspace."""
    ws = ensure_workspace(workspace) if workspace else ensure_workspace("./workspace")

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


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """MokioClaw entry point.  Use ``run`` to execute a task, or ``mokioclaw --help``."""
    if ctx.invoked_subcommand is None:
        typer.echo(app.get_help(ctx))


if __name__ == "__main__":
    app()