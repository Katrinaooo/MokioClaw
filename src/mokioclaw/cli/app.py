"""Typer CLI — the ``mokioclaw`` command."""

from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from mokioclaw.core.agent import stream_agent_events
from mokioclaw.core.paths import ensure_workspace

app = typer.Typer(
    name="mokioclaw",
    help="MokioClaw — A CLI AI agent framework with tool-based architecture.",
)

console = Console()


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
    max_loops: int = typer.Option(
        10,
        "--max-loops",
        "-l",
        help="Maximum ReAct loop iterations.",
    ),
):
    """Run an AI agent to accomplish *task* within the given workspace."""
    if workspace is None:
        ws = ensure_workspace("./workspace")
    else:
        ws = ensure_workspace(workspace)

    console.print()
    console.print(
        Panel.fit(
            f"[bold]Workspace:[/bold] {ws}\n[bold]Task:[/bold] {task}\n[bold]Model:[/bold] {model}",
            title="🚀 MokioClaw",
            border_style="blue",
        )
    )
    console.print()

    for event in stream_agent_events(
        task,
        workspace=ws,
        model_name=model,
        max_loops=max_loops,
    ):
        etype = event["type"]

        if etype == "ai_message":
            content = event["content"]
            if content:
                console.print(Panel(Markdown(content), border_style="green", title="🤖 AI"))

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
                    title=f"📋 {name} result",
                )
            )

        elif etype == "final_answer":
            console.print()
            console.print(
                Panel(
                    Markdown(event["content"]),
                    border_style="bold green",
                    title="✅ Final Answer",
                )
            )

    console.print()


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
