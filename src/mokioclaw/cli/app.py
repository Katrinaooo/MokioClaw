"""Typer CLI — the ``mokioclaw`` command."""

from pathlib import Path

import typer

from mokioclaw.core.paths import ensure_workspace
from mokioclaw.core.state import RuntimeState

app = typer.Typer(
    name="mokioclaw",
    help="MokioClaw — A CLI AI agent framework with tool-based architecture.",
)


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
        "gpt-4o",
        "--model",
        "-m",
        help="OpenAI model name to use.",
    ),
):
    """Run an AI agent to accomplish *task* within the given workspace."""
    if workspace is None:
        # Auto-create a workspace under the current directory
        ws = ensure_workspace("./mokioclaw_workspace")
    else:
        ws = ensure_workspace(workspace)

    state = RuntimeState(workspace=ws)

    typer.echo(f"Workspace: {ws}")
    typer.echo(f"Task: {task}")
    typer.echo(f"Model: {model}")

    # TODO: wire up the agent loop with build_tools(state) + create_model(model)
    typer.echo("\nAgent loop not yet implemented — tools and provider are ready.")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
) -> None:
    """MokioClaw entry point.  Use ``run`` to execute a task, or ``mokioclaw --help``."""
    if ctx.invoked_subcommand is None:
        # Show help when no subcommand is given
        typer.echo(app.get_help(ctx))


if __name__ == "__main__":
    app()
