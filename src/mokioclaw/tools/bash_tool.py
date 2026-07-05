"""BashTool — execute shell commands within the workspace."""

import subprocess
import time

from langchain_core.tools import StructuredTool

from mokioclaw.core.state import RuntimeState


def _run_bash(state: RuntimeState, command: str, timeout_seconds: int = 120) -> str:
    """Run a shell command in the workspace directory.

    Args:
        command:         The shell command to execute.
        timeout_seconds: Maximum runtime in seconds (default 120, max 600).
    """
    timeout_seconds = min(max(timeout_seconds, 1), 600)

    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(state.workspace),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout_seconds}s."

    parts: list[str] = []
    if proc.stdout:
        parts.append(proc.stdout.rstrip())
    if proc.stderr:
        parts.append(f"[stderr]\n{proc.stderr.rstrip()}")
    if proc.returncode != 0:
        parts.append(f"[exit code: {proc.returncode}]")

    return "\n".join(parts) if parts else "(no output)"


def build_bash_tool(state: RuntimeState) -> StructuredTool:
    """Return a ``StructuredTool`` that executes shell commands."""

    def bash(command: str, timeout_seconds: int = 120) -> str:
        return _run_bash(state, command, timeout_seconds)

    return StructuredTool.from_function(
        func=bash,
        name="BashTool",
        description="Execute a shell command in the workspace directory. "
        "timeout_seconds: max runtime (1-600, default 120).",
    )
