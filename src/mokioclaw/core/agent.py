"""Graph-based agent — stream_agent_events via build_workflow().stream()."""

import json
from pathlib import Path
from typing import Any, Iterator

from mokioclaw.core.state import RuntimeState
from mokioclaw.graph.workflow import build_workflow


def stream_agent_events(
    task: str,
    *,
    workspace: Path,
    model_name: str = "gpt-5.5",
    max_attempts: int = 3,
) -> Iterator[dict[str, Any]]:
    """Run the plan→verify graph, yielding structured events.

    Yields:
        node_start / node_end — bookend each graph node.
        plan — planner output (summary, todos, criteria, commands).
        verification — verifier verdict and checks.
        final_answer — the final summary.
    """
    runtime = RuntimeState(workspace=workspace)
    graph = build_workflow()

    inputs: dict[str, Any] = {
        "task": task,
        "model_name": model_name,
        "runtime": runtime,
        "max_attempts": max_attempts,
    }

    for chunk in graph.stream(inputs, stream_mode=["updates", "custom"]):
        mode, data = chunk
        if mode == "custom":
            # writer() events from inside nodes
            yield from _emit_custom(data)
            continue

        # mode == "updates": node-level state changes
        for node_name, node_output in data.items():
            yield {"type": "node_start", "node": node_name}

            if node_name == "planner":
                yield from _emit_planner(node_output)

            elif node_name == "verifier":
                yield from _emit_verifier(node_output)

            elif node_name == "final":
                yield from _emit_final(node_output)

            yield {"type": "node_end", "node": node_name}


# ---------------------------------------------------------------------------
# Per-node event emitters
# ---------------------------------------------------------------------------

def _emit_planner(output: dict) -> Iterator[dict]:
    """Emit the planner's structured plan."""
    todos = output.get("todos") or []
    criteria = output.get("acceptance_criteria") or []
    commands = output.get("verification_commands") or []

    yield {
        "type": "plan",
        "summary": output.get("plan_summary", ""),
        "todos": todos,
        "acceptance_criteria": criteria,
        "verification_commands": commands,
    }


def _emit_custom(event: dict) -> Iterator[dict]:
    """Forward writer() events from inside nodes to the CLI."""
    etype = event.get("type", "")
    if etype in ("tool_call", "tool_result", "handoff", "search_results"):
        yield event
    else:
        yield event  # pass through unknown custom events too


def _emit_verifier(output: dict) -> Iterator[dict]:
    """Emit verification results."""
    yield {
        "type": "verification",
        "passed": output.get("passed", False),
        "checks": output.get("verification_checks") or [],
        "results": output.get("verification_results") or [],
    }


def _emit_final(output: dict) -> Iterator[dict]:
    """Emit the final answer."""
    yield {
        "type": "final_answer",
        "content": output.get("final_answer", ""),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(msg) -> str:
    """Extract plain-text content from any message."""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item if isinstance(item, str) else item.get("text", "")
            for item in content
            if isinstance(item, str) or item.get("type") == "text"
        ]
        return "".join(parts)
    return str(content)