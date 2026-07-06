"""Graph-based agent — stream_agent_events via build_workflow().stream()."""

import json
from pathlib import Path
from typing import Any, Iterator

from langchain_core.messages import AIMessage, ToolMessage

from mokioclaw.core.state import RuntimeState
from mokioclaw.graph.workflow import build_workflow


def stream_agent_events(
    task: str,
    *,
    workspace: Path,
    model_name: str = "gpt-5.5",
    max_attempts: int = 3,
) -> Iterator[dict[str, Any]]:
    """Run the plan→execute→verify graph, yielding structured events.

    Yields:
        node_start / node_end — bookend each graph node.
        plan — planner output (summary, todos, criteria, commands).
        ai_message / tool_call / tool_result — actor's ReAct loop.
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

    seen_ids: set[str] = set()

    for chunk in graph.stream(inputs, stream_mode="updates"):
        for node_name, node_output in chunk.items():
            yield {"type": "node_start", "node": node_name}

            if node_name == "planner":
                yield from _emit_planner(node_output)

            elif node_name == "actor":
                yield from _emit_actor(node_output, seen_ids)

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


def _emit_actor(output: dict, seen_ids: set[str]) -> Iterator[dict]:
    """Emit AI messages and tool calls from the actor's message history.

    Only emits messages whose id hasn't been seen in a previous actor run
    (the graph may loop back through the actor via planner→actor on retry).
    """
    messages = output.get("messages") or []
    for msg in messages:
        mid = getattr(msg, "id", None)
        if mid and mid in seen_ids:
            continue
        if mid:
            seen_ids.add(mid)

        if isinstance(msg, AIMessage):
            content = _extract_text(msg)
            if content:
                yield {"type": "ai_message", "content": content}
            if msg.tool_calls:
                for call in msg.tool_calls:
                    yield {
                        "type": "tool_call",
                        "name": call["name"],
                        "args": call["args"],
                    }

        elif isinstance(msg, ToolMessage):
            yield {
                "type": "tool_result",
                "name": getattr(msg, "name", None) or "unknown",
                "result": _extract_text(msg),
            }


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