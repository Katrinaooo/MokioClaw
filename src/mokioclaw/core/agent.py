"""Graph-based agent — stream_agent_events via build_workflow().stream()."""

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterator

from mokioclaw.core.checkpoint import CheckpointManager
from mokioclaw.core.session import (
    append_assistant_turn,
    append_user_turn,
    build_session_context,
    load_or_create_session,
    save_session,
)
from mokioclaw.core.state import RuntimeState, create_runtime
from mokioclaw.core.trace import TraceRecorder
from mokioclaw.graph.workflow import build_entry_workflow, build_workflow


def stream_agent_events(
    task: str,
    *,
    workspace: Path,
    model_name: str = "gpt-5.5",
    max_attempts: int = 3,
    approval_mode: str = "inline",
    approval_handler: Callable[..., Any] | None = None,
    checkpoint_mode: str = "light",
    trace_mode: str = "on",
    resume_workspace: str = "",
) -> Iterator[dict[str, Any]]:
    """Run the plan→verify graph, yielding structured events.

    Integrates checkpoint save/restore, execution tracing, and command
    approval.  Supports ``KeyboardInterrupt`` for graceful interruption.

    Yields:
        node_start / node_end — bookend each graph node.
        plan — planner output (summary, todos, criteria, commands).
        tool_call / tool_result / handoff — custom events from inside nodes.
        verification — verifier verdict and checks.
        final_answer — the final summary.
    """
    # --- 1. Create runtime ---
    state = create_runtime(
        workspace,
        approval_mode=approval_mode,
        approval_handler=approval_handler,
        checkpoint_mode=checkpoint_mode,
        trace_mode=trace_mode,
        resume_from=resume_workspace,
    )

    # --- 2. Create checkpoint manager and trace recorder ---
    manager = CheckpointManager(state, task=task)
    trace = TraceRecorder(state, task=task)

    # --- 3. Build inputs (or resume) ---
    resumed = False
    resume_event = None

    if resume_workspace:
        inputs, resume_event = CheckpointManager.load_resume_inputs(
            state, task=task, max_attempts=max_attempts
        )
        resumed = resume_event.get("found", False)
    else:
        inputs: dict[str, Any] = {
            "task": task,
            "model_name": model_name,
            "runtime": state,
            "max_attempts": max_attempts,
        }

    # --- 4. Record start ---
    trace.start(inputs, resumed=resumed, resume_event=resume_event)
    if resume_event:
        yield resume_event

    graph = build_workflow()
    current_state: dict[str, Any] = dict(inputs)
    latest_node = "start"

    manager.save(current_state, status="started", latest_node="start")

    try:
        # --- 5. Run the workflow ---
        for chunk in graph.stream(inputs, stream_mode=["updates", "custom"]):
            mode, data = chunk

            if mode == "custom":
                trace.record_custom_event(data)
                if data.get("type") in ("tool_call", "handoff"):
                    manager.save(
                        current_state, status="running",
                        latest_node=latest_node, event=data,
                    )
                yield from _emit_custom(data)
                continue

            # mode == "updates": node-level state changes
            for node_name, node_output in data.items():
                latest_node = node_name
                current_state.update(node_output)

                trace.record_graph_update({"node": node_name})
                manager.save(
                    current_state, status="running",
                    latest_node=latest_node,
                    event={"type": "node_update", "node": node_name},
                )

                yield {"type": "node_start", "node": node_name}

                if node_name == "planner":
                    yield from _emit_planner(node_output)

                elif node_name == "verifier":
                    yield from _emit_verifier(node_output)

                elif node_name == "final":
                    yield from _emit_final(node_output)

                elif node_name == "context_monitor":
                    yield from _emit_context_monitor(node_output)

                elif node_name == "context_compressor":
                    yield from _emit_context_compressor(node_output)

                yield {"type": "node_end", "node": node_name}

    finally:
        _finalize(manager, trace, current_state, latest_node)


def stream_session_events(
    task: str,
    *,
    workspace: Path,
    model_name: str = "gpt-5.5",
    max_attempts: int = 3,
    approval_mode: str = "inline",
    approval_handler: Callable[..., Any] | None = None,
    checkpoint_mode: str = "light",
    trace_mode: str = "on",
    resume_workspace: str = "",
) -> Iterator[dict[str, Any]]:
    """Multi-turn session-aware agent loop.

    1. Loads or creates a session for the workspace.
    2. Records the user's turn.
    3. Runs the entry workflow (intent_router → chat or workflow).
    4. If chat: replies directly, records assistant turn.
    5. If workflow: runs the full plan→verify graph, records assistant turn.
    6. Saves the session after each turn.
    """
    # --- 1. Load session ---
    session = load_or_create_session(workspace)
    turn = append_user_turn(session, task)
    session_context = build_session_context(workspace, session)

    # --- 2. Run entry workflow ---
    entry_graph = build_entry_workflow()
    entry_inputs: dict[str, Any] = {
        "task": task,
        "plan_summary": session_context,
        "model_name": model_name,
    }

    entry_result = entry_graph.invoke(entry_inputs)
    route = entry_result.get("intent_route", "workflow")

    yield {
        "type": "session_start",
        "session_id": session.get("session_id", ""),
        "turn": turn,
        "route": route,
    }

    # --- 3. Chat branch ---
    if route == "chat":
        chat_response = entry_result.get("chat_response", "")
        yield {
            "type": "final_answer",
            "content": chat_response,
        }

        append_assistant_turn(
            session, turn=turn, route="chat",
            content=chat_response,
            summary=chat_response[:200],
        )
        save_session(workspace, session)
        return

    # --- 4. Workflow branch ---
    # Enrich task with session context
    enriched_task = f"{task}\n\n[Session context]\n{session_context}"

    state = create_runtime(
        workspace,
        approval_mode=approval_mode,
        approval_handler=approval_handler,
        checkpoint_mode=checkpoint_mode,
        trace_mode=trace_mode,
        resume_from=resume_workspace,
    )

    manager = CheckpointManager(state, task=task)
    trace = TraceRecorder(state, task=task)

    inputs: dict[str, Any] = {
        "task": task,
        "model_name": model_name,
        "runtime": state,
        "max_attempts": max_attempts,
    }

    trace.start(inputs)
    graph = build_workflow()
    current_state: dict[str, Any] = dict(inputs)
    latest_node = "start"
    manager.save(current_state, status="started", latest_node="start")

    final_content = ""

    try:
        for chunk in graph.stream(inputs, stream_mode=["updates", "custom"]):
            mode, data = chunk

            if mode == "custom":
                trace.record_custom_event(data)
                if data.get("type") in ("tool_call", "handoff"):
                    manager.save(
                        current_state, status="running",
                        latest_node=latest_node, event=data,
                    )
                yield from _emit_custom(data)
                continue

            for node_name, node_output in data.items():
                latest_node = node_name
                current_state.update(node_output)

                trace.record_graph_update({"node": node_name})
                manager.save(
                    current_state, status="running",
                    latest_node=latest_node,
                    event={"type": "node_update", "node": node_name},
                )

                yield {"type": "node_start", "node": node_name}

                if node_name == "planner":
                    yield from _emit_planner(node_output)

                elif node_name == "verifier":
                    yield from _emit_verifier(node_output)

                elif node_name == "final":
                    final_content = node_output.get("final_answer", "")
                    yield from _emit_final(node_output)

                elif node_name == "context_monitor":
                    yield from _emit_context_monitor(node_output)

                elif node_name == "context_compressor":
                    yield from _emit_context_compressor(node_output)

                yield {"type": "node_end", "node": node_name}

    finally:
        _finalize(manager, trace, current_state, latest_node)

    # --- 5. Record assistant turn ---
    append_assistant_turn(
        session, turn=turn, route="workflow",
        content=final_content,
        summary=final_content[:200],
    )
    save_session(workspace, session)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finalize(
    manager: CheckpointManager,
    trace: TraceRecorder,
    current_state: dict,
    latest_node: str,
) -> None:
    """Save final checkpoint and trace."""
    manager.save(current_state, status="finished", latest_node=latest_node)
    trace.end(
        status="finished",
        latest_node=latest_node,
        final_state=current_state,
    )


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
    yield event


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


def _emit_context_monitor(output: dict) -> Iterator[dict]:
    """Emit context monitor status."""
    yield {
        "type": "context_monitor",
        "token_count": output.get("context_token_count", 0),
        "token_limit": output.get("context_token_limit", 400_000),
        "should_compress": output.get("context_should_compress", False),
        "next_node": output.get("context_next_node", ""),
    }


def _emit_context_compressor(output: dict) -> Iterator[dict]:
    """Emit compression summary."""
    events = output.get("compression_events") or []
    last = events[-1] if events else {}
    yield {
        "type": "context_compression",
        "tokens_before": last.get("tokens_before", 0),
        "tokens_after": last.get("tokens_after", 0),
        "summary_preview": (output.get("context_summary") or "")[:300],
    }


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