"""Code agent — a focused implementation specialist for the workspace."""

import json
import traceback
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mokioclaw.core.state import RuntimeState
from mokioclaw.graph.memory import build_layered_memory, format_layered_memory_for_prompt
from mokioclaw.providers.openai_provider import create_model
from mokioclaw.tools.registry import build_tools

CODE_AGENT_PROMPT = """\
You are codeAgent, a focused implementation specialist.

You implement the planner's instruction inside the workspace using file and
shell tools.

Rules:
- You must update todo progress explicitly.
- Before starting a todo, call TodoUpdateTool with status "in_progress".
- After finishing that todo, call TodoUpdateTool with status "completed".
- If a todo is impossible, call TodoUpdateTool with status "blocked" and explain.
- Use FileWriteTool for new files.
- Use FileReadTool before editing existing files.
- Use FileEditTool for focused edits.
- Use BashTool for non-interactive checks.
- Use NotepadAppendTool to record durable findings, decisions, important files,
  blockers, and next-step context that should survive compression.
- Use NotepadReadTool when you need to recover prior notes.
- BashTool already runs inside the workspace. Use relative paths, never "cd /workspace".
- Incorporate research notes and source URLs when the task asks for researched content.
- End with a concise summary of files changed and checks run.
"""


def run_code_agent(
    state: dict,
    instruction: str,
    *,
    writer: Any = None,
    max_loops: int = 10,
) -> dict:
    """Run the code agent to implement a task in the workspace.

    Args:
        state: The current graph state (must contain ``runtime``, ``model_name``,
               and optionally ``todos``).
        instruction: What to implement.
        writer: Optional ``StreamWriter`` for emitting events.
        max_loops: Maximum ReAct iterations (default 10).

    Returns:
        ``{ok: bool, summary: str, todos: list[dict], messages: list,
           tool_events: list}``
    """
    model_name = state.get("model_name", "gpt-5.5")
    runtime = state.get("runtime") or RuntimeState()
    todos = state.get("todos", [])

    # --- 1. Build tools ---
    from mokioclaw.graph.nodes import build_todo_update_tool  # lazy — breaks circular import

    file_tools = build_tools(runtime)
    todo_tool = build_todo_update_tool(todos)
    tools = file_tools + [todo_tool]
    tool_by_name = {t.name: t for t in tools}

    model = create_model(model_name=model_name).bind_tools(tools)

    # --- 2. Build layered memory ---
    memory = build_layered_memory(state, node="codeAgent")
    if writer is not None:
        writer({
            "type": "memory_snapshot",
            "node": "codeAgent",
            "memory": memory,
        })

    # --- 3. Build messages ---
    plan_summary = state.get("plan_summary", "")
    research_notes = state.get("research_notes", "")

    instruction_block = (
        f"Task: {state.get('task', '')}\n\n"
        f"Instruction: {instruction}\n\n"
        f"Plan: {plan_summary}\n\n"
        f"Research notes: {research_notes or '(none)'}\n\n"
        f"## Memory Snapshot\n{format_layered_memory_for_prompt(memory)}"
    )

    messages: list = [
        SystemMessage(content=CODE_AGENT_PROMPT),
        HumanMessage(content=instruction_block),
    ]

    # --- 4. ReAct loop ---
    tool_events: list[dict] = []
    last_text = ""

    for _ in range(max_loops):
        response = model.invoke(messages)
        messages.append(response)

        text = _extract_text(response)
        last_text = text or last_text

        if not response.tool_calls:
            break

        for call in response.tool_calls:
            name = call["name"]
            args = call["args"]
            tool_id = call["id"]

            # Emit tool call event
            event = {"type": "tool_call", "name": name, "args": args}
            tool_events.append(event)
            if writer is not None:
                writer(event)

            # Execute
            tool = tool_by_name.get(name)
            if tool is None:
                result = f"Error: unknown tool {name!r}"
            else:
                try:
                    result = tool.invoke(args)
                except Exception:
                    result = f"Error executing {name}: {traceback.format_exc()}"

            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)

            # Emit result event
            result_event = {
                "type": "tool_result",
                "name": name,
                "result": result,
            }
            tool_events.append(result_event)
            if writer is not None:
                writer(result_event)

            messages.append(
                ToolMessage(content=result, tool_call_id=tool_id, name=name)
            )

    return {
        "ok": True,
        "summary": last_text,
        "todos": todos,  # mutated in-place by TodoUpdateTool
        "messages": messages,
        "tool_events": tool_events,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(response: AIMessage) -> str:
    """Extract plain-text content from an AIMessage."""
    content = response.content
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