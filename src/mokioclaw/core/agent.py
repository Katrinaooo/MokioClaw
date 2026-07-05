"""ReAct agent loop — stream_agent_events generator."""

import json
import traceback
from pathlib import Path
from typing import Any, Iterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mokioclaw.core.state import RuntimeState
from mokioclaw.providers.openai_provider import create_model
from mokioclaw.tools.registry import build_tools

ACTOR_PROMPT = """\
You are the actor node in MokioClaw's ReAct workflow.

You implement the user's task using tools. Work inside the workspace only.

Rules:
- Use FileWriteTool for new files.
- Use FileReadTool before editing existing files.
- Use FileEditTool for focused edits.
- Use BashTool to run commands and test results.
- BashTool already runs inside the workspace. Use relative paths, never "cd /workspace".
- End with a concise summary of files changed and commands run.
"""


def stream_agent_events(
    task: str,
    *,
    workspace: Path,
    model_name: str = "gpt-4o",
    max_loops: int = 10,
) -> Iterator[dict[str, Any]]:
    """Run the ReAct loop, yielding structured events.

    Yields:
        ``{"type": "ai_message", "content": str}``
            The full text content of each AI response (may be empty if only tool calls).

        ``{"type": "tool_call", "name": str, "args": dict}``
            One event per tool call the model requested.

        ``{"type": "tool_result", "name": str, "result": str}``
            The result of executing a tool call.

        ``{"type": "final_answer", "content": str}``
            Emitted when the loop ends (no more tool calls, or max_loops reached).
            Contains the last AI text content.
    """
    # 1. Create state and tools
    state = RuntimeState(workspace=workspace)
    tools = build_tools(state)
    tool_by_name = {t.name: t for t in tools}

    # 2. Build initial messages
    model = create_model(model_name=model_name).bind_tools(tools)
    messages: list = [
        SystemMessage(content=ACTOR_PROMPT),
        HumanMessage(content=task),
    ]

    last_text_content = ""

    # 3. ReAct loop
    for _loop_index in range(max_loops):
        response = model.invoke(messages)
        messages.append(response)

        # Extract text content
        text = _extract_text(response)
        last_text_content = text or last_text_content

        yield {"type": "ai_message", "content": text}

        # No tool calls → done
        if not response.tool_calls:
            yield {
                "type": "final_answer",
                "content": text or "(no output)",
            }
            return

        # 4. Execute each tool call
        for call in response.tool_calls:
            name = call["name"]
            args = call["args"]
            tool_id = call["id"]

            yield {"type": "tool_call", "name": name, "args": args}

            tool = tool_by_name.get(name)
            if tool is None:
                result = f"Error: unknown tool {name!r}"
            else:
                try:
                    result = tool.invoke(args)
                except Exception:
                    result = f"Error executing {name}: {traceback.format_exc()}"

            # Ensure result is a string for ToolMessage
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)

            yield {"type": "tool_result", "name": name, "result": result}

            messages.append(ToolMessage(content=result, tool_call_id=tool_id))

    # 5. Max loops reached
    yield {
        "type": "final_answer",
        "content": last_text_content or "(max loops reached, no response)",
    }


def _extract_text(response: AIMessage) -> str:
    """Extract plain-text content from an AIMessage, handling both string and list content."""
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item for item in content if isinstance(item, str) or item.get("type") == "text"]
        return "".join(
            item if isinstance(item, str) else item.get("text", "")
            for item in parts
        )
    return str(content)
