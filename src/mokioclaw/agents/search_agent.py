"""Search agent — a focused research specialist using WebSearchTool."""

import json
import traceback
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mokioclaw.providers.openai_provider import create_model
from mokioclaw.tools.web_search_tool import build_web_search_tool

SEARCH_AGENT_PROMPT = """\
You are searchAgent, a focused research specialist.

Your only external capability is WebSearchTool. Search for reliable information
needed by the planner and codeAgent.

Rules:
- Use WebSearchTool for factual research.
- Prefer official or encyclopedia-style sources when available.
- Return a concise research summary and list the useful source URLs.
- Do not write files or produce application code.
"""


def run_search_agent(
    state: dict,
    instruction: str,
    *,
    writer: Any = None,
    max_loops: int = 4,
) -> dict:
    """Run the search agent to research a topic.

    Args:
        state: The current graph state (must contain ``model_name``).
        instruction: What to research.
        writer: Optional LangGraph ``StreamWriter`` for emitting events.
        max_loops: Maximum ReAct iterations (default 4).

    Returns:
        ``{ok: bool, summary: str, queries: list[str], sources: list[str],
           messages: list, tool_events: list}``
    """
    model_name = state.get("model_name", "gpt-5.5")
    research_notes = state.get("research_notes", "")

    tool = build_web_search_tool()
    model = create_model(model_name=model_name).bind_tools([tool])

    messages: list = [
        SystemMessage(content=SEARCH_AGENT_PROMPT),
        HumanMessage(
            content=(
                f"Research task: {instruction}\n\n"
                f"Existing research notes:\n{research_notes or '(none)'}\n\n"
                "Search the web for relevant, reliable information and "
                "return a concise summary with source URLs."
            )
        ),
    ]

    queries: list[str] = []
    sources: list[str] = []
    tool_events: list[dict] = []

    for _ in range(max_loops):
        response = model.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for call in response.tool_calls:
            name = call["name"]
            args = call["args"]

            # Emit tool call event
            event = {"type": "tool_call", "name": name, "args": args}
            tool_events.append(event)
            if writer is not None:
                writer(event)

            # Execute
            try:
                result = tool.invoke(args)
            except Exception:
                result = json.dumps(
                    {"ok": False, "error": traceback.format_exc()},
                    ensure_ascii=False,
                )

            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)

            # Parse result to collect queries and sources
            try:
                parsed = json.loads(result)
                if parsed.get("ok"):
                    queries.append(parsed.get("query", ""))
                    for r in parsed.get("results", []):
                        url = r.get("url", "")
                        if url and url not in sources:
                            sources.append(url)
            except json.JSONDecodeError:
                pass

            # Emit result event
            result_event = {
                "type": "search_results",
                "name": name,
                "result": result,
            }
            tool_events.append(result_event)
            if writer is not None:
                writer(result_event)

            messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=call["id"],
                    name=name,
                )
            )

    # Extract the final summary from the last AI response
    summary = _extract_text(response) if isinstance(response, AIMessage) else ""

    return {
        "ok": True,
        "summary": summary,
        "queries": queries,
        "sources": sources,
        "messages": messages,
        "tool_events": tool_events,
    }


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