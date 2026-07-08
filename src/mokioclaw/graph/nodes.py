"""LangGraph nodes: planner, actor, verifier — plus the verifier route."""

import json
import re
import subprocess
import traceback
from typing import Any

from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph.message import RemoveMessage

from mokioclaw.agents.code_agent import run_code_agent
from mokioclaw.agents.search_agent import run_search_agent
from mokioclaw.core.state import RuntimeState
from mokioclaw.graph.memory import build_layered_memory
from mokioclaw.graph.state import MokioGraphState, VerificationResult
from mokioclaw.prompts.stage2 import ACTOR_PROMPT
from mokioclaw.prompts.stage3 import PLANNER_PROMPT, VERIFIER_PROMPT
from mokioclaw.prompts.stage4 import CONTEXT_COMPRESSION_PROMPT
from mokioclaw.providers.openai_provider import create_model
from mokioclaw.tools.file_tools import build_file_read_tool
from mokioclaw.tools.grep_tool import build_grep_tool
from mokioclaw.tools.registry import build_tools

# ---------------------------------------------------------------------------
# Custom tools for the graph
# ---------------------------------------------------------------------------

def build_todo_write_tool() -> StructuredTool:
    """Tool for the planner to emit a structured plan in one call."""

    def write_plan(
        plan_summary: str,
        todos: list[dict],
        acceptance_criteria: list[str],
        verification_commands: list[str],
    ) -> str:
        return json.dumps(
            {
                "status": "plan_recorded",
                "todo_count": len(todos),
                "criteria_count": len(acceptance_criteria),
                "command_count": len(verification_commands),
            },
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        func=write_plan,
        name="TodoWriteTool",
        description=(
            "Emit the full execution plan. "
            "plan_summary: 1-2 sentence overview. "
            "todos: list of {id, content, status, note} — status must be 'pending'. "
            "acceptance_criteria: list of verifiable success conditions. "
            "verification_commands: shell commands to run for verification."
        ),
    )


def build_todo_update_tool(todos: list[dict]) -> StructuredTool:
    """Tool for the actor to update a single todo's status."""

    def update_todo(id: str, status: str, note: str = "") -> str:
        for t in todos:
            if t["id"] == id:
                t["status"] = status
                if note:
                    t["note"] = note
                return f"Todo {id!r} -> {status!r}"
        return f"Error: todo {id!r} not found"

    return StructuredTool.from_function(
        func=update_todo,
        name="TodoUpdateTool",
        description=(
            "Update a todo item's status. "
            "id: the todo's id. "
            "status: 'pending' | 'in_progress' | 'completed' | 'blocked'. "
            "note: optional detail (required when status is 'blocked')."
        ),
    )


def build_read_only_tools(state: RuntimeState) -> list[StructuredTool]:
    """Return only read-only tools (no write, no bash)."""
    return [
        build_file_read_tool(state),
        build_grep_tool(state),
    ]


# ---------------------------------------------------------------------------
# Stage 3: sub-agent tool wrappers
# ---------------------------------------------------------------------------

def _build_call_search_agent_tool(
    state: dict,
    writer: Any,
    accumulated: dict,
) -> StructuredTool:
    """Build a tool that delegates research to the searchAgent."""

    def call_search_agent(instruction: str) -> str:
        writer({
            "type": "handoff",
            "from": "planner",
            "to": "searchAgent",
            "instruction": instruction,
        })

        result = run_search_agent(state, instruction, writer=writer)

        # Accumulate state updates
        summary = result.get("summary", "")
        if summary:
            prev = accumulated.get("research_notes", "")
            accumulated["research_notes"] = (prev + "\n\n" + summary).strip()

        new_sources = result.get("sources", [])
        if new_sources:
            existing = accumulated.get("sources", [])
            existing_urls = {s.get("url", "") for s in existing}
            for s in new_sources:
                if isinstance(s, str) and s not in existing_urls:
                    existing.append({"url": s, "title": "", "content": "", "score": 0.0})
                    existing_urls.add(s)
            accumulated["sources"] = existing

        handoffs = accumulated.get("agent_handoffs", [])
        handoffs.append({
            "from_agent": "planner",
            "to_agent": "searchAgent",
            "instruction": instruction,
            "result": summary[:500] if summary else "",
        })
        accumulated["agent_handoffs"] = handoffs

        return json.dumps(
            {
                "status": "ok",
                "summary": summary,
                "queries": result.get("queries", []),
                "sources": result.get("sources", []),
            },
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        func=call_search_agent,
        name="CallSearchAgentTool",
        description=(
            "Delegate web/document research to searchAgent. "
            "instruction: a detailed research question. "
            "Use this for facts, documentation, API references, and current information."
        ),
    )


def _build_call_code_agent_tool(
    state: dict,
    writer: Any,
    accumulated: dict,
) -> StructuredTool:
    """Build a tool that delegates implementation to the codeAgent."""

    def call_code_agent(instruction: str) -> str:
        writer({
            "type": "handoff",
            "from": "planner",
            "to": "codeAgent",
            "instruction": instruction,
        })

        result = run_code_agent(state, instruction, writer=writer)

        # Accumulate state updates
        summary = result.get("summary", "")
        accumulated["code_agent_summary"] = summary

        updated_todos = result.get("todos")
        if updated_todos is not None:
            accumulated["todos"] = updated_todos

        handoffs = accumulated.get("agent_handoffs", [])
        handoffs.append({
            "from_agent": "planner",
            "to_agent": "codeAgent",
            "instruction": instruction,
            "result": summary[:500] if summary else "",
        })
        accumulated["agent_handoffs"] = handoffs

        # Forward code agent messages to state
        code_messages = result.get("messages", [])
        if code_messages:
            prev_messages = accumulated.get("messages", [])
            accumulated["messages"] = prev_messages + code_messages

        return json.dumps(
            {
                "status": "ok",
                "summary": summary,
                "todo_count": len(updated_todos) if updated_todos else 0,
            },
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        func=call_code_agent,
        name="CallCodeAgentTool",
        description=(
            "Delegate file/code implementation to codeAgent. "
            "instruction: a detailed implementation task. "
            "The codeAgent has access to the workspace and can read, write, edit files "
            "and run shell commands."
        ),
    )


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


# ---------------------------------------------------------------------------
# Memory injection helpers
# ---------------------------------------------------------------------------

def memory_event(memory: dict, *, node: str) -> dict:
    """Build a custom event for a memory snapshot."""
    return {
        "type": "memory_snapshot",
        "node": node,
        "memory": memory,
    }


def _planner_input(state: dict, memory: dict) -> str:
    """Build the planner's HumanMessage content with memory injected."""
    from mokioclaw.graph.memory import format_layered_memory_for_prompt

    runtime = state.get("runtime") or RuntimeState()
    existing_todos = state.get("todos")

    if not existing_todos:
        return (
            f"Task: {state['task']}\n\n"
            f"Workspace: {runtime.workspace}\n\n"
            f"Create a plan using TodoWriteTool, then delegate work to "
            f"the specialist agents.\n\n"
            f"## Memory Snapshot\n{format_layered_memory_for_prompt(memory)}"
        )

    last_error = state.get("last_error", "Verification failed")
    verification_checks = state.get("verification_checks", [])

    return (
        f"Task: {state['task']}\n\n"
        f"Previous plan summary: {state.get('plan_summary', '')}\n\n"
        f"Previous todos:\n"
        f"{json.dumps(existing_todos, ensure_ascii=False, indent=2)}\n\n"
        f"Verification FAILED.\n"
        f"Error: {last_error}\n\n"
        f"Verification checks:\n"
        f"{json.dumps(verification_checks, ensure_ascii=False, indent=2)}\n\n"
        f"Revise the plan using TodoWriteTool, then delegate only "
        f"the missing fixes to the specialist agents.\n\n"
        f"## Memory Snapshot\n{format_layered_memory_for_prompt(memory)}"
    )


def _code_agent_input(state: dict, instruction: str, memory: dict) -> str:
    """Build the codeAgent's HumanMessage content with memory injected."""
    from mokioclaw.graph.memory import format_layered_memory_for_prompt

    plan_summary = state.get("plan_summary", "")
    research_notes = state.get("research_notes", "")

    return (
        f"Task: {state.get('task', '')}\n\n"
        f"Instruction: {instruction}\n\n"
        f"Plan: {plan_summary}\n\n"
        f"Research notes: {research_notes or '(none)'}\n\n"
        f"## Memory Snapshot\n{format_layered_memory_for_prompt(memory)}"
    )


def _verifier_input(state: dict, cmd_results_summary: str, memory: dict) -> str:
    """Build the verifier's HumanMessage content with memory injected."""
    from mokioclaw.graph.memory import format_layered_memory_for_prompt

    criteria_text = "\n".join(
        f"- {c}" for c in state.get("acceptance_criteria", [])
    )
    agent_output = (
        state.get("code_agent_summary")
        or state.get("last_actor_summary", "")
    )

    return (
        f"Task: {state['task']}\n\n"
        f"Plan: {state.get('plan_summary', '')}\n\n"
        f"Acceptance criteria:\n{criteria_text}\n\n"
        f"Agent output:\n{agent_output}\n\n"
        f"Verification command results:\n{cmd_results_summary}\n\n"
        f"Use FileReadTool and GrepTool to inspect the workspace, "
        f"then output your JSON verdict.\n\n"
        f"## Memory Snapshot\n{format_layered_memory_for_prompt(memory)}"
    )


def _run_command(command: str, cwd: str) -> VerificationResult:
    """Run a single shell command and return a VerificationResult."""
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return VerificationResult(
            command=command,
            ok=proc.returncode == 0,
            exit_code=proc.returncode,
            stdout=proc.stdout.rstrip(),
            stderr=proc.stderr.rstrip(),
        )
    except subprocess.TimeoutExpired:
        return VerificationResult(
            command=command,
            ok=False,
            exit_code=None,
            stdout="",
            stderr="Command timed out after 30s",
        )
    except Exception as exc:
        return VerificationResult(
            command=command,
            ok=False,
            exit_code=None,
            stdout="",
            stderr=str(exc),
        )


def _parse_verifier_output(text: str) -> dict:
    """Try to parse the verifier's JSON output from the response text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "passed": "passed" in text.lower() and "true" in text.lower(),
        "reason": text[:500],
        "checks": [],
        "recommended_next_instruction": "",
    }


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def planner_node(state: MokioGraphState) -> dict:
    """Supervisor planner — creates plans and delegates to specialist agents.

    Tools: TodoWriteTool, CallSearchAgentTool, CallCodeAgentTool.

    Runs a ReAct loop (max 5 iterations).  Each iteration the model may call
    any combination of the three tools.  Sub-agent results are accumulated
    into the state.
    """
    from langgraph.config import get_stream_writer

    runtime = state.get("runtime") or RuntimeState()
    writer = get_stream_writer()

    # Build and inject layered memory
    memory = build_layered_memory(state, node="planner")
    writer(memory_event(memory, node="planner"))

    # Accumulated updates that will be returned at the end
    accumulated: dict[str, Any] = {
        "research_notes": state.get("research_notes", ""),
        "sources": list(state.get("sources") or []),
        "agent_handoffs": list(state.get("agent_handoffs") or []),
        "code_agent_summary": state.get("code_agent_summary", ""),
        "todos": state.get("todos") or [],
        "messages": list(state.get("messages") or []),
    }

    # Build the three tools
    todo_tool = build_todo_write_tool()
    search_tool = _build_call_search_agent_tool(state, writer, accumulated)
    code_tool = _build_call_code_agent_tool(state, writer, accumulated)

    tools = [todo_tool, search_tool, code_tool]
    tool_by_name = {t.name: t for t in tools}

    model = create_model(
        model_name=state.get("model_name", "gpt-5.5")
    ).bind_tools(tools)

    existing_todos = state.get("todos")

    if not existing_todos:
        # --- First run: generate plan ---
        messages = [
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=_planner_input(state, memory)),
        ]
    else:
        # --- Retry: revise plan ---
        messages = [
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=_planner_input(state, memory)),
        ]

    # --- ReAct loop ---
    last_text = ""
    plan_summary = state.get("plan_summary", "")
    todos = accumulated["todos"]
    acceptance_criteria: list[str] = list(state.get("acceptance_criteria") or [])
    verification_commands: list[str] = list(state.get("verification_commands") or [])

    for _ in range(5):
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
            writer({
                "type": "tool_call",
                "node": "planner",
                "name": name,
                "args": args,
            })

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

            # Parse TodoWriteTool result to capture plan updates
            if name == "TodoWriteTool":
                plan_summary = args.get("plan_summary", plan_summary)
                todos = args.get("todos", todos)
                acceptance_criteria = args.get("acceptance_criteria", acceptance_criteria)
                verification_commands = args.get("verification_commands", verification_commands)
                accumulated["todos"] = todos

            # Emit result event
            writer({
                "type": "tool_result",
                "node": "planner",
                "name": name,
                "result": result,
            })

            messages.append(
                ToolMessage(content=result, tool_call_id=tool_id, name=name)
            )

    # --- Return accumulated state updates ---
    return {
        "plan_summary": plan_summary,
        "todos": todos,
        "acceptance_criteria": acceptance_criteria,
        "verification_commands": verification_commands,
        "research_notes": accumulated["research_notes"],
        "sources": accumulated["sources"],
        "agent_handoffs": accumulated["agent_handoffs"],
        "code_agent_summary": accumulated["code_agent_summary"],
        "messages": accumulated["messages"],
        "context_next_node": "verifier",
    }


def actor_node(state: MokioGraphState) -> dict:
    """Execute the plan using the full tool set.

    Runs a ReAct loop (max 10 iterations).  The actor can call TodoUpdateTool
    to mark progress through the plan.

    .. note::

       In stage 3 this node is no longer wired into the graph — the planner
       delegates to codeAgent directly.  Kept for backward compatibility.
    """
    runtime = state.get("runtime") or RuntimeState()
    todos = state.get("todos", [])

    tools = build_tools(runtime) + [build_todo_update_tool(todos)]
    tool_by_name = {t.name: t for t in tools}

    model = create_model(
        model_name=state.get("model_name", "gpt-5.5")
    ).bind_tools(tools)

    plan_summary = state.get("plan_summary", "")
    todos_json = json.dumps(todos, ensure_ascii=False, indent=2)

    plan_context = (
        f"\n\n## Plan\n{plan_summary}\n\n"
        f"## Todos\n```json\n{todos_json}\n```\n"
    )

    messages: list = [
        SystemMessage(content=ACTOR_PROMPT + plan_context),
        HumanMessage(content=state["task"]),
    ]

    last_text = ""

    for _ in range(10):
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

            messages.append(
                ToolMessage(content=result, tool_call_id=tool_id, name=name)
            )

    return {
        "messages": messages,
        "last_actor_summary": last_text,
        "todos": todos,
    }


def verifier_node(state: MokioGraphState) -> dict:
    """Verify the agent's work.

    1. Runs every verification command and collects results.
    2. Uses a read-only LLM agent to inspect files and judge correctness.
    3. Returns passed/failed verdict and structured checks.
    """
    runtime = state.get("runtime") or RuntimeState()

    # Build and inject layered memory
    memory = build_layered_memory(state, node="verifier")
    from langgraph.config import get_stream_writer
    writer = get_stream_writer()
    writer(memory_event(memory, node="verifier"))

    # --- 1. Run verification commands ---
    verification_results: list[VerificationResult] = []
    for cmd in state.get("verification_commands", []):
        result = _run_command(cmd, str(runtime.workspace))
        verification_results.append(result)

    # --- 2. LLM verification with read-only tools ---
    model = create_model(
        model_name=state.get("model_name", "gpt-5.5")
    ).bind_tools(build_read_only_tools(runtime))
    tool_by_name = {
        t.name: t for t in build_read_only_tools(runtime)
    }

    cmd_results_summary = json.dumps(
        [
            {
                "command": r["command"],
                "ok": r["ok"],
                "exit_code": r["exit_code"],
                "stdout": r["stdout"][:500] if r["stdout"] else "",
                "stderr": r["stderr"][:500] if r["stderr"] else "",
            }
            for r in verification_results
        ],
        ensure_ascii=False,
        indent=2,
    )

    criteria_text = "\n".join(
        f"- {c}" for c in state.get("acceptance_criteria", [])
    )

    # Use code_agent_summary if available (stage 3), else fall back to last_actor_summary
    agent_output = state.get("code_agent_summary") or state.get("last_actor_summary", "")

    messages = [
        SystemMessage(content=VERIFIER_PROMPT),
        HumanMessage(
            content=_verifier_input(state, cmd_results_summary, memory)
        ),
    ]

    for _ in range(5):
        response = model.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for call in response.tool_calls:
            tool = tool_by_name.get(call["name"])
            if tool:
                try:
                    result = tool.invoke(call["args"])
                except Exception:
                    result = f"Error: {traceback.format_exc()}"
            else:
                result = f"Error: unknown tool {call['name']!r}"

            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)

            messages.append(
                ToolMessage(content=result, tool_call_id=call["id"], name=call["name"])
            )

    final_text = (
        _extract_text(response)
        if isinstance(response, AIMessage)
        else str(response)
    )
    verdict = _parse_verifier_output(final_text)

    attempts = state.get("attempts", 0) + 1
    passed = verdict.get("passed", False)

    update: dict = {
        "passed": passed,
        "attempts": attempts,
        "verification_results": verification_results,
        "verification_checks": verdict.get("checks", []),
        "messages": messages,
        "context_next_node": "final" if passed else "planner",
    }

    if not passed:
        update["last_error"] = verdict.get(
            "recommended_next_instruction",
            verdict.get("reason", "Verification failed"),
        )
        updated_todos = state.get("todos", [])
        for t in updated_todos:
            if t.get("status") not in ("completed", "blocked"):
                t["status"] = "blocked"
                t["note"] = t.get("note", "") + " [verification failed]"
        update["todos"] = updated_todos

    return update


# ---------------------------------------------------------------------------
# Context monitor
# ---------------------------------------------------------------------------

def context_monitor_node(state: MokioGraphState) -> dict:
    """Estimate token usage and flag whether context compression is needed.

    1. Estimates token count from messages + memory snapshot.
    2. Sets ``context_should_compress`` if the count exceeds the limit.
    3. ``context_next_node`` is set by the upstream node to control routing.
    """
    token_count = _estimate_tokens(state)
    token_limit = state.get("context_token_limit", 50000)
    should_compress = token_count > token_limit

    return {
        "context_token_count": token_count,
        "context_should_compress": should_compress,
        "context_next_node": state.get("context_next_node", "verifier"),
    }


# ---------------------------------------------------------------------------
# Context compressor
# ---------------------------------------------------------------------------

def context_compressor_node(state: MokioGraphState) -> dict:
    """Compress the conversation history to free up context window.

    1. Calls an LLM to summarise the full message history + memory snapshot.
    2. Replaces all messages with a single ``AIMessage`` containing the summary.
    3. Persists the summary to ``HISTORY_SUMMARY.md``.
    4. Truncates long-running state fields and records a compression event.
    """
    from mokioclaw.graph.memory import build_layered_memory, format_layered_memory_for_prompt

    runtime = state.get("runtime") or RuntimeState()

    # --- 1. Build the compression prompt ---
    memory = build_layered_memory(state, node="context_compressor")
    memory_payload = format_layered_memory_for_prompt(memory)

    messages = list(state.get("messages") or [])
    messages_text = _messages_to_text(messages)

    model = create_model(
        model_name=state.get("model_name", "gpt-5.5")
    )

    response = model.invoke([
        SystemMessage(content=CONTEXT_COMPRESSION_PROMPT),
        HumanMessage(
            content=(
                "Compress the following conversation and memory snapshot "
                "into the JSON format described.\n\n"
                f"## Messages\n{messages_text}\n\n"
                f"## Memory Snapshot\n{memory_payload}"
            )
        ),
    ])

    # --- 2. Parse the compression result ---
    raw = _extract_text(response)
    try:
        compressed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                compressed = json.loads(m.group(0))
            except json.JSONDecodeError:
                compressed = {"summary": raw[:2000]}
        else:
            compressed = {"summary": raw[:2000]}

    summary = compressed.get("summary", raw[:2000])

    # --- 3. Replace message history ---
    tokens_before = _estimate_tokens(state)
    # Remove each message individually by its actual id
    remove_ops = [RemoveMessage(id=msg.id) for msg in messages if getattr(msg, "id", None)]
    new_aimessage = AIMessage(
        content=(
            f"[Context compressed]\n\n{summary}\n\n"
            f"Active goal: {compressed.get('active_goal', '')}\n"
            f"Completed: {compressed.get('completed_work', '')}\n"
            f"Open todos: {json.dumps(compressed.get('open_todos', []), ensure_ascii=False)}\n"
            f"Important files: {json.dumps(compressed.get('important_files', []), ensure_ascii=False)}\n"
            f"Next steps: {compressed.get('next_steps', '')}\n"
            f"Risks: {compressed.get('risks', '')}"
        )
    )

    # --- 4. Persist to HISTORY_SUMMARY.md ---
    _write_history_summary(runtime, summary)

    # --- 5. Truncate long-running fields ---
    updated = {
        "messages": remove_ops + [new_aimessage],
        "context_summary": _short_text(summary, 4000),
        "context_token_count": len(summary) // 4,
        "context_should_compress": False,
        "research_notes": _short_text(
            state.get("research_notes", ""), 1200
        ),
        "agent_handoffs": (state.get("agent_handoffs") or [])[-4:],
        "history_summary": summary,
    }

    # --- 6. Record compression event ---
    events = list(state.get("compression_events") or [])
    events.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": "token_limit_exceeded",
        "tokens_before": tokens_before,
        "tokens_after": len(summary) // 4,
    })
    updated["compression_events"] = events

    return updated


# Sentinel value for RemoveMessage — tells LangGraph to clear all messages
REMOVE_ALL_MESSAGES = "remove_all"


def _messages_to_text(messages: list) -> str:
    """Convert a list of LangChain messages to a plain-text transcript."""
    parts: list[str] = []
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            text = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in content
            )
        else:
            text = str(content)
        if text.strip():
            parts.append(f"[{role}] {text[:800]}")
    return "\n".join(parts)


def _write_history_summary(runtime: RuntimeState, summary: str) -> None:
    """Write the compressed summary to HISTORY_SUMMARY.md."""
    path = runtime.resolve("HISTORY_SUMMARY.md")
    try:
        path.write_text(summary, encoding="utf-8")
    except OSError:
        pass  # non-fatal: workspace may not be writable


def _short_text(text: str, limit: int) -> str:
    """Truncate *text* to *limit* characters, appending '...' if needed."""
    if not text or len(text) <= limit:
        return text or ""
    return text[:limit] + "..."


# ---------------------------------------------------------------------------
# Entry workflow: intent router + chat responder
# ---------------------------------------------------------------------------

INTENT_ROUTER_PROMPT = """\
You are the intent router for MokioClaw.

Classify the user's latest input into exactly one route:
- chat: greetings, thanks, identity/help questions, ordinary conceptual Q&A,
  or conversational messages that do not need workspace access.
- workflow: any request that needs creating/editing/reading files, running commands,
  installing packages, searching the web, checking the current project, verifying a
  result, or producing a concrete deliverable.

When session context is provided, use it only to understand whether the latest
input is a continuation of prior coding work. A short follow-up like "继续",
"修一下", or "运行测试" should be workflow if it refers to prior workspace work.

Return only JSON with this shape:
{"route":"chat"|"workflow","reason":"brief reason","confidence":0.0}

If uncertain, choose workflow.
"""

CHAT_RESPONDER_PROMPT = """\
You are MokioClaw's lightweight chat node.

Answer the user directly and concisely. Do not claim that you read files,
searched the web, ran commands, edited files, or inspected the workspace.
If the user asks for work requiring tools or project context, say that it
should be handled by the workflow route.

If session context is provided, you may use the recent conversation summary to
answer conversational follow-ups, but do not invent workspace facts.
"""


def intent_router_node(state: MokioGraphState) -> dict:
    """Classify the user's input as ``"chat"`` or ``"workflow"``.

    Uses a lightweight LLM call with no tools.  Falls back to ``"workflow"``
    if the model output is unparseable or confidence is below 0.55.
    """
    model = create_model(
        model_name=state.get("model_name", "gpt-5.5")
    )
    session_context = state.get("plan_summary", "")  # reused as session context

    messages = [
        SystemMessage(content=INTENT_ROUTER_PROMPT),
        HumanMessage(
            content=(
                f"User input: {state['task']}\n\n"
                f"Session context:\n{session_context or '(none)'}"
            )
        ),
    ]

    response = model.invoke(messages)
    text = _extract_text(response)

    try:
        verdict = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                verdict = json.loads(m.group(0))
            except json.JSONDecodeError:
                verdict = {}
        else:
            verdict = {}

    route = verdict.get("route", "workflow")
    confidence = float(verdict.get("confidence", 0.0))

    if route not in ("chat", "workflow"):
        route = "workflow"
    if confidence < 0.55:
        route = "workflow"

    return {
        "intent_route": route,
        "intent_reason": verdict.get("reason", ""),
        "intent_confidence": confidence,
    }


def chat_responder_node(state: MokioGraphState) -> dict:
    """Reply to a chat message directly, without any tools."""
    model = create_model(
        model_name=state.get("model_name", "gpt-5.5")
    )
    session_context = state.get("plan_summary", "")

    messages = [
        SystemMessage(content=CHAT_RESPONDER_PROMPT),
        HumanMessage(
            content=(
                f"User: {state['task']}\n\n"
                f"Session context:\n{session_context or '(none)'}"
            )
        ),
    ]

    response = model.invoke(messages)
    return {
        "chat_response": _extract_text(response),
        "final_answer": _extract_text(response),
    }


def intent_route_fn(state: MokioGraphState) -> str:
    """Route after intent classification."""
    return "chat_responder" if state.get("intent_route") == "chat" else "planner"


def context_monitor_route(state: MokioGraphState) -> str:
    """Route after context monitor.

    - If the task passed verification → ``"final"``
    - If compression is needed → ``"context_compressor"``
    - Otherwise → the node set in ``context_next_node``
    """
    if state.get("passed"):
        return "final"

    if state.get("context_should_compress"):
        return "context_compressor"

    return state.get("context_next_node", "verifier")


def context_compressor_route(state: MokioGraphState) -> str:
    """Route after context compressor.

    Returns ``context_next_node`` (default ``"verifier"``).
    """
    return state.get("context_next_node", "verifier")


def _estimate_tokens(state: MokioGraphState) -> int:
    """Estimate token count from messages and memory snapshot.

    Uses tiktoken if available, otherwise falls back to ``len(text) // 4``.
    """
    # Collect all text from messages
    messages = state.get("messages") or []
    text_parts: list[str] = []
    for msg in messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)

    # Add memory snapshot
    from mokioclaw.graph.memory import build_layered_memory, format_layered_memory_for_prompt
    memory = build_layered_memory(state)
    text_parts.append(format_layered_memory_for_prompt(memory))

    combined = "\n".join(text_parts)

    # Try tiktoken
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")  # gpt-4o / gpt-5 tokenizer
        return len(enc.encode(combined))
    except Exception:
        pass

    # Fallback
    return len(combined) // 4

def verifier_route(state: MokioGraphState) -> str:
    """Route after verifier: go to final, or loop back to planner for retry."""
    if state.get("passed", False):
        return "final"

    attempts = state.get("attempts", 0)
    max_attempts = state.get("max_attempts", 3)
    if attempts >= max_attempts:
        return "final"

    return "planner"


# ---------------------------------------------------------------------------
# Entry / intent routing
# ---------------------------------------------------------------------------

INTENT_ROUTER_PROMPT = """\
You are the intent router for MokioClaw.

Classify the user's latest input into exactly one route:
- chat: greetings, thanks, identity/help questions, ordinary conceptual Q&A,
  or conversational messages that do not need workspace access.
- workflow: any request that needs creating/editing/reading files, running commands,
  installing packages, searching the web, checking the current project, verifying a
  result, or producing a concrete deliverable.

When session context is provided, use it only to understand whether the latest
input is a continuation of prior coding work. A short follow-up like "继续",
"修一下", or "运行测试" should be workflow if it refers to prior workspace work.

Return only JSON with this shape:
{"route":"chat"|"workflow","reason":"brief reason","confidence":0.0}

If uncertain, choose workflow.
"""

CHAT_RESPONDER_PROMPT = """\
You are MokioClaw's lightweight chat node.

Answer the user directly and concisely. Do not claim that you read files,
searched the web, ran commands, edited files, or inspected the workspace.
If the user asks for work requiring tools or project context, say that it
should be handled by the workflow route.

If session context is provided, you may use the recent conversation summary to
answer conversational follow-ups, but do not invent workspace facts.
"""


def intent_router_node(state: MokioGraphState) -> dict:
    """Classify the user's intent as chat or workflow.

    1. Calls a lightweight LLM with INTENT_ROUTER_PROMPT.
    2. Parses the JSON response for route, reason, confidence.
    3. Defaults to workflow if confidence < 0.55 or parsing fails.
    """
    model = create_model(
        model_name=state.get("model_name", "gpt-5.5")
    )

    response = model.invoke([
        SystemMessage(content=INTENT_ROUTER_PROMPT),
        HumanMessage(
            content=(
                f"User input: {state.get('task', '')}\n\n"
                f"Session context: {state.get('plan_summary', '')[:200]}"
            )
        ),
    ])

    raw = _extract_text(response)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = {}

    route = parsed.get("route", "workflow")
    reason = parsed.get("reason", "")
    confidence = float(parsed.get("confidence", 0))

    if route not in ("chat", "workflow"):
        route = "workflow"
    if confidence < 0.55:
        route = "workflow"

    return {
        "intent_route": route,
        "intent_reason": reason,
        "intent_confidence": confidence,
    }


def chat_responder_node(state: MokioGraphState) -> dict:
    """Lightweight chat response — no tools, direct LLM answer."""
    model = create_model(
        model_name=state.get("model_name", "gpt-5.5")
    )

    response = model.invoke([
        SystemMessage(content=CHAT_RESPONDER_PROMPT),
        HumanMessage(
            content=(
                f"User: {state.get('task', '')}\n\n"
                f"Session context: {state.get('plan_summary', '')[:200]}"
            )
        ),
    ])

    chat_response = _extract_text(response)

    return {
        "chat_response": chat_response,
        "final_answer": chat_response,
    }


def intent_route_fn(state: MokioGraphState) -> str:
    """Route after intent classification."""
    return "chat_responder" if state.get("intent_route") == "chat" else "planner"