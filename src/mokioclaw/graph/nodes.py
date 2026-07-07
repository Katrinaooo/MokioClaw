"""LangGraph nodes: planner, actor, verifier — plus the verifier route."""

import json
import re
import subprocess
import traceback
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from mokioclaw.agents.code_agent import run_code_agent
from mokioclaw.agents.search_agent import run_search_agent
from mokioclaw.core.state import RuntimeState
from mokioclaw.graph.state import MokioGraphState, VerificationResult
from mokioclaw.prompts.stage2 import ACTOR_PROMPT
from mokioclaw.prompts.stage3 import PLANNER_PROMPT, VERIFIER_PROMPT
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
            HumanMessage(
                content=(
                    f"Task: {state['task']}\n\n"
                    f"Workspace: {runtime.workspace}\n\n"
                    "Create a plan using TodoWriteTool, then delegate work to "
                    "the specialist agents."
                )
            ),
        ]
    else:
        # --- Retry: revise plan ---
        last_error = state.get("last_error", "Verification failed")
        verification_checks = state.get("verification_checks", [])

        messages = [
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(
                content=(
                    f"Task: {state['task']}\n\n"
                    f"Previous plan summary: {state.get('plan_summary', '')}\n\n"
                    f"Previous todos:\n"
                    f"{json.dumps(existing_todos, ensure_ascii=False, indent=2)}\n\n"
                    f"Verification FAILED.\n"
                    f"Error: {last_error}\n\n"
                    f"Verification checks:\n"
                    f"{json.dumps(verification_checks, ensure_ascii=False, indent=2)}\n\n"
                    "Revise the plan using TodoWriteTool, then delegate only "
                    "the missing fixes to the specialist agents."
                )
            ),
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
            content=(
                f"Task: {state['task']}\n\n"
                f"Plan: {state.get('plan_summary', '')}\n\n"
                f"Acceptance criteria:\n{criteria_text}\n\n"
                f"Agent output:\n{agent_output}\n\n"
                f"Verification command results:\n{cmd_results_summary}\n\n"
                "Use FileReadTool and GrepTool to inspect the workspace, "
                "then output your JSON verdict."
            )
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
# Route
# ---------------------------------------------------------------------------

def verifier_route(state: MokioGraphState) -> str:
    """Route after verifier: go to final, or loop back to planner for retry."""
    if state.get("passed", False):
        return "final"

    attempts = state.get("attempts", 0)
    max_attempts = state.get("max_attempts", 3)
    if attempts >= max_attempts:
        return "final"

    return "planner"