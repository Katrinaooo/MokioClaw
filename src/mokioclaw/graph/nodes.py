"""LangGraph nodes: planner, actor, verifier — plus the verifier route."""

import json
import re
import subprocess
import traceback
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from mokioclaw.core.state import RuntimeState
from mokioclaw.graph.state import MokioGraphState, VerificationResult
from mokioclaw.providers.openai_provider import create_model
from mokioclaw.tools.file_tools import build_file_read_tool
from mokioclaw.tools.grep_tool import build_grep_tool
from mokioclaw.tools.registry import build_tools

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PLANNER_PROMPT = """\
You are the planner in MokioClaw's workflow. Your job is to create a detailed,
executable plan for the user's task.

Break the task down into a structured plan.  You MUST call the TodoWriteTool
with your plan — do not output plain text.

Rules for the plan:
- Each todo item must be concrete and actionable.
- Acceptance criteria must be specific and verifiable
  (e.g. "file main.py exists and runs without error").
- Verification commands are shell commands that can be run to check success.
  Use relative paths — the workspace is already the current directory.
- Keep verification commands simple: ls, cat, python -c "...", test -f, etc.
"""

PLANNER_REVISE_PROMPT = """\
You are the planner in MokioClaw's workflow.  The previous plan failed
verification.  Your job is to revise the plan based on the error feedback.

Focus on fixing the specific issues identified by the verifier.  You MUST call
the TodoWriteTool with your revised plan.
"""

ACTOR_PROMPT = """\
You are the actor in MokioClaw's workflow.  Execute the plan step by step.

You have access to a todo list.  Use TodoUpdateTool to mark each item as
"in_progress" when you start it and "completed" when you finish.
If something cannot be done, mark it "blocked" with a note.

Rules:
- Use FileWriteTool for new files.
- Use FileReadTool before editing existing files.
- Use FileEditTool for focused edits — old_text must be unique in the file.
- Use BashTool to run commands and test results.
- BashTool already runs inside the workspace. Use relative paths,
  never "cd /workspace".
- Work through the todos in order. Do not skip items without explanation.
- End with a concise summary of files changed and commands run.
"""

VERIFIER_PROMPT = """\
You are the verifier in MokioClaw's workflow.  Your job is to check whether the
actor successfully completed the plan.

You have read-only tools (FileReadTool, GrepTool).  Inspect the workspace to
verify each acceptance criterion and check the actor's work.

After inspection, output a JSON object with this structure:

{
  "passed": true,
  "reason": "All criteria met. Files are correct and tests pass.",
  "checks": [
    {"name": "File exists", "passed": true, "detail": "main.py found"},
    {"name": "Syntax valid", "passed": true, "detail": "python -c 'import main' succeeds"}
  ],
  "recommended_next_instruction": ""
}

If something is wrong, set "passed": false, explain in "reason", and provide a
specific "recommended_next_instruction" that tells the planner what to fix.
"""

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
    # Try the whole text first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON block
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback: heuristic
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
    """Generate or revise the execution plan.

    - First run (no existing todos): create a plan from scratch.
    - Retry (verification failed): revise the plan based on last_error.
    """
    runtime = state.get("runtime") or RuntimeState()
    model = create_model().bind_tools([build_todo_write_tool()])

    existing_todos = state.get("todos")

    if not existing_todos:
        # --- First run: generate plan ---
        messages = [
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(
                content=(
                    f"Task: {state['task']}\n\n"
                    f"Workspace: {runtime.workspace}\n\n"
                    "List the files in the workspace first (use BashTool via the actor) "
                    "if you need to understand the existing state. "
                    "For now, create a plan assuming the workspace may be empty."
                )
            ),
        ]
    else:
        # --- Retry: revise plan ---
        last_error = state.get("last_error", "Verification failed")
        verification_checks = state.get("verification_checks", [])

        messages = [
            SystemMessage(content=PLANNER_REVISE_PROMPT),
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
                    "Please revise the plan to fix the issues identified above."
                )
            ),
        ]

    response = model.invoke(messages)

    # Parse the TodoWriteTool call
    if response.tool_calls:
        call = response.tool_calls[0]
        args = call["args"]
        return {
            "plan_summary": args.get("plan_summary", ""),
            "todos": args.get("todos", []),
            "acceptance_criteria": args.get("acceptance_criteria", []),
            "verification_commands": args.get("verification_commands", []),
            "messages": [response],
        }

    # Fallback: model didn't use the tool — try to parse from text
    text = _extract_text(response)
    fallback = _parse_verifier_output(text)
    return {
        "plan_summary": fallback.get("reason", text[:200]),
        "todos": fallback.get("todos", []),
        "acceptance_criteria": fallback.get("acceptance_criteria", []),
        "verification_commands": fallback.get("verification_commands", []),
        "messages": [response],
    }


def actor_node(state: MokioGraphState) -> dict:
    """Execute the plan using the full tool set.

    Runs a ReAct loop (max 10 iterations).  The actor can call TodoUpdateTool
    to mark progress through the plan.
    """
    runtime = state.get("runtime") or RuntimeState()
    todos = state.get("todos", [])

    tools = build_tools(runtime) + [build_todo_update_tool(todos)]
    tool_by_name = {t.name: t for t in tools}

    model = create_model().bind_tools(tools)

    plan_summary = state.get("plan_summary", "")
    todos_json = json.dumps(todos, ensure_ascii=False, indent=2)

    plan_context = (
        f"\n\n## Plan\n{plan_summary}\n\n"
        f"## Todos\n```json\n{todos_json}\n```\n"
    )

    messages: list = list(state.get("messages") or [])
    if not messages:
        messages = [
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

            messages.append(ToolMessage(content=result, tool_call_id=tool_id))

    return {
        "messages": messages,
        "last_actor_summary": last_text,
        "todos": todos,  # mutated in-place by TodoUpdateTool
    }


def verifier_node(state: MokioGraphState) -> dict:
    """Verify the actor's work.

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
    model = create_model().bind_tools(build_read_only_tools(runtime))
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

    messages = [
        SystemMessage(content=VERIFIER_PROMPT),
        HumanMessage(
            content=(
                f"Task: {state['task']}\n\n"
                f"Plan: {state.get('plan_summary', '')}\n\n"
                f"Acceptance criteria:\n{criteria_text}\n\n"
                f"Actor's final output:\n{state.get('last_actor_summary', '')}\n\n"
                f"Verification command results:\n{cmd_results_summary}\n\n"
                "Use FileReadTool and GrepTool to inspect the workspace, "
                "then output your JSON verdict."
            )
        ),
    ]

    # Single ReAct loop for the verifier (it may call read-only tools)
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
                ToolMessage(content=result, tool_call_id=call["id"])
            )

    # Parse the final response
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
        # Mark incomplete todos as blocked
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