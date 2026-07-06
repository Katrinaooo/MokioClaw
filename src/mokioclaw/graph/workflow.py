"""LangGraph workflow builder — assembles the plan→execute→verify pipeline."""

import json

from langgraph.graph import END, START, StateGraph

from mokioclaw.graph.nodes import (
    actor_node,
    planner_node,
    verifier_node,
    verifier_route,
)
from mokioclaw.graph.state import MokioGraphState


def final_node(state: MokioGraphState) -> dict:
    """Format the final result as a user-facing summary.

    Reads the verification outcome, todo status, and plan summary to produce
    a ``final_answer`` string.  No LLM call — pure formatting.
    """
    passed = state.get("passed", False)
    plan_summary = state.get("plan_summary", "")
    todos = state.get("todos", [])
    attempts = state.get("attempts", 0)
    verification_checks = state.get("verification_checks", [])
    last_error = state.get("last_error", "")

    # --- Build todo summary ---
    todo_lines: list[str] = []
    for t in todos:
        icon = {"completed": "[x]", "blocked": "[!]", "in_progress": "[-]", "pending": "[ ]"}.get(
            t.get("status", "pending"), "[ ]"
        )
        note = f" — {t['note']}" if t.get("note") else ""
        todo_lines.append(f"  {icon} {t['content']}{note}")

    todo_summary = "\n".join(todo_lines) if todo_lines else "  (none)"

    # --- Build check summary ---
    check_lines: list[str] = []
    for c in verification_checks:
        icon = "PASS" if c.get("passed") else "FAIL"
        check_lines.append(f"  [{icon}] {c.get('name', '?')}: {c.get('detail', '')}")

    check_summary = "\n".join(check_lines) if check_lines else "  (none)"

    # --- Build final answer ---
    if passed:
        final_answer = (
            f"## ✅ Task Completed\n\n"
            f"**Plan:** {plan_summary}\n\n"
            f"**Attempts:** {attempts}\n\n"
            f"### Todos\n{todo_summary}\n\n"
            f"### Verification Checks\n{check_summary}"
        )
    else:
        final_answer = (
            f"## ❌ Task Not Completed\n\n"
            f"**Plan:** {plan_summary}\n\n"
            f"**Attempts:** {attempts} (max {state.get('max_attempts', 3)})\n\n"
            f"**Last error:** {last_error}\n\n"
            f"### Todos\n{todo_summary}\n\n"
            f"### Verification Checks\n{check_summary}"
        )

    return {"final_answer": final_answer}


def build_workflow():
    """Build and compile the MokioClaw LangGraph.

    Graph structure::

        START → planner → actor → verifier ──passed──→ final → END
                                ↑        │
                                │ failed & attempts < max
                                └──────────────────────┘
    """
    graph = StateGraph(MokioGraphState)

    graph.add_node("planner", planner_node)
    graph.add_node("actor", actor_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("final", final_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "actor")
    graph.add_edge("actor", "verifier")
    graph.add_conditional_edges(
        "verifier",
        verifier_route,
        {
            "final": "final",
            "planner": "planner",
        },
    )
    graph.add_edge("final", END)

    return graph.compile()