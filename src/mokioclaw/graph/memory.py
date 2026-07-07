"""Layered memory system for MokioClaw.

Three layers:
  1. Rules — fixed, immutable rules injected into every prompt.
  2. Working Memory — current task state (plan, todos, research, handoffs).
  3. History Summary Store — compressed durable context from NOTEPAD.md
     and HISTORY_SUMMARY.md.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mokioclaw.core.state import RuntimeState

# ---------------------------------------------------------------------------
# Layer 1: Rules (fixed)
# ---------------------------------------------------------------------------

RULES_LAYER: dict[str, Any] = {
    "scope": "workspace",
    "storage": "internal",
    "rules": [
        "Work inside the current workspace only.",
        "Use paths relative to the workspace; do not prefix paths with workspace/.",
        "Keep durable task context outside the raw messages transcript when possible.",
        "Treat TODO.md as working plan state, NOTEPAD.md as durable notes, "
        "and HISTORY_SUMMARY.md as compressed history.",
        "Do not expose memory write tools to agents; layered memory is "
        "assembled by the runtime.",
    ],
}

# ---------------------------------------------------------------------------
# Layer 2 + 3: working memory + history summary store
# ---------------------------------------------------------------------------

def build_layered_memory(
    state: dict,
    *,
    node: str = "graph",
) -> dict:
    """Assemble the three-layer memory snapshot from state and workspace files.

    Args:
        state: The current ``MokioGraphState`` (or compatible dict).
        node: Label for the node requesting the snapshot (e.g. ``"planner"``).

    Returns:
        ``{rules, working_memory, history_summary_store}``
    """
    runtime = state.get("runtime") or RuntimeState()

    notepad = _read_notepad(runtime)
    history = _read_history_summary(runtime)

    session_id = state.get("session_id", str(uuid.uuid4())[:8])
    session_turn = state.get("session_turn", 0) + 1

    # --- Working memory ---
    sources_compact: list[dict] = []
    for s in (state.get("sources") or [])[:10]:
        sources_compact.append({
            "title": s.get("title", ""),
            "url": s.get("url", ""),
        })

    working_memory: dict[str, Any] = {
        "node": node,
        "task": state.get("task", ""),
        "session_id": session_id,
        "session_turn": session_turn,
        "plan_summary": state.get("plan_summary", ""),
        "todos": state.get("todos", []),
        "acceptance_criteria": state.get("acceptance_criteria", []),
        "verification_commands": state.get("verification_commands", []),
        "research_notes": _short_text(state.get("research_notes", ""), 1600),
        "sources": sources_compact,
        "agent_handoffs": _trim_handoffs(state.get("agent_handoffs", [])),
        "code_agent_summary": _short_text(
            state.get("code_agent_summary", ""), 1000
        ),
        "verifier_summary": _short_text(
            state.get("verifier_summary", ""), 1000
        ),
        "last_error": _short_text(state.get("last_error", ""), 1400),
        "attempts": state.get("attempts", 0),
        "max_attempts": state.get("max_attempts", 3),
    }

    # --- History summary store ---
    history_summary_store: dict[str, Any] = {
        "history_path": "HISTORY_SUMMARY.md",
        "history_exists": history.get("exists", False),
        "history_summary": _short_text(history.get("content", ""), 2200),
        "notepad_path": "NOTEPAD.md",
        "notepad_exists": notepad.get("exists", False),
        "notepad": _short_text(notepad.get("content", ""), 1800),
        "context_summary": _short_text(
            state.get("context_summary", ""), 1600
        ),
        "compression_events": (state.get("compression_events") or [])[-3:],
    }

    return {
        "rules": dict(RULES_LAYER),
        "working_memory": working_memory,
        "history_summary_store": history_summary_store,
    }


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_layered_memory_for_prompt(memory: dict) -> str:
    """Render the layered memory snapshot as a JSON string for prompt injection."""
    return json.dumps(memory, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _read_notepad(runtime: RuntimeState) -> dict:
    """Read NOTEPAD.md from the workspace."""
    path = runtime.resolve("NOTEPAD.md")
    try:
        content = path.read_text(encoding="utf-8")
        return {"exists": True, "content": content}
    except (OSError, FileNotFoundError):
        return {"exists": False, "content": ""}


def _read_history_summary(runtime: RuntimeState) -> dict:
    """Read HISTORY_SUMMARY.md from the workspace."""
    path = runtime.resolve("HISTORY_SUMMARY.md")
    try:
        content = path.read_text(encoding="utf-8")
        return {"exists": True, "content": content}
    except (OSError, FileNotFoundError):
        return {"exists": False, "content": ""}


# ---------------------------------------------------------------------------
# Truncation helpers
# ---------------------------------------------------------------------------

def _short_text(text: str, limit: int) -> str:
    """Truncate *text* to *limit* characters, appending '…' if needed."""
    if not text or len(text) <= limit:
        return text or ""
    return text[:limit] + "…"


def _trim_handoffs(handoffs: list[dict]) -> list[dict]:
    """Keep only the most recent 6 handoff records."""
    return handoffs[-6:] if len(handoffs) > 6 else handoffs


# ---------------------------------------------------------------------------
# Compression stub (extended in later stages)
# ---------------------------------------------------------------------------

def should_compress(state: dict) -> bool:
    """Check whether the context has exceeded the token limit."""
    count = state.get("context_token_count", 0)
    limit = state.get("context_token_limit", 128_000)
    return count >= limit


def record_compression_event(
    state: dict,
    reason: str,
    tokens_before: int,
    tokens_after: int,
) -> dict:
    """Append a compression event to the state and return the new event."""
    event: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
    }
    events = list(state.get("compression_events") or [])
    events.append(event)
    return event