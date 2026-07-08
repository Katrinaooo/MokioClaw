"""Session management for MokioClaw — persistent conversation state."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SESSION_ROOT = ".mokioclaw/session"
SESSION_FILE = "session.json"
SESSION_SUMMARY_FILE = "SESSION_SUMMARY.md"
MAX_SESSION_CONTEXT = 7000
MAX_TURN_CONTENT = 4000


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def load_or_create_session(workspace: Path) -> dict:
    """Load an existing session.json or create a new one.

    Returns:
        The session dict with keys ``session_id``, ``turn_index``,
        ``recent_turns``, ``created_at``, ``updated_at``.
    """
    session_dir = workspace / SESSION_ROOT
    session_path = session_dir / SESSION_FILE

    if session_path.exists():
        try:
            session = json.loads(session_path.read_text(encoding="utf-8"))
            # Ensure all required keys exist
            session.setdefault("session_id", _new_session_id())
            session.setdefault("turn_index", 0)
            session.setdefault("recent_turns", [])
            session.setdefault("created_at", "")
            session.setdefault("updated_at", "")
            return session
        except (OSError, json.JSONDecodeError):
            pass

    # Create new session
    now = _now()
    return {
        "session_id": _new_session_id(),
        "turn_index": 0,
        "recent_turns": [],
        "created_at": now,
        "updated_at": now,
    }


# ---------------------------------------------------------------------------
# Append turns
# ---------------------------------------------------------------------------

def append_user_turn(session: dict, content: str) -> int:
    """Record a user turn and return the turn number."""
    turn = int(session.get("turn_index", 0)) + 1
    session["turn_index"] = turn

    session.setdefault("recent_turns", []).append({
        "turn": turn,
        "role": "user",
        "content": content[:MAX_TURN_CONTENT],
        "timestamp": _now(),
    })

    _trim_turns(session)
    return turn


def append_assistant_turn(
    session: dict,
    *,
    turn: int,
    route: str,
    content: str,
    summary: str = "",
) -> None:
    """Record an assistant turn."""
    session.setdefault("recent_turns", []).append({
        "turn": turn,
        "role": "assistant",
        "route": route,
        "content": content[:MAX_TURN_CONTENT],
        "summary": summary[:MAX_TURN_CONTENT],
        "timestamp": _now(),
    })

    _trim_turns(session)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_session(workspace: Path, session: dict) -> dict:
    """Persist session.json and generate SESSION_SUMMARY.md."""
    session["updated_at"] = _now()

    session_dir = workspace / SESSION_ROOT
    session_dir.mkdir(parents=True, exist_ok=True)

    # Write session.json
    tmp = session_dir / (SESSION_FILE + ".tmp")
    tmp.write_text(
        json.dumps(session, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(session_dir / SESSION_FILE)

    # Write SESSION_SUMMARY.md
    summary_path = session_dir / SESSION_SUMMARY_FILE
    summary_path.write_text(
        _build_session_summary_markdown(session),
        encoding="utf-8",
    )

    return session


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_session_context(workspace: Path, session: dict | None = None) -> str:
    """Build a session context string for the intent router and chat responder.

    Includes session metadata, recent workspace files, and the last 10 turns.
    Total length is capped at ``MAX_SESSION_CONTEXT``.
    """
    if session is None:
        session = load_or_create_session(workspace)

    parts: list[str] = []

    # Session metadata
    parts.append(
        f"Session: {session.get('session_id', '?')} "
        f"(turn {session.get('turn_index', 0)})"
    )

    # Workspace file manifest
    files = _list_recent_files(workspace, limit=30)
    if files:
        parts.append("Workspace files:\n" + "\n".join(f"- {f}" for f in files))

    # Recent turns
    turns = session.get("recent_turns") or []
    if turns:
        recent = turns[-10:]
        parts.append("Recent conversation:")
        for t in recent:
            role = t.get("role", "?")
            route = t.get("route", "")
            content = t.get("content", "") or t.get("summary", "")
            label = f"{role}/{route}" if route else role
            parts.append(f"  [{label}] {content[:200]}")

    combined = "\n\n".join(parts)
    if len(combined) > MAX_SESSION_CONTEXT:
        combined = combined[:MAX_SESSION_CONTEXT] + "…"

    return combined


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _new_session_id() -> str:
    return f"session-{uuid.uuid4().hex[:12]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trim_turns(session: dict) -> None:
    """Keep at most 20 most recent turns."""
    turns = session.get("recent_turns") or []
    if len(turns) > 20:
        session["recent_turns"] = turns[-20:]


def _list_recent_files(workspace: Path, limit: int = 30) -> list[str]:
    """List workspace files, excluding hidden and .mokioclaw."""
    files: list[str] = []
    try:
        for entry in sorted(workspace.rglob("*")):
            if entry.is_file() and ".mokioclaw" not in entry.parts:
                name = entry.name
                if not name.startswith("."):
                    try:
                        rel = entry.relative_to(workspace)
                        files.append(str(rel).replace("\\", "/"))
                    except ValueError:
                        pass
    except OSError:
        pass
    return files[:limit]


def _build_session_summary_markdown(session: dict) -> str:
    """Generate a human-readable session summary."""
    sid = session.get("session_id", "?")
    turns = session.get("recent_turns") or []
    created = session.get("created_at", "")[:19]
    updated = session.get("updated_at", "")[:19]

    lines: list[str] = [
        f"# Session {sid}",
        "",
        f"**Created:** {created}",
        f"**Updated:** {updated}",
        f"**Turns:** {len(turns)}",
        "",
        "## Turns",
        "",
    ]

    for t in turns:
        turn_num = t.get("turn", "?")
        role = t.get("role", "?")
        route = t.get("route", "")
        content = t.get("content", "") or t.get("summary", "")
        label = f"{role}/{route}" if route else role
        lines.append(f"### Turn {turn_num} ({label})")
        lines.append("")
        lines.append(content[:500])
        lines.append("")

    return "\n".join(lines)