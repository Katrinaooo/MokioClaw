"""Checkpoint save and restore for MokioClaw sessions."""

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mokioclaw.core.state import RuntimeState

VALID_CHECKPOINT_MODES = {"light", "strict", "off"}


def normalize_checkpoint_mode(mode: str | None) -> str:
    """Normalise a checkpoint mode string.

    Defaults to ``"light"``; invalid values fall back to ``"off"``.
    """
    if mode and mode in VALID_CHECKPOINT_MODES:
        return mode
    return "off"


class CheckpointManager:
    """Save and restore checkpoints for a MokioClaw session."""

    def __init__(self, runtime: RuntimeState, task: str = "") -> None:
        self.workspace = runtime.workspace
        self.mode = normalize_checkpoint_mode(runtime.checkpoint_mode)
        self.task = task
        self.root = self.workspace / ".mokioclaw" / "checkpoints"

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def save(
        self,
        state: dict,
        *,
        status: str = "running",
        latest_node: str | None = None,
        event: dict | None = None,
    ) -> dict | None:
        """Save a checkpoint.

        - **light**: checkpoint.json + RECOVERY.md + git snapshot.
        - **strict**: additionally state.json + events.jsonl.
        - **off**: no-op, returns ``None``.
        """
        if not self.enabled:
            return None

        self.root.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()

        manifest = _build_workspace_manifest(self.workspace)
        git_commit = _snapshot_workspace_git(self.workspace)

        if self.mode == "strict":
            _save_json(self.root / "state.json", _serialize_state(state))
            if event is not None:
                _append_jsonl(self.root / "events.jsonl", {
                    "timestamp": now,
                    "node": latest_node,
                    "event": event,
                })

        checkpoint: dict[str, Any] = {
            "timestamp": now,
            "task": self.task,
            "status": status,
            "latest_node": latest_node,
            "mode": self.mode,
            "attempts": state.get("attempts", 0),
            "max_attempts": state.get("max_attempts", 3),
            "plan_summary": state.get("plan_summary", ""),
            "passed": state.get("passed", False),
            "todos": state.get("todos") or [],
            "git_commit": git_commit,
            "workspace_manifest": manifest,
        }
        _save_json(self.root / "checkpoint.json", checkpoint)

        recovery = build_recovery_markdown({
            **checkpoint,
            "workspace": str(self.workspace),
        })
        (self.root / "RECOVERY.md").write_text(recovery, encoding="utf-8")

        return {
            "type": "checkpoint_saved",
            "timestamp": now,
            "mode": self.mode,
            "node": latest_node,
            "git_commit": git_commit,
        }

    @classmethod
    def load_resume_inputs(
        cls,
        runtime: RuntimeState,
        task: str | None = None,
        max_attempts: int = 3,
    ) -> tuple[dict, dict]:
        """Attempt to restore from a previous checkpoint.

        Returns:
            ``(inputs, resume_event)`` — *inputs* is suitable for
            ``graph.stream()`` and *resume_event* describes what was found.
        """
        root = runtime.workspace / ".mokioclaw" / "checkpoints"
        checkpoint_path = root / "checkpoint.json"

        if not checkpoint_path.exists():
            return (
                {"task": task or "", "runtime": runtime, "max_attempts": max_attempts},
                {"type": "resume", "found": False, "reason": "no checkpoint found"},
            )

        checkpoint = _load_json(checkpoint_path)
        if checkpoint is None:
            return (
                {"task": task or "", "runtime": runtime, "max_attempts": max_attempts},
                {"type": "resume", "found": False, "reason": "checkpoint unreadable"},
            )

        git_commit = checkpoint.get("git_commit", "")
        if git_commit:
            _restore_git_snapshot(runtime.workspace, git_commit)

        inputs: dict[str, Any] = {
            "task": task or checkpoint.get("task", ""),
            "runtime": runtime,
            "max_attempts": checkpoint.get("max_attempts", max_attempts),
            "attempts": checkpoint.get("attempts", 0),
            "plan_summary": checkpoint.get("plan_summary", ""),
            "todos": checkpoint.get("todos") or [],
            "passed": checkpoint.get("passed", False),
        }

        return (
            inputs,
            {
                "type": "resume",
                "found": True,
                "timestamp": checkpoint.get("timestamp", ""),
                "status": checkpoint.get("status", "running"),
                "latest_node": checkpoint.get("latest_node", ""),
                "git_commit": git_commit,
            },
        )


def resume_command(workspace: Path) -> str:
    """Generate the shell command to resume a session."""
    return f"mokioclaw run --resume {workspace}"


def build_recovery_markdown(payload: dict) -> str:
    """Generate the contents of RECOVERY.md."""
    ws = payload.get("workspace", "?")
    task = payload.get("task", "?")
    status = payload.get("status", "?")
    node = payload.get("latest_node", "?")
    git_commit = payload.get("git_commit", "(none)")
    manifest = payload.get("workspace_manifest", [])
    todos = payload.get("todos") or []

    files = "\n".join(f"- {f}" for f in manifest[:30]) or "(empty)"
    todo_lines = "\n".join(
        f"- [{t.get('status', '?')}] {t.get('content', '')}"
        for t in todos
    ) or "(none)"

    return (
        f"# MokioClaw Recovery\n\n"
        f"**Task:** {task}\n\n"
        f"**Status:** {status}\n"
        f"**Last node:** {node}\n"
        f"**Git commit:** `{git_commit}`\n\n"
        f"## Resume\n\n"
        f"```bash\n{resume_command(Path(ws))}\n```\n\n"
        f"## Todos\n\n{todo_lines}\n\n"
        f"## Workspace files\n\n{files}\n"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_workspace_manifest(workspace: Path) -> list[str]:
    """List tracked files in the workspace, relative to *workspace*."""
    files: list[str] = []
    try:
        for entry in workspace.rglob("*"):
            if entry.is_file() and ".mokioclaw" not in entry.parts:
                try:
                    rel = entry.relative_to(workspace)
                    files.append(str(rel).replace("\\", "/"))
                except ValueError:
                    pass
    except OSError:
        pass
    return sorted(files)[:100]


def _snapshot_workspace_git(workspace: Path) -> str:
    """Stage all files and create a git commit. Returns commit hash or ''."""
    try:
        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=10,
        )
        result = subprocess.run(
            ["git", "commit", "-m", f"checkpoint {int(time.time())}", "--allow-empty"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


def _restore_git_snapshot(workspace: Path, commit: str) -> bool:
    """Restore workspace files to *commit*."""
    try:
        subprocess.run(
            ["git", "checkout", commit, "--", "."],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False


def _serialize_state(state: dict) -> dict:
    """Convert a graph state dict to a JSON-serialisable dict."""
    out: dict[str, Any] = {}
    for key, value in state.items():
        if key == "runtime":
            out[key] = {"workspace": str(value.workspace)}
        elif key == "messages":
            msgs = []
            for m in value or []:
                msgs.append({
                    "type": getattr(m, "type", "unknown"),
                    "content": str(getattr(m, "content", ""))[:500],
                    "id": getattr(m, "id", ""),
                })
            out[key] = msgs
        elif key in ("todos", "verification_results", "verification_checks",
                     "sources", "agent_handoffs", "compression_events"):
            out[key] = value
        elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
            out[key] = value
        else:
            out[key] = str(value)[:200]
    return out


def _save_json(path: Path, data: dict) -> None:
    """Atomically write *data* as JSON to *path*."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path) -> dict | None:
    """Read JSON from *path*, returning ``None`` on failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _append_jsonl(path: Path, record: dict) -> None:
    """Append a line of JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")