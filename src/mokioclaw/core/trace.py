"""Execution trace recorder for MokioClaw sessions."""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mokioclaw.core.state import RuntimeState

VALID_TRACE_MODES = {"on", "off"}


def normalize_trace_mode(mode: str | None) -> str:
    """Normalise a trace mode string.

    Defaults to ``"on"``; invalid values fall back to ``"off"``.
    """
    if mode and mode in VALID_TRACE_MODES:
        return mode
    return "off"


class TraceRecorder:
    """Record execution traces for a MokioClaw session."""

    def __init__(self, runtime: RuntimeState, task: str = "") -> None:
        self.workspace = runtime.workspace
        self.mode = normalize_trace_mode(runtime.trace_mode)
        self.trace_id = runtime.trace_id or _new_trace_id()
        self.task = task
        self.root = self.workspace / ".mokioclaw" / "traces" / self.trace_id

        self.node_visits: dict[str, int] = {}
        self.tool_calls = 0
        self.failed_tool_calls = 0
        self.approval_count = 0
        self.checkpoint_count = 0
        self.handoff_count = 0
        self._started_at = ""
        self._timeline: list[dict] = []

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def start(
        self,
        inputs: dict,
        *,
        resumed: bool = False,
        resume_event: dict | None = None,
    ) -> dict | None:
        """Record a ``run_start`` event."""
        if not self.enabled:
            return None

        self._started_at = datetime.now(timezone.utc).isoformat()
        event = {
            "type": "run_start",
            "trace_id": self.trace_id,
            "task": self.task,
            "started_at": self._started_at,
            "resumed": resumed,
            "resume_event": resume_event,
        }
        self._emit(event)
        return event

    def record_custom_event(self, event: dict) -> None:
        """Record a custom (writer-emitted) event and update counters."""
        if not self.enabled:
            return

        etype = event.get("type", "")

        if etype == "tool_call":
            self.tool_calls += 1
        elif etype == "tool_result":
            if not event.get("ok", True):
                self.failed_tool_calls += 1
            if event.get("requires_approval"):
                self.approval_count += 1
        elif etype == "handoff":
            self.handoff_count += 1
        elif etype == "checkpoint_saved":
            self.checkpoint_count += 1

        self._emit(event)

    def record_graph_update(self, event: dict) -> None:
        """Record a node update event and track visit counts."""
        if not self.enabled:
            return

        node = event.get("node", "?")
        self.node_visits[node] = self.node_visits.get(node, 0) + 1
        self._emit(event)

    def end(
        self,
        *,
        status: str,
        latest_node: str | None,
        final_state: dict | None = None,
    ) -> dict | None:
        """Finalise the trace, writing trace.json and timeline.md."""
        if not self.enabled:
            return None

        ended_at = datetime.now(timezone.utc).isoformat()
        started = self._started_at or ended_at
        duration_ms = _duration_ms(started, ended_at)

        total = len(self._timeline)
        head = self._timeline[:20]
        tail = self._timeline[-80:] if total > 100 else self._timeline[20:]
        omitted = max(0, total - 100)

        summary: dict[str, Any] = {
            "trace_id": self.trace_id,
            "task": self.task,
            "status": status,
            "started_at": started,
            "ended_at": ended_at,
            "duration_ms": duration_ms,
            "node_visits": self.node_visits,
            "tool_calls": self.tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "approval_count": self.approval_count,
            "checkpoint_count": self.checkpoint_count,
            "handoff_count": self.handoff_count,
            "timeline_head": head,
            "timeline_tail": tail,
            "timeline_omitted": omitted,
            "total_events": total,
        }
        _save_json(self.root / "trace.json", summary)
        _save_text(self.root / "timeline.md", _build_timeline_markdown(summary))

        return summary

    def _emit(self, event: dict) -> None:
        """Append an event to the in-memory timeline."""
        ts = datetime.now(timezone.utc).isoformat()
        entry = {"timestamp": ts, **event}
        self._timeline.append(entry)


def _new_trace_id() -> str:
    return f"trace-{uuid.uuid4().hex[:12]}"


def _duration_ms(started: str, ended: str) -> int:
    """Compute duration in milliseconds between two ISO timestamps."""
    try:
        s = datetime.fromisoformat(started)
        e = datetime.fromisoformat(ended)
        return int((e - s).total_seconds() * 1000)
    except Exception:
        return 0


def _build_timeline_markdown(summary: dict) -> str:
    """Build a human-readable timeline.md from the trace summary."""
    lines: list[str] = [
        f"# Trace {summary.get('trace_id', '?')}",
        "",
        f"**Task:** {summary.get('task', '?')}",
        f"**Status:** {summary.get('status', '?')}",
        f"**Duration:** {summary.get('duration_ms', 0):,} ms",
        "",
        "## Stats",
        "",
        f"| Metric | Count |",
        f"|---|---|",
        f"| Node visits | {len(summary.get('node_visits', {}))} |",
        f"| Tool calls | {summary.get('tool_calls', 0)} |",
        f"| Failed calls | {summary.get('failed_tool_calls', 0)} |",
        f"| Approvals | {summary.get('approval_count', 0)} |",
        f"| Checkpoints | {summary.get('checkpoint_count', 0)} |",
        f"| Handoffs | {summary.get('handoff_count', 0)} |",
        f"| Total events | {summary.get('total_events', 0)} |",
        "",
    ]

    head = summary.get("timeline_head") or []
    if head:
        lines.append("## Timeline (first 20)")
        lines.append("")
        for e in head:
            ts = e.get("timestamp", "")[:19]
            etype = e.get("type", "?")
            node = e.get("node", "")
            lines.append(f"- `{ts}` **{etype}** {node}".rstrip())

    omitted = summary.get("timeline_omitted", 0)
    if omitted:
        lines.append(f"\n... {omitted} events omitted ...\n")

    tail = summary.get("timeline_tail") or []
    if tail:
        lines.append("## Timeline (last 80)")
        lines.append("")
        for e in tail:
            ts = e.get("timestamp", "")[:19]
            etype = e.get("type", "?")
            node = e.get("node", "")
            lines.append(f"- `{ts}` **{etype}** {node}".rstrip())

    return "\n".join(lines)


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")