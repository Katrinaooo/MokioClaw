"""Textual TUI for MokioClaw — interactive multi-turn agent UI."""

import threading
from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    Static,
)

from mokioclaw.cli.tui.approval import (
    ApprovalGate,
    ApprovalModal,
    _ApprovalRequestedMessage,
    make_approval_handler,
)
from mokioclaw.core.agent import stream_session_events
from mokioclaw.core.paths import ensure_workspace


# ---------------------------------------------------------------------------
# Agent event message — bridges the agent thread and the TUI thread
# ---------------------------------------------------------------------------


class AgentEventMessage(Message):
    """A message carrying an agent event from the background thread to the UI."""

    def __init__(self, event: dict) -> None:
        super().__init__()
        self.event = event


# ---------------------------------------------------------------------------
# UI widgets
# ---------------------------------------------------------------------------


class PlanPanel(Static):
    """Shows the current plan, todos, and acceptance criteria."""

    plan_summary: reactive[str] = reactive("")
    todos: reactive[list[dict]] = reactive([])
    criteria: reactive[list[str]] = reactive([])

    def watch_plan_summary(self, value: str) -> None:
        self._refresh()

    def watch_todos(self, value: list[dict]) -> None:
        self._refresh()

    def watch_criteria(self, value: list[str]) -> None:
        self._refresh()

    def _refresh(self) -> None:
        lines: list[str] = []
        if self.plan_summary:
            lines.append(f"[bold cyan]Plan:[/bold cyan] {self.plan_summary}")

        if self.todos:
            lines.append("")
            for t in self.todos:
                icon = {"completed": "✅", "blocked": "❌", "in_progress": "🔄", "pending": "⬜"}.get(
                    t.get("status", "pending"), "⬜"
                )
                lines.append(f"  {icon} {t.get('content', '')}")

        if self.criteria:
            lines.append("")
            lines.append("[bold]Criteria:[/bold]")
            for c in self.criteria:
                lines.append(f"  • {c}")

        self.update("\n".join(lines) if lines else "[dim]No plan yet[/dim]")


class EventStream(RichLog):
    """Scrollable event log showing tool calls, results, and handoffs."""

    def add_event(self, event: dict) -> None:
        etype = event.get("type", "")
        if etype == "tool_call":
            name = event.get("name", "?")
            args = event.get("args", {})
            args_str = ", ".join(f"{k}={v}" for k, v in list(args.items())[:3])
            self.write(f"[bold yellow]🔧 {name}[/bold yellow] [dim]{args_str}[/dim]")
        elif etype == "tool_result":
            name = event.get("name", "?")
            result = str(event.get("result", ""))[:200]
            self.write(f"[dim]📋 {name}: {result}[/dim]")
        elif etype == "handoff":
            fr = event.get("from", "?")
            to = event.get("to", "?")
            inst = str(event.get("instruction", ""))[:100]
            self.write(f"[bold magenta]🔄 {fr} → {to}[/bold magenta] [dim]{inst}[/dim]")
        elif etype == "checkpoint_saved":
            self.write(f"[dim]💾 Checkpoint saved at {event.get('timestamp', '')[:19]}[/dim]")
        elif etype == "final_answer":
            content = str(event.get("content", ""))[:500]
            self.write(f"[bold green]✅ {content}[/bold green]")
        elif etype == "verification":
            passed = event.get("passed", False)
            icon = "✅" if passed else "❌"
            self.write(f"[bold]{icon} Verification: {'PASSED' if passed else 'FAILED'}[/bold]")
        elif etype == "node_start":
            node = event.get("node", "?")
            self.write(f"[bold blue]▶ {node}[/bold blue]")
        elif etype == "session_start":
            self.write(f"[bold]📂 Session {event.get('session_id', '')[:12]} turn {event.get('turn', 0)}[/bold]")


# ---------------------------------------------------------------------------
# Main TUI app
# ---------------------------------------------------------------------------


class MokioClawTuiApp(App[None]):
    """Interactive TUI for MokioClaw with plan panel, event stream, and input.

    Layout::

        ┌─────────────────────────────────────┐
        │ 🐾 MokioClaw           session: xxx │
        ├─────────────────────────────────────┤
        │  [Plan Panel]                       │
        ├─────────────────────────────────────┤
        │  [Event Stream — scrollable]        │
        ├─────────────────────────────────────┤
        │  💬 [Input field]                   │
        └─────────────────────────────────────┘
    """

    TITLE = "🐾 MokioClaw"
    SUB_TITLE = "Multi-Agent TUI"
    CSS = """
    #plan-panel {
        height: auto;
        min-height: 3;
        max-height: 12;
        border: solid $primary;
        padding: 0 1;
        margin: 0 1;
    }
    #event-stream {
        height: 1fr;
        border: solid $surface;
        margin: 0 1;
    }
    #input-area {
        height: 3;
        margin: 0 1;
        padding: 0 1;
    }
    #approval-dialog {
        width: 60;
        height: auto;
        border: thick $error;
        background: $surface;
        padding: 1 2;
    }
    #approval-title {
        color: $error;
        text-style: bold;
        content-align: center;
    }
    #approval-hint {
        color: $text-disabled;
        content-align: center;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(
        self,
        workspace: Path,
        model_name: str = "gpt-5.5",
        max_attempts: int = 3,
        approval_mode: str = "inline",
        checkpoint_mode: str = "light",
        trace_mode: str = "on",
    ) -> None:
        super().__init__()
        self._workspace = workspace
        self._model_name = model_name
        self._max_attempts = max_attempts
        self._approval_mode = approval_mode
        self._checkpoint_mode = checkpoint_mode
        self._trace_mode = trace_mode
        self._agent_thread: threading.Thread | None = None
        self._approval_handler = make_approval_handler(self._post_approval)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield PlanPanel(id="plan-panel")
        yield EventStream(id="event-stream", markup=True, wrap=True, highlight=True)
        with Container(id="input-area"):
            yield Input(placeholder="💬 Enter your task...", id="task-input")
        yield Footer()

    def on_mount(self) -> None:
        """Focus the input field on startup."""
        self.query_one("#task-input", Input).focus()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Start a new agent run when the user submits input."""
        task = event.value.strip()
        if not task:
            return

        event.input.clear()
        self._start_agent(task)

    # ------------------------------------------------------------------
    # Agent thread
    # ------------------------------------------------------------------

    @work(thread=True)
    def _start_agent(self, task: str) -> None:
        """Run the agent in a background thread, posting events to the UI."""
        query = self.query_one("#task-input", Input)
        query.disabled = True

        try:
            for event in stream_session_events(
                task,
                workspace=self._workspace,
                model_name=self._model_name,
                max_attempts=self._max_attempts,
                approval_mode=self._approval_mode,
                approval_handler=self._approval_handler,
                checkpoint_mode=self._checkpoint_mode,
                trace_mode=self._trace_mode,
            ):
                self.post_message(AgentEventMessage(event))
        except Exception as exc:
            self.post_message(AgentEventMessage({
                "type": "final_answer",
                "content": f"Error: {exc}",
            }))
        finally:
            query.disabled = False
            query.focus()

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def on_agent_event_message(self, message: AgentEventMessage) -> None:
        """Handle an agent event in the UI thread."""
        event = message.event
        etype = event.get("type", "")

        # Update plan panel
        if etype == "plan":
            plan_panel = self.query_one(PlanPanel)
            plan_panel.plan_summary = event.get("summary", "")
            plan_panel.todos = event.get("todos") or []
            plan_panel.criteria = event.get("acceptance_criteria") or []

        # Add to event stream
        self.query_one(EventStream).add_event(event)

        # Handle approval requests
        if etype == "approval_requested":
            gate = event.get("gate")
            if gate:
                self._show_approval(gate)

    # ------------------------------------------------------------------
    # Approval
    # ------------------------------------------------------------------

    def _post_approval(self, gate: ApprovalGate) -> None:
        """Called from the tool thread — post an approval request to the UI."""
        self.post_message(AgentEventMessage({
            "type": "approval_requested",
            "gate": gate,
        }))

    @work(thread=True)
    async def _show_approval(self, gate: ApprovalGate) -> None:
        """Show the approval modal and resolve the gate."""
        approved = await self.push_screen_wait(ApprovalModal(gate.request))
        gate.resolve(approved, "user decision" if approved else "user denied")

    def action_quit(self) -> None:
        self.exit()