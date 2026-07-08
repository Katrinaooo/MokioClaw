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
            if name == "WebSearchTool":
                q = args.get("query", "")
                self.write(f"[bold cyan]🔍[/bold cyan] [italic]\"{q}\"[/italic] → searching...")
            elif name in ("CallSearchAgentTool", "CallCodeAgentTool"):
                inst = str(args.get("instruction", ""))[:80]
                icon = "🔍" if "Search" in name else "🔧"
                self.write(f"[bold yellow]{icon} Delegating to {name.replace('Call','').replace('AgentTool','')}[/bold yellow]")
            else:
                args_str = ", ".join(f"{k}={v}" for k, v in list(args.items())[:2])
                self.write(f"[bold yellow]🔧 {name}[/bold yellow] [dim]{args_str}[/dim]")

        elif etype == "tool_result":
            name = event.get("name", "?")
            result = str(event.get("result", ""))
            if name == "WebSearchTool":
                try:
                    import json
                    r = json.loads(result)
                    count = len(r.get("results", []))
                    self.write(f"[dim]📋 Found {count} results[/dim]")
                except Exception:
                    self.write(f"[dim]📋 Search completed[/dim]")
            elif name == "FileWriteTool":
                self.write(f"[dim]📝 File written ({len(result)} bytes)[/dim]")
            else:
                self.write(f"[dim]📋 {name}: {result[:120]}[/dim]")

        elif etype == "search_results":
            try:
                import json
                r = json.loads(str(event.get("result", "{}")))
                results = r.get("results", [])
                for res in results[:3]:
                    title = res.get("title", "")[:60]
                    self.write(f"  [dim]• {title}[/dim]")
            except Exception:
                pass

        elif etype == "handoff":
            fr = event.get("from", "?")
            to = event.get("to", "?")
            inst = str(event.get("instruction", ""))[:100]
            if to == "searchAgent":
                self.write(f"[bold cyan]🔄 {fr} → {to}[/bold cyan] [dim]searching...[/dim]")
            elif to == "codeAgent":
                self.write(f"[bold yellow]🔄 {fr} → {to}[/bold yellow] [dim]working...[/dim]")
            else:
                self.write(f"[bold magenta]🔄 {fr} → {to}[/bold magenta] [dim]{inst}[/dim]")

        elif etype == "checkpoint_saved":
            self.write(f"[dim]💾 Checkpoint saved[/dim]")

        elif etype == "final_answer":
            content = str(event.get("content", ""))[:600]
            self.write(f"\n[bold green]✅ {content}[/bold green]\n")

        elif etype == "verification":
            passed = event.get("passed", False)
            if passed:
                checks = event.get("checks", [])
                detail = ", ".join(c.get("name", "") for c in checks[:3])
                self.write(f"[bold green]✅ Verified:[/bold green] [dim]{detail}[/dim]")
            else:
                self.write(f"[bold red]❌ Verification FAILED[/bold red]")

        elif etype == "node_start":
            node = event.get("node", "?")
            if node == "planner":
                self.write(f"\n[bold cyan]📋 Plan:[/bold cyan]")
            elif node == "verifier":
                self.write(f"\n[bold]🔍 Verifying...[/bold]")
            elif node == "final":
                pass  # handled by final_answer

        elif etype == "session_start":
            sid = event.get("session_id", "")[:12]
            turn = event.get("turn", 0)
            route = event.get("route", "workflow")
            self.write(f"[dim]Session: {sid} | Turn: {turn} | Route: {route}[/dim]")

        elif etype == "plan":
            summary = event.get("summary", "")
            self.write(f"[bold cyan]📋 Plan:[/bold cyan] [italic]{summary}[/italic]")
            for t in (event.get("todos") or []):
                icon = {"completed": "✅", "blocked": "❌", "in_progress": "🔄", "pending": "⬜"}.get(
                    t.get("status", "pending"), "⬜"
                )
                self.write(f"  {icon} {t.get('content', '')}")

        elif etype == "chat_response":
            self.write(f"[bold]🗨️ Chat:[/bold] {event.get('content', '')[:500]}")


class StatusBar(Static):
    """Shows session info, mode, and workspace path."""

    def update_status(
        self,
        session_id: str = "",
        workspace: str = "",
        approval_mode: str = "inline",
        checkpoint_mode: str = "light",
        trace_mode: str = "on",
    ) -> None:
        sid = session_id[:12] if session_id else "—"
        ws = Path(workspace).name if workspace else "—"
        self.update(
            f"[dim]Session:[/dim] {sid}  "
            f"[dim]Workspace:[/dim] {ws}  "
            f"[dim]Approval:[/dim] {approval_mode}  "
            f"[dim]Checkpoint:[/dim] {checkpoint_mode}  "
            f"[dim]Trace:[/dim] {trace_mode}"
        )


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
    SUB_TITLE = "Stage 6 · MultiAgent + Context + Harness"
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
    #status-bar {
        height: 1;
        padding: 0 1;
        margin: 0 1;
        background: $surface;
    }
    #input-area {
        height: 3;
        margin: 0 1;
        padding: 0 1;
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
        yield StatusBar(id="status-bar")
        yield PlanPanel(id="plan-panel")
        yield EventStream(id="event-stream", markup=True, wrap=True, highlight=True)
        with Container(id="input-area"):
            yield Input(placeholder="💬 输入任务或聊天...", id="task-input")
        yield Footer()

    def on_mount(self) -> None:
        """Show welcome message and focus the input field."""
        stream = self.query_one(EventStream)
        stream.write(f"[bold cyan]🐾 MokioClaw[/bold cyan] [dim]v0.6.0 — Stage 6: TUI + Session[/dim]")
        stream.write("")

        self.query_one(StatusBar).update_status(
            workspace=str(self._workspace),
            approval_mode=self._approval_mode,
            checkpoint_mode=self._checkpoint_mode,
            trace_mode=self._trace_mode,
        )

        # Check for existing checkpoint
        import json
        cp_path = self._workspace / ".mokioclaw" / "checkpoints" / "checkpoint.json"
        if cp_path.exists():
            try:
                cp = json.loads(cp_path.read_text(encoding="utf-8"))
                task = cp.get("task", "?")
                status = cp.get("status", "?")
                stream.write(f"[yellow]⚡ Found checkpoint:[/yellow] [dim]{task[:80]}[/dim] ([italic]{status}[/italic])")
                stream.write(f"[dim]Type [bold]/resume[/bold] to continue, or enter a new task to start fresh.[/dim]")
                stream.write("")
            except Exception:
                pass

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

        if task == "/resume":
            self._resume_agent()
            return

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

    def _resume_agent(self) -> None:
        """Resume from the last checkpoint."""
        cp_path = self._workspace / ".mokioclaw" / "checkpoints" / "checkpoint.json"
        if not cp_path.exists():
            self.query_one(EventStream).write("[red]No checkpoint found.[/red]")
            return

        import json
        try:
            cp = json.loads(cp_path.read_text(encoding="utf-8"))
        except Exception:
            self.query_one(EventStream).write("[red]Checkpoint unreadable.[/red]")
            return

        task = cp.get("task", "resume")
        self.query_one(EventStream).write(f"[yellow]⚡ Resuming:[/yellow] [dim]{task[:80]}[/dim]")
        self._start_agent(task)

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

        # Update status bar on session start
        if etype == "session_start":
            self.query_one(StatusBar).update_status(
                session_id=event.get("session_id", ""),
                workspace=str(self._workspace),
                approval_mode=self._approval_mode,
                checkpoint_mode=self._checkpoint_mode,
                trace_mode=self._trace_mode,
            )

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


# ---------------------------------------------------------------------------
# Entry point for the CLI
# ---------------------------------------------------------------------------


def run_tui(
    workspace: str | None = None,
    model_name: str = "gpt-5.5",
    max_attempts: int = 3,
    approval_mode: str = "inline",
    checkpoint_mode: str = "light",
    trace_mode: str = "on",
) -> None:
    """Launch the MokioClaw TUI."""
    ws = ensure_workspace(workspace) if workspace else ensure_workspace("./workspace")
    app = MokioClawTuiApp(
        workspace=ws,
        model_name=model_name,
        max_attempts=max_attempts,
        approval_mode=approval_mode,
        checkpoint_mode=checkpoint_mode,
        trace_mode=trace_mode,
    )
    app.run()