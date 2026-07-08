"""Approval modal and gate for the MokioClaw TUI."""

import threading
from typing import Callable

from textual.app import ComposeResult
from textual.containers import Center, Container
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from mokioclaw.core.approval import ApprovalDecision, ApprovalRequest


# ---------------------------------------------------------------------------
# ApprovalGate — thread-safe synchronisation
# ---------------------------------------------------------------------------


class ApprovalGate:
    """A thread-safe gate that blocks the tool thread until the user decides.

    Usage in tool thread::

        gate = ApprovalGate(request)
        # ... send request to TUI thread ...
        decision = gate.wait()   # blocks until user clicks Approve/Deny
        # decision is ApprovalDecision(approved=True/False)
    """

    def __init__(self, request: ApprovalRequest) -> None:
        self.request = request
        self._event = threading.Event()
        self._approved = False
        self._reason = ""

    def resolve(self, approved: bool, reason: str = "") -> None:
        """Signal the waiting thread with the decision."""
        self._approved = approved
        self._reason = reason
        self._event.set()

    def wait(self, timeout: float | None = None) -> ApprovalDecision:
        """Block until :meth:`resolve` is called.

        Args:
            timeout: Optional timeout in seconds.  If the timeout expires
                the command is denied.

        Returns:
            An ``ApprovalDecision``.
        """
        self._event.wait(timeout=timeout)
        return ApprovalDecision(
            approved=self._approved and self._event.is_set(),
            reason=self._reason or ("timeout" if not self._event.is_set() else ""),
        )


# ---------------------------------------------------------------------------
# ApprovalModal — the user-facing dialog
# ---------------------------------------------------------------------------


class ApprovalModal(ModalScreen[bool]):
    """A modal dialog that shows a risky command and asks for approval.

    Returns ``True`` if approved, ``False`` if denied.
    """

    BINDINGS = [
        ("y", "approve", "Approve"),
        ("enter", "approve", "Approve"),
        ("n", "deny", "Deny"),
        ("escape", "deny", "Deny"),
    ]

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        with Center():
            with Container(id="approval-dialog"):
                yield Label("⚠️  Command Requires Approval", id="approval-title")
                yield Static(
                    f"[bold]Tool:[/bold] {self.request.tool_name}\n"
                    f"[bold]Risk:[/bold] {self.request.risk_reason}\n\n"
                    f"[bold]Command:[/bold]\n"
                    f"[dim]{self.request.command}[/dim]",
                    id="approval-body",
                )
                yield Label(
                    "[Y] Approve  /  [N] Deny",
                    id="approval-hint",
                )

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


# ---------------------------------------------------------------------------
# Thread-safe approval handler factory
# ---------------------------------------------------------------------------


def make_approval_handler(
    post_message: Callable,
) -> Callable[[ApprovalRequest], ApprovalDecision]:
    """Create an approval handler that posts to the TUI for user decision.

    Returns a callable suitable for passing to ``stream_session_events``
    as the ``approval_handler`` parameter.
    """
    def handler(request: ApprovalRequest) -> ApprovalDecision:
        gate = ApprovalGate(request)
        post_message(_ApprovalRequestedMessage(gate))
        return gate.wait()

    return handler


# ---------------------------------------------------------------------------
# Internal message types
# ---------------------------------------------------------------------------


class _ApprovalRequestedMessage:
    """Internal message posted from tool thread to TUI thread."""

    def __init__(self, gate: ApprovalGate) -> None:
        self.gate = gate