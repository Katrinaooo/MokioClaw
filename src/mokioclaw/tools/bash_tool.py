"""BashTool — execute shell commands within the workspace."""

import json
import subprocess
from typing import Callable

from langchain_core.tools import StructuredTool

from mokioclaw.core.approval import (
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    classify_command_risk,
    new_approval_id,
    normalize_approval_mode,
)
from mokioclaw.core.state import RuntimeState


def _run_bash(
    state: RuntimeState,
    command: str,
    timeout_seconds: int = 120,
    *,
    approval_mode: str = "inline",
    approval_handler: ApprovalHandler | None = None,
) -> str:
    """Run a shell command in the workspace directory.

    Args:
        command:         The shell command to execute.
        timeout_seconds: Maximum runtime in seconds (default 120, max 600).
        approval_mode:   ``"inline"``, ``"auto"``, or ``"deny"``.
        approval_handler: Callback for ``"inline"`` mode.
    """
    timeout_seconds = min(max(timeout_seconds, 1), 600)
    mode = normalize_approval_mode(approval_mode)

    # --- Risk check ---
    risk = classify_command_risk(command)
    if risk is not None:
        if mode == "deny":
            return json.dumps(
                {
                    "ok": False,
                    "command": command,
                    "risk_reason": risk,
                    "approval": "denied",
                    "reason": "approval mode is 'deny'",
                },
                ensure_ascii=False,
            )

        if mode == "inline":
            if approval_handler is None:
                return json.dumps(
                    {
                        "ok": False,
                        "command": command,
                        "risk_reason": risk,
                        "approval": "denied",
                        "reason": "no approval handler configured for inline mode",
                    },
                    ensure_ascii=False,
                )

            req = ApprovalRequest(
                id=new_approval_id(),
                command=command,
                risk_reason=risk,
            )
            decision = approval_handler(req)
            if not decision.approved:
                return json.dumps(
                    {
                        "ok": False,
                        "command": command,
                        "risk_reason": risk,
                        "approval": "denied",
                        "reason": decision.reason or "rejected by user",
                    },
                    ensure_ascii=False,
                )

            requires_approval = True
        else:  # mode == "auto"
            requires_approval = True
    else:
        requires_approval = False

    # --- Execute ---
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(state.workspace),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return json.dumps(
            {"ok": False, "error": f"Command timed out after {timeout_seconds}s"},
            ensure_ascii=False,
        )

    result = {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": proc.stdout.rstrip(),
        "stderr": proc.stderr.rstrip(),
    }
    if requires_approval:
        result["requires_approval"] = True

    return json.dumps(result, ensure_ascii=False)


def build_bash_tool(
    state: RuntimeState,
    *,
    approval_mode: str = "inline",
    approval_handler: ApprovalHandler | None = None,
) -> StructuredTool:
    """Return a ``StructuredTool`` that executes shell commands.

    Args:
        state: The workspace-scoped runtime state.
        approval_mode: ``"inline"``, ``"auto"``, or ``"deny"``.
        approval_handler: Required when *approval_mode* is ``"inline"``.
    """

    def bash(command: str, timeout_seconds: int = 120) -> str:
        return _run_bash(
            state,
            command,
            timeout_seconds,
            approval_mode=approval_mode,
            approval_handler=approval_handler,
        )

    return StructuredTool.from_function(
        func=bash,
        name="BashTool",
        description="Execute a shell command in the workspace directory. "
        "timeout_seconds: max runtime (1-600, default 120).",
    )