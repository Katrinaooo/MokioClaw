"""Command risk classification and approval system."""

import re
import uuid
from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------------
# Risk patterns — regex → human-readable reason
# ---------------------------------------------------------------------------

RISK_PATTERNS: list[tuple[str, str]] = [
    (r"(?:^|&&|\|\||;)\s*(?:python\s+-m\s+)?pip\s+install\b", "Python package installation"),
    (r"(?:^|&&|\|\||;)\s*uv\s+add\b", "Project dependency change with uv add"),
    (r"(?:^|&&|\|\||;)\s*uv\s+sync\b", "Dependency synchronization with uv sync"),
    (r"(?:^|&&|\|\||;)\s*uv\s+pip\s+install\b", "Python package installation with uv pip"),
    (r"(?:^|&&|\|\||;)\s*npm\s+install\b", "Node package installation"),
    (r"(?:^|&&|\|\||;)\s*pnpm\s+install\b", "Node package installation"),
    (r"(?:^|&&|\|\||;)\s*yarn\s+(?:install\b|add\b)", "Node package installation"),
    (r"(?:^|&&|\|\||;)\s*(?:curl|wget)\b", "Network download command"),
    (r"(?:^|&&|\|\||;)\s*uvicorn\b", "Long-running development server"),
    (r"(?:^|&&|\|\||;)\s*python\s+-m\s+http\.server\b", "Long-running development server"),
]


def classify_command_risk(command: str) -> str | None:
    """Check *command* against the risk patterns.

    Returns:
        The risk reason string if a pattern matches, or ``None`` if the
        command is considered safe.
    """
    for pattern, reason in RISK_PATTERNS:
        if re.search(pattern, command):
            return reason
    return None


# ---------------------------------------------------------------------------
# Approval data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ApprovalRequest:
    """An approval request for a risky command."""

    id: str
    command: str
    risk_reason: str
    tool_name: str = "BashTool"


@dataclass(frozen=True)
class ApprovalDecision:
    """The human decision on an approval request."""

    approved: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# Approval modes
# ---------------------------------------------------------------------------

VALID_APPROVAL_MODES = {"inline", "auto", "deny"}


def normalize_approval_mode(mode: str | None) -> str:
    """Normalise the approval mode string.

    Defaults to ``"inline"``; invalid values also fall back to ``"inline"``.
    """
    if mode and mode in VALID_APPROVAL_MODES:
        return mode
    return "inline"


# ---------------------------------------------------------------------------
# Approval handler type
# ---------------------------------------------------------------------------

ApprovalHandler = Callable[[ApprovalRequest], ApprovalDecision]


def new_approval_id() -> str:
    """Generate a short approval id."""
    return f"approval-{uuid.uuid4().hex[:8]}"