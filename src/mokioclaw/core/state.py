"""RuntimeState — the shared state object passed through the agent loop."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class RuntimeState:
    """Holds the current workspace path and any runtime context.

    All tool operations are scoped to `workspace` — paths outside it are rejected.
    """

    workspace: Path = field(default_factory=Path.cwd)
    """Root directory for all tool I/O."""

    approval_mode: str = "inline"
    """Approval mode: ``"inline"``, ``"auto"``, or ``"deny"``."""

    approval_handler: Callable[..., Any] | None = None
    """Callback for ``"inline"`` approval mode."""

    checkpoint_mode: str = "light"
    """Checkpoint mode: ``"light"``, ``"strict"``, or ``"off"``."""

    trace_mode: str = "on"
    """Trace mode: ``"on"`` or ``"off"``."""

    trace_id: str = ""
    """Trace session id.  Auto-generated if empty."""

    resume_from: str = ""
    """Path to a previous workspace to resume from."""

    def resolve(self, path: str | Path) -> Path:
        """Resolve a user-supplied path against the workspace.

        Returns the absolute, resolved path.  Does NOT check containment —
        call :meth:`validate_path` for that.
        """
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace / p
        return p.resolve()

    def validate_path(self, path: str | Path) -> Path:
        """Resolve *path* and raise ``ValueError`` if it escapes the workspace.

        Returns the resolved ``Path`` on success.
        """
        resolved = self.resolve(path)
        try:
            resolved.relative_to(self.workspace)
        except ValueError:
            raise ValueError(
                f"Path {str(resolved)!r} is outside the workspace "
                f"{str(self.workspace)!r}"
            )
        return resolved


def create_runtime(
    workspace: str | Path,
    *,
    approval_mode: str = "inline",
    approval_handler: Callable[..., Any] | None = None,
    checkpoint_mode: str = "light",
    trace_mode: str = "on",
    resume_from: str = "",
) -> RuntimeState:
    """Create a ``RuntimeState`` with sensible defaults."""
    return RuntimeState(
        workspace=Path(workspace),
        approval_mode=approval_mode,
        approval_handler=approval_handler,
        checkpoint_mode=checkpoint_mode,
        trace_mode=trace_mode,
        resume_from=resume_from,
    )
