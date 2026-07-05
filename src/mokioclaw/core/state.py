"""RuntimeState — the shared state object passed through the agent loop."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RuntimeState:
    """Holds the current workspace path and any runtime context.

    All tool operations are scoped to `workspace` — paths outside it are rejected.
    """

    workspace: Path = field(default_factory=Path.cwd)
    """Root directory for all tool I/O. Paths are resolved relative to this."""

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
