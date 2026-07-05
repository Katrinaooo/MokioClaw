"""Workspace path helpers."""

from pathlib import Path


def ensure_workspace(path: str) -> Path:
    """Resolve *path* to an absolute directory, creating it if needed.

    Returns the resolved ``Path``.
    """
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p
