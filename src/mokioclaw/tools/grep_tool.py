"""GrepTool — regex search within the workspace."""

import re
from pathlib import Path

from langchain_core.tools import StructuredTool

from mokioclaw.core.state import RuntimeState


def _grep(
    state: RuntimeState,
    pattern: str,
    path: str = ".",
    glob: str = "",
    head_limit: int = 250,
    ignore_case: bool = False,
) -> str:
    """Search files under *path* for lines matching *pattern*.

    Args:
        pattern:     Python regex pattern.
        path:        Directory or file to search (relative to workspace).
        glob:        Optional fnmatch-style glob to filter files (e.g. ``"*.py"``).
        head_limit:  Max matching lines to return.
        ignore_case: If True, compile with ``re.IGNORECASE``.
    """
    resolved = state.validate_path(path)
    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return f"Error: invalid regex pattern: {exc}"

    lines_out: list[str] = []
    files = _walk(resolved, glob)

    for fp in files:
        if head_limit <= 0:
            break
        try:
            content = fp.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(content.splitlines(), start=1):
            if head_limit <= 0:
                break
            if regex.search(line):
                lines_out.append(f"{fp}:{lineno}: {line.rstrip()}")
                head_limit -= 1

    return "\n".join(lines_out) if lines_out else "No matches found."


def _walk(root: Path, glob_pattern: str) -> list[Path]:
    """Return a sorted list of files under *root*, optionally filtered by glob."""
    if root.is_file():
        return [root]
    if not root.is_dir():
        return []

    if glob_pattern:
        files = list(root.rglob(glob_pattern))
    else:
        files = [p for p in root.rglob("*") if p.is_file()]

    files.sort()
    return files


def build_grep_tool(state: RuntimeState) -> StructuredTool:
    """Return a ``StructuredTool`` for regex search within the workspace."""

    def grep(
        pattern: str,
        path: str = ".",
        glob: str = "",
        head_limit: int = 250,
        ignore_case: bool = False,
    ) -> str:
        return _grep(state, pattern, path, glob, head_limit, ignore_case)

    return StructuredTool.from_function(
        func=grep,
        name="GrepTool",
        description="Search files using a regex pattern. "
        "path: directory or file to search. glob: file-name filter (e.g. '*.py'). "
        "head_limit: max matching lines. ignore_case: case-insensitive search.",
    )
