"""File-system tools: read, write, edit — all scoped to the workspace."""

from pathlib import Path

from langchain_core.tools import StructuredTool

from mokioclaw.core.state import RuntimeState


# ---------------------------------------------------------------------------
# FileReadTool
# ---------------------------------------------------------------------------

def _read_file(state: RuntimeState, file_path: str, offset: int = 0, limit: int = 2000) -> str:
    """Read a file's contents, returning lines with line numbers.

    Args:
        file_path: Path relative to the workspace (or absolute, inside it).
        offset:  0-based line number to start reading from.
        limit:   Maximum number of lines to return.
    """
    resolved = state.validate_path(file_path)
    if not resolved.is_file():
        return f"Error: {str(resolved)!r} is not a file or does not exist."

    lines = resolved.read_text(encoding="utf-8").splitlines()
    total = len(lines)

    if offset < 0:
        offset = 0
    if offset >= total:
        return f"Error: offset {offset} is beyond file end (total {total} lines)."

    end = min(offset + limit, total)
    snippet = lines[offset:end]

    # Format with 1-based line numbers
    out: list[str] = []
    for i, line in enumerate(snippet, start=offset + 1):
        out.append(f"{i}\t{line}")

    header = f"File: {resolved}  (lines {offset + 1}-{end} of {total})"
    return header + "\n" + "\n".join(out)


def build_file_read_tool(state: RuntimeState) -> StructuredTool:
    """Return a ``StructuredTool`` that reads files within the workspace."""

    def read_file(file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return _read_file(state, file_path, offset, limit)

    return StructuredTool.from_function(
        func=read_file,
        name="FileReadTool",
        description="Read a file. Returns line-numbered content. "
        "file_path is relative to workspace. "
        "offset: 0-based start line. limit: max lines to return.",
    )


# ---------------------------------------------------------------------------
# FileWriteTool
# ---------------------------------------------------------------------------

def _write_file(state: RuntimeState, file_path: str, content: str) -> str:
    """Create or overwrite a file."""
    resolved = state.validate_path(file_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {str(resolved)}"


def build_file_write_tool(state: RuntimeState) -> StructuredTool:
    """Return a ``StructuredTool`` that writes files within the workspace."""

    def write_file(file_path: str, content: str) -> str:
        return _write_file(state, file_path, content)

    return StructuredTool.from_function(
        func=write_file,
        name="FileWriteTool",
        description="Create or overwrite a file. file_path is relative to workspace. "
        "content is the full file contents.",
    )


# ---------------------------------------------------------------------------
# FileEditTool
# ---------------------------------------------------------------------------

def _edit_file(state: RuntimeState, file_path: str, old_text: str, new_text: str) -> str:
    """Replace *old_text* with *new_text* in a file.

    The *old_text* must appear exactly once in the file.  If it appears zero
    times or more than once, an error is returned.
    """
    resolved = state.validate_path(file_path)
    if not resolved.is_file():
        return f"Error: {str(resolved)!r} is not a file or does not exist."

    original = resolved.read_text(encoding="utf-8")
    count = original.count(old_text)

    if count == 0:
        return f"Error: old_text not found in {str(resolved)!r}."
    if count > 1:
        return (
            f"Error: old_text appears {count} times in {str(resolved)!r}. "
            "It must be unique — provide more context to make it unambiguous."
        )

    replaced = original.replace(old_text, new_text, 1)
    resolved.write_text(replaced, encoding="utf-8")
    return f"Replaced 1 occurrence in {str(resolved)}"


def build_file_edit_tool(state: RuntimeState) -> StructuredTool:
    """Return a ``StructuredTool`` that edits files within the workspace."""

    def edit_file(file_path: str, old_text: str, new_text: str) -> str:
        return _edit_file(state, file_path, old_text, new_text)

    return StructuredTool.from_function(
        func=edit_file,
        name="FileEditTool",
        description="Replace a unique text fragment in a file. "
        "old_text must appear exactly once. "
        "If it appears 0 or >1 times an error is returned.",
    )
