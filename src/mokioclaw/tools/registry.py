"""build_tools — assemble the tool list bound to a RuntimeState."""

from langchain_core.tools import StructuredTool

from mokioclaw.core.state import RuntimeState
from mokioclaw.tools.file_tools import (
    build_file_read_tool,
    build_file_write_tool,
    build_file_edit_tool,
)
from mokioclaw.tools.grep_tool import build_grep_tool
from mokioclaw.tools.bash_tool import build_bash_tool


def build_tools(state: RuntimeState) -> list[StructuredTool]:
    """Return the full list of ``StructuredTool`` instances for the given state.

    These are ready to be passed to ``model.bind_tools(tools)``.
    """
    return [
        build_file_read_tool(state),
        build_file_write_tool(state),
        build_file_edit_tool(state),
        build_grep_tool(state),
        build_bash_tool(state),
    ]
