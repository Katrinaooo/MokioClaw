"""LangGraph shared state for MokioClaw's ReAct + verification workflow."""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from mokioclaw.core.state import RuntimeState


class TodoItem(TypedDict):
    """A single todo entry tracked during plan execution."""

    id: str
    content: str
    status: str  # "pending" | "in_progress" | "completed" | "blocked"
    note: str


class VerificationResult(TypedDict):
    """Result of running a verification command."""

    command: str
    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str


class SourceItem(TypedDict, total=False):
    """A source URL collected during research."""

    title: str
    url: str
    content: str
    score: float


class AgentHandoff(TypedDict, total=False):
    """Record of a handoff between planner and a specialist agent."""

    from_agent: str
    to_agent: str
    instruction: str
    result: str


class MokioGraphState(TypedDict, total=False):
    """Shared state for the MokioClaw LangGraph.

    ``total=False`` means every key is optional — nodes fill in what they produce.
    """

    task: str
    """The user's task description."""

    model_name: str
    """OpenAI model name to use for all LLM calls."""

    runtime: RuntimeState
    """Workspace-scoped runtime state (paths, tools, etc.)."""

    messages: Annotated[list[BaseMessage], add_messages]
    """Conversation history.  ``add_messages`` ensures LangGraph merges
    new messages (including ToolMessage → AIMessage id matching) instead of
    overwriting the list."""

    plan_summary: str
    """High-level plan produced by the plan node."""

    todos: list[TodoItem]
    """Structured todo list, updated as the agent progresses."""

    acceptance_criteria: list[str]
    """Criteria the plan node generates to judge success."""

    verification_commands: list[str]
    """Shell commands that the verify node runs to validate the result."""

    verification_results: list[VerificationResult]
    """Results collected from running each verification command."""

    passed: bool
    """Did verification pass?"""

    attempts: int
    """How many plan→execute→verify cycles have been attempted."""

    max_attempts: int
    """Maximum number of plan→execute→verify cycles allowed."""

    final_answer: str
    """The final summary returned to the user."""

    # --- Dynamic fields populated during execution ---

    last_error: str
    """Error summary from the last failed verification (used by planner to revise)."""

    last_actor_summary: str
    """Final text output from the most recent actor run."""

    verification_checks: list[dict]
    """Structured check results from the verifier LLM:
    ``[{name: str, passed: bool, detail: str}, ...]``."""

    # --- Stage 3 fields: sub-agent orchestration ---

    research_notes: str
    """Consolidated research notes from searchAgent runs."""

    sources: list[SourceItem]
    """Source URLs and metadata collected during research."""

    agent_handoffs: list[AgentHandoff]
    """Log of planner → specialist handoffs for audit and debugging."""

    code_agent_summary: str
    """Final summary from the most recent codeAgent run."""
