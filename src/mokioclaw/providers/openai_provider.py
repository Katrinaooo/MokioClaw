"""create_model — build a model wrapper around the OpenAI Chat Completions API."""

import json
import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from openai import OpenAI

load_dotenv()


# ---------------------------------------------------------------------------
# Model wrapper — Chat Completions API with LangChain-compatible interface
# ---------------------------------------------------------------------------

@dataclass
class CodexModel:
    """Wraps the OpenAI Chat Completions API, mirroring ChatOpenAI's interface.

    Supports ``.bind_tools(tools)`` → ``.invoke(messages)`` so that nodes
    and agent code don't need to change.
    """

    model_name: str
    client: OpenAI

    _tools: list[StructuredTool] = field(default_factory=list)
    _tool_schemas: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bind_tools(self, tools: list[StructuredTool]) -> "CodexModel":
        """Store tool definitions and return self for chaining."""
        self._tools = tools
        self._tool_schemas = [self._convert_tool(t) for t in tools]
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Send *messages* to the Chat Completions API and return an ``AIMessage``.

        The returned ``AIMessage`` has:
        - ``.content`` — plain-text assistant reply
        - ``.tool_calls`` — list of ``{name, args, id}`` dicts (or ``None``)
        - ``.id`` — response id
        """
        chat_messages = self._messages_to_chat(messages)

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": chat_messages,
        }
        if self._tool_schemas:
            kwargs["tools"] = self._tool_schemas

        resp = self.client.chat.completions.create(**kwargs)
        return self._parse_response(resp)

    # ------------------------------------------------------------------
    # Tool schema conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tool(tool: StructuredTool) -> dict:
        """Convert a LangChain ``StructuredTool`` to a Chat Completions tool dict."""
        schema = tool.args_schema.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                    "additionalProperties": False,
                },
            },
        }

    # ------------------------------------------------------------------
    # Message conversion: LangChain → Chat Completions
    # ------------------------------------------------------------------

    @staticmethod
    def _messages_to_chat(messages: list[BaseMessage]) -> list[dict]:
        """Convert LangChain messages to Chat Completions format."""
        chat_msgs: list[dict] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                chat_msgs.append({"role": "system", "content": _to_str(msg.content)})

            elif isinstance(msg, HumanMessage):
                chat_msgs.append({"role": "user", "content": _to_str(msg.content)})

            elif isinstance(msg, AIMessage):
                entry: dict = {"role": "assistant", "content": _to_str(msg.content) or ""}
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"], ensure_ascii=False),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                chat_msgs.append(entry)

            elif isinstance(msg, ToolMessage):
                chat_msgs.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": _to_str(msg.content),
                })

        return chat_msgs

    # ------------------------------------------------------------------
    # Response parsing: Chat Completions → AIMessage
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(resp) -> AIMessage:
        """Parse a Chat Completions response into a LangChain ``AIMessage``."""
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                args_str = tc.function.arguments or "{}"
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({
                    "name": tc.function.name,
                    "args": args,
                    "id": tc.id,
                })

        return AIMessage(
            content=msg.content or "",
            **(dict(tool_calls=tool_calls) if tool_calls else {}),
            id=getattr(resp, "id", ""),
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_model(model_name: str = "gpt-5.5") -> CodexModel:
    """Return a ``CodexModel`` configured from environment variables.

    Reads ``OPENAI_API_KEY`` and ``OPENAI_API_BASE`` from the environment /
    ``.env`` file.  The returned object supports ``.bind_tools()`` and
    ``.invoke()``, a drop-in replacement for ``ChatOpenAI``.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Set it in your environment or in a .env file."
        )

    base_url = os.getenv("OPENAI_API_BASE")
    client_kwargs: dict = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    return CodexModel(model_name=model_name, client=client)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_str(content) -> str:
    """Normalise LangChain message content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", "") or item.get("output_text", ""))
        return "\n".join(parts)
    return str(content)