"""WebSearchTool — Tavily-powered web search for the search agent."""

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool

load_dotenv()


def _search_web(query: str, max_results: int = 5) -> str:
    """Call the Tavily Search API and return structured results as JSON.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 5, max 10).
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return json.dumps(
            {"ok": False, "error": "missing TAVILY_API_KEY", "query": query},
            ensure_ascii=False,
        )

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=min(max_results, 10),
            search_depth="basic",
        )

        results: list[dict[str, Any]] = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": (r.get("content", "") or "")[:500],
                "score": r.get("score", 0.0),
            })

        return json.dumps(
            {
                "ok": True,
                "query": response.get("query", query),
                "answer": response.get("answer", ""),
                "results": results,
            },
            ensure_ascii=False,
        )

    except Exception as exc:
        return json.dumps(
            {"ok": False, "error": str(exc), "query": query},
            ensure_ascii=False,
        )


def build_web_search_tool() -> StructuredTool:
    """Return a ``StructuredTool`` wrapping the Tavily search API."""

    return StructuredTool.from_function(
        func=_search_web,
        name="WebSearchTool",
        description=(
            "Search the web for information. "
            "query: the search query string. "
            "max_results: number of results (default 5, max 10). "
            "Returns JSON with ok, query, answer, and results list."
        ),
    )