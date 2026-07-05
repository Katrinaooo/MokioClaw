"""create_model — build a LangChain ChatOpenAI instance."""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def create_model(model_name: str = "gpt-4o") -> ChatOpenAI:
    """Return a ``ChatOpenAI`` instance configured from environment variables.

    Reads ``OPENAI_API_KEY``, ``OPENAI_API_BASE`` (optional), and
    ``OPENAI_ORGANIZATION`` (optional) from the environment / ``.env`` file.

    Args:
        model_name: The OpenAI model id to use (default ``"gpt-4o"``).
    """
    kwargs: dict = {
        "model": model_name,
        "temperature": 0,
    }

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Set it in your environment or in a .env file."
        )
    kwargs["api_key"] = api_key

    base_url = os.getenv("OPENAI_API_BASE")
    if base_url:
        kwargs["base_url"] = base_url

    org = os.getenv("OPENAI_ORGANIZATION")
    if org:
        kwargs["organization"] = org

    return ChatOpenAI(**kwargs)
