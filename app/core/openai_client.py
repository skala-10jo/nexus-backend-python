"""
OpenAI client singleton for shared use across all AI agents.
"""
from openai import AsyncOpenAI
from app.config import settings

# Global client instance
_client = None


def get_openai_client() -> AsyncOpenAI:
    """
    Get or create OpenAI client singleton.

    This ensures only one OpenAI client instance is created and reused
    across all AI agents, optimizing resource usage and HTTP connection pooling.

    Returns:
        AsyncOpenAI: Shared OpenAI client instance

    Example:
        >>> client = get_openai_client()
        >>> response = await client.chat.completions.create(...)
    """
    global _client

    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    return _client
