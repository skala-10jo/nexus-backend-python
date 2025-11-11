"""
Base Agent class for all AI agents.
Provides shared OpenAI client and defines the process() interface.
"""
from abc import ABC, abstractmethod
from openai import AsyncOpenAI
from app.core.openai_client import get_openai_client


class BaseAgent(ABC):
    """
    Abstract base class for all AI agents.

    All AI agents must:
    1. Inherit from BaseAgent
    2. Implement the process() method
    3. Use self.client for OpenAI API calls

    The OpenAI client is automatically shared across all agents via singleton pattern.

    Example:
        >>> class MyAgent(BaseAgent):
        ...     async def process(self, text: str) -> str:
        ...         response = await self.client.chat.completions.create(
        ...             model="gpt-4o",
        ...             messages=[{"role": "user", "content": text}]
        ...         )
        ...         return response.choices[0].message.content
        ...
        >>> agent = MyAgent()
        >>> result = await agent.process("Hello")
    """

    def __init__(self):
        """
        Initialize the agent with shared OpenAI client.

        The client is obtained via singleton pattern, ensuring only one
        instance is created across all agents.
        """
        self.client: AsyncOpenAI = get_openai_client()

    @abstractmethod
    async def process(self, *args, **kwargs):
        """
        Core processing method that each agent must implement.

        This is the main interface for agent functionality.
        Each agent defines its own input/output signature.

        Args:
            *args: Agent-specific positional arguments
            **kwargs: Agent-specific keyword arguments

        Returns:
            Agent-specific result (varies by implementation)

        Raises:
            Exception: If AI processing fails

        Note:
            Subclasses MUST implement this method with proper type hints
            and docstrings describing their specific inputs and outputs.
        """
        pass
