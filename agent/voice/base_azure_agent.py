"""
Base Azure Agent class for Azure-based AI agents.
Provides abstract interface for agents using Azure Cognitive Services (Speech, Vision, etc.).
"""
from abc import ABC, abstractmethod


class BaseAzureAgent(ABC):
    """
    Abstract base class for all Azure-based AI agents.

    Azure agents (TTS, STT, Pronunciation) use Azure Cognitive Services
    instead of OpenAI, so they don't inherit from BaseAgent.

    All Azure agents must:
    1. Inherit from BaseAzureAgent
    2. Implement the process() method
    3. Handle their own Azure service clients (e.g., Azure Speech SDK)

    Example:
        >>> class MyAzureAgent(BaseAzureAgent):
        ...     def __init__(self, api_key: str, region: str):
        ...         self.api_key = api_key
        ...         self.region = region
        ...
        ...     def process(self, data: bytes) -> dict:
        ...         # Azure service processing logic
        ...         return {"result": "..."}
        ...
        >>> agent = MyAzureAgent(api_key="...", region="koreacentral")
        >>> result = agent.process(data)
    """

    @abstractmethod
    def process(self, *args, **kwargs):
        """
        Core processing method that each Azure agent must implement.

        This is the main interface for Azure agent functionality.
        Each agent defines its own input/output signature.

        Args:
            *args: Agent-specific positional arguments
            **kwargs: Agent-specific keyword arguments

        Returns:
            Agent-specific result (varies by implementation)

        Raises:
            Exception: If Azure service processing fails

        Note:
            Subclasses MUST implement this method with proper type hints
            and docstrings describing their specific inputs and outputs.
        """
        pass
