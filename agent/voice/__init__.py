"""
Voice processing agents.

Agents:
- PronunciationStreamingAgent: Real-time pronunciation assessment (Azure)
- BaseAzureAgent: Base class for Azure Speech SDK agents
"""

from agent.voice.base_azure_agent import BaseAzureAgent
from agent.voice.pronunciation_streaming_agent import PronunciationStreamingAgent

__all__ = [
    "BaseAzureAgent",
    "PronunciationStreamingAgent",
]
