"""
Speaking Tutor Agent Package

Provides AI agents for:
- Audio transcription with speaker diarization (Azure)
- Speech feedback generation (GPT-4o)
- Learning content generation
"""
from agent.speaking_tutor.diarization_agent import DiarizationAgent
from agent.speaking_tutor.feedback_agent import SpeakingFeedbackAgent
from agent.speaking_tutor.learning_agent import LearningAgent

__all__ = [
    "DiarizationAgent",
    "SpeakingFeedbackAgent",
    "LearningAgent"
]
