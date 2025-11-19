"""
Video processing agents package.

Contains agents for:
- STT (Speech-to-Text) using OpenAI Whisper
- Subtitle generation (SRT format)
"""

from agent.video.stt_agent import VideoSTTAgent
from agent.video.subtitle_generator_agent import SubtitleGeneratorAgent

__all__ = [
    "VideoSTTAgent",
    "SubtitleGeneratorAgent",
]
