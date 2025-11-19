"""
Azure Speech STT + Translation Agent 패키지

브라우저에서 Azure Speech SDK를 직접 사용하므로,
이 Agent는 토큰 관리만 담당합니다.

실제 STT 및 Translation은 프론트엔드에서 처리됩니다.
"""
from agent.stt_translation.azure_speech_agent import AzureSpeechAgent, get_azure_speech_agent

__all__ = ["AzureSpeechAgent", "get_azure_speech_agent"]
