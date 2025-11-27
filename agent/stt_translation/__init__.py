"""
STT Translation Agent Module

Azure Speech Service를 사용한 음성→텍스트 변환 Agent
"""
from .azure_speech_agent import AzureSpeechAgent, get_azure_speech_agent
from .stt_agent import STTAgent, get_stt_agent

__all__ = [
    'AzureSpeechAgent',
    'get_azure_speech_agent',
    'STTAgent',
    'get_stt_agent',
]
