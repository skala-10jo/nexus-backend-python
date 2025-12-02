"""
STT Translation Agent Module

Azure Speech Service를 사용한 음성→텍스트 변환 Agent
"""
from .stt_agent import STTAgent, get_stt_agent
from .translation_agent import TranslationAgent, get_translation_agent

# 하위 호환성: AzureSpeechAgent는 app/core/azure_speech_token_manager.py로 이동
from app.core.azure_speech_token_manager import AzureSpeechTokenManager as AzureSpeechAgent

def get_azure_speech_agent():
    """하위 호환성 함수"""
    return AzureSpeechAgent.get_instance()

__all__ = [
    'AzureSpeechAgent',
    'get_azure_speech_agent',
    'STTAgent',
    'get_stt_agent',
    'TranslationAgent',
    'get_translation_agent'
]
