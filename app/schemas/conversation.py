"""
Conversation API Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel
from typing import List, Dict, Optional


class StartConversationRequest(BaseModel):
    """대화 시작 요청"""
    scenarioId: str


class SendMessageRequest(BaseModel):
    """메시지 전송 요청"""
    scenarioId: str
    message: str
    history: List[Dict[str, str]] = []


class EndConversationRequest(BaseModel):
    """대화 종료 요청"""
    scenarioId: str
    history: List[Dict[str, str]] = []


class MessageFeedbackRequest(BaseModel):
    """메시지 피드백 요청"""
    scenarioId: str
    message: str
    detectedTerms: List[str] = []
    audioData: Optional[str] = None  # Base64 encoded audio for pronunciation assessment


class TranslateMessageRequest(BaseModel):
    """메시지 번역 요청"""
    message: str
    targetLanguage: str = "ko"  # Default to Korean
