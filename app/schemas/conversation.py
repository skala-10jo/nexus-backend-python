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
    currentStepIndex: int = 0  # 현재 스텝 인덱스 (0-based)


class EndConversationRequest(BaseModel):
    """대화 종료 요청"""
    scenarioId: str
    history: List[Dict[str, str]] = []


class StepInfo(BaseModel):
    """대화 단계 정보"""
    name: str  # 단계 영문 식별자 (e.g., "ice_breaking")
    title: str  # 단계 한글 제목 (e.g., "인사")
    guide: Optional[str] = None  # 단계 가이드
    terminology: List[str] = []  # 이 단계에서 사용할 표현들


class MessageFeedbackRequest(BaseModel):
    """메시지 피드백 요청"""
    scenarioId: str
    message: str
    detectedTerms: List[str] = []
    audioData: Optional[str] = None  # Base64 encoded audio for pronunciation assessment
    currentStep: Optional[StepInfo] = None  # 현재 대화 단계


class TranslateMessageRequest(BaseModel):
    """메시지 번역 요청"""
    scenarioId: str  # 시나리오 ID (컨텍스트 기반 번역용)
    message: str
    targetLanguage: str = "ko"  # Default to Korean
    sourceLanguage: str = "en"  # Default to English (AI 응답 언어)


class HintRequest(BaseModel):
    """힌트 생성 요청"""
    scenarioId: str
    history: List[Dict[str, str]] = []
    lastAiMessage: str = ""
    currentStep: Optional[StepInfo] = None  # 현재 대화 단계
