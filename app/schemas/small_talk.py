"""
Small Talk API Pydantic schemas for request/response validation.
DB 없이 메모리 기반으로 동작하는 스몰토크 대화용 스키마.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class StartSmallTalkRequest(BaseModel):
    """스몰토크 대화 시작 요청 (body 없음)"""
    pass


class StartSmallTalkResponse(BaseModel):
    """스몰토크 대화 시작 응답"""
    initial_message: str


class SendSmallTalkMessageRequest(BaseModel):
    """스몰토크 메시지 전송 요청"""
    message: str = Field(..., description="사용자 메시지")
    history: List[Dict[str, str]] = Field(default=[], description="대화 히스토리")


class SendSmallTalkMessageResponse(BaseModel):
    """스몰토크 메시지 전송 응답"""
    ai_message: str


class SmallTalkFeedbackRequest(BaseModel):
    """스몰토크 피드백 요청"""
    message: str = Field(..., description="피드백 받을 사용자 메시지")
    history: List[Dict[str, str]] = Field(default=[], description="대화 히스토리 (맥락 파악용)")
    audio_data: Optional[str] = Field(default=None, description="Base64 인코딩된 오디오 데이터")


class SmallTalkFeedbackResponse(BaseModel):
    """스몰토크 피드백 응답"""
    grammar_corrections: List[str] = []
    suggestions: List[str] = []
    score: int
    score_breakdown: Dict[str, int]
    pronunciation_feedback: Optional[List[str]] = None
    pronunciation_details: Optional[Dict] = None


class SmallTalkHintRequest(BaseModel):
    """스몰토크 힌트 요청"""
    history: List[Dict[str, str]] = Field(default=[], description="대화 히스토리")
    last_ai_message: str = Field(default="", description="마지막 AI 메시지")
    hint_count: int = Field(default=3, description="생성할 힌트 개수")


class SmallTalkHintResponse(BaseModel):
    """스몰토크 힌트 응답"""
    hints: List[str]
    explanations: List[str]
