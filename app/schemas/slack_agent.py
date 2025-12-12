"""
Slack Agent Pydantic 스키마

Slack 메시지 번역 및 초안 작성 API 요청/응답 스키마
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class SlackTranslateRequest(BaseModel):
    """Slack 메시지 번역 요청"""
    text: str = Field(..., description="번역할 텍스트")
    source_lang: str = Field(default="auto", description="원본 언어 (auto: 자동 감지)")
    target_lang: str = Field(..., description="목표 언어 (ko, en, ja, vi, zh)")


class SlackTranslateResponse(BaseModel):
    """Slack 메시지 번역 응답"""
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str


class SlackDraftRequest(BaseModel):
    """Slack 초안 작성 요청"""
    message: str = Field(..., description="작성하고 싶은 내용/의도")
    language: str = Field(default="ko", description="목표 언어 (ko, en)")
    keywords: Optional[List[str]] = Field(default=None, description="RAG 검색 키워드 (선택)")


class BizGuideSuggestion(BaseModel):
    """비즈니스 표현 제안"""
    text: str
    chapter: Optional[str] = None
    section: Optional[str] = None
    score: float


class SlackDraftResponse(BaseModel):
    """Slack 초안 작성 응답"""
    draft: str = Field(..., description="작성된 초안")
    suggestions: List[BizGuideSuggestion] = Field(default_factory=list, description="참고된 비즈니스 표현")
    status: str


class SlackChatMessage(BaseModel):
    """대화 메시지"""
    role: str = Field(..., description="역할 (user 또는 assistant)")
    content: str = Field(..., description="메시지 내용")


class SlackChatRequest(BaseModel):
    """Slack 챗봇 요청 (세션 기반)"""
    message: str = Field(..., description="사용자 메시지")
    session_id: Optional[str] = Field(default=None, description="세션 ID (없으면 새 세션 생성)")
    language: str = Field(default="ko", description="기본 언어 (ko, en)")


class SlackChatResponse(BaseModel):
    """Slack 챗봇 응답"""
    session_id: str = Field(..., description="세션 ID")
    message: str = Field(..., description="AI 응답 메시지")
    draft: Optional[str] = Field(default=None, description="생성된 초안 (있는 경우)")
    action_type: str = Field(..., description="수행된 작업 (draft, translate, refine, general)")
    suggestions: List[BizGuideSuggestion] = Field(default_factory=list, description="참고된 비즈니스 표현")
