"""
Pydantic schemas for Mail Agent API.

Request/Response models for email embedding and search endpoints.

Author: NEXUS Team
Date: 2025-01-12
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class GenerateEmbeddingsRequest(BaseModel):
    """단일 메일 임베딩 생성 요청"""
    email_id: str = Field(..., description="이메일 ID")


class BatchGenerateRequest(BaseModel):
    """일괄 임베딩 생성 요청"""
    user_id: str = Field(..., description="사용자 ID")


class SearchRequest(BaseModel):
    """메일 검색 요청"""
    query: str = Field(..., min_length=2, description="검색 쿼리 (자연어, 최소 2자)")
    user_id: str = Field(..., description="사용자 ID")
    top_k: int = Field(default=10, ge=1, le=50, description="최대 결과 개수 (1-50)")
    folder: Optional[str] = Field(None, description="폴더 필터 (Inbox/SentItems)")
    date_from: Optional[str] = Field(None, description="시작 날짜 (YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="종료 날짜 (YYYY-MM-DD)")


class EmailSearchResult(BaseModel):
    """메일 검색 결과 아이템"""
    email_id: str
    subject: str
    from_name: Optional[str]
    to_recipients: Optional[str]
    folder: str
    date: datetime
    similarity: float = Field(..., ge=0.0, le=1.0, description="유사도 점수 (0.0-1.0)")
    matched_chunk: str = Field(..., description="매칭된 청크 미리보기")


class SearchResponse(BaseModel):
    """메일 검색 응답"""
    success: bool
    data: List[EmailSearchResult]
    count: int


class GenerateEmbeddingsResponse(BaseModel):
    """임베딩 생성 응답"""
    status: str
    chunks_created: Optional[int] = None
    reason: Optional[str] = None
    error: Optional[str] = None


class BatchGenerateResponse(BaseModel):
    """일괄 임베딩 생성 응답"""
    status: str
    total: int
    processed: int
    skipped: int
    failed: int


class ChatMessage(BaseModel):
    """채팅 메시지"""
    role: str = Field(..., description="메시지 역할 (user/assistant)")
    content: str = Field(..., description="메시지 내용")


class ChatRequest(BaseModel):
    """메일 검색 챗봇 요청"""
    message: str = Field(..., min_length=1, description="사용자 메시지")
    user_id: str = Field(..., description="사용자 ID")
    conversation_history: Optional[List[ChatMessage]] = Field(default=[], description="대화 히스토리")


class ChatResponse(BaseModel):
    """메일 검색 챗봇 응답"""
    query: Optional[str] = Field(None, description="추출된 검색 쿼리")
    folder: Optional[str] = Field(None, description="폴더 필터")
    date_from: Optional[str] = Field(None, description="시작 날짜")
    date_to: Optional[str] = Field(None, description="종료 날짜")
    needs_search: bool = Field(..., description="검색이 필요한지 여부")
    response: str = Field(..., description="사용자에게 보여줄 응답")
    search_results: Optional[List[EmailSearchResult]] = Field(None, description="검색 결과 (검색 수행된 경우)")
