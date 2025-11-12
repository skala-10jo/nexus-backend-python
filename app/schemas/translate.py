"""
번역 API Pydantic 스키마

번역 요청 및 응답 데이터 검증
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from uuid import UUID


class DetectedTermResponse(BaseModel):
    """탐지된 용어 응답"""
    matched_text: str = Field(..., description="매칭된 텍스트")
    position_start: int = Field(..., description="시작 위치")
    position_end: int = Field(..., description="종료 위치")
    glossary_term_id: Optional[str] = Field(None, description="용어집 ID")
    korean_term: str = Field(..., description="한글 용어")
    english_term: Optional[str] = Field(None, description="영어 용어")
    vietnamese_term: Optional[str] = Field(None, description="베트남어 용어")
    definition: Optional[str] = Field(None, description="용어 정의")
    domain: Optional[str] = Field(None, description="분야")

    class Config:
        json_schema_extra = {
            "example": {
                "matched_text": "클라우드",
                "position_start": 0,
                "position_end": 4,
                "glossary_term_id": "550e8400-e29b-41d4-a716-446655440000",
                "korean_term": "클라우드 컴퓨팅",
                "english_term": "Cloud Computing",
                "vietnamese_term": "Điện toán đám mây",
                "definition": "인터넷을 통해 IT 리소스를 제공하는 서비스",
                "domain": "IT"
            }
        }


class TranslateRequest(BaseModel):
    """번역 요청"""
    text: str = Field(..., min_length=1, max_length=10000, description="번역할 텍스트")
    source_lang: str = Field(..., min_length=2, max_length=10, description="원본 언어 (ko, en, ja, vi 등)")
    target_lang: str = Field(..., min_length=2, max_length=10, description="목표 언어")
    user_id: UUID = Field(..., description="사용자 ID")
    project_id: Optional[UUID] = Field(None, description="프로젝트 ID (선택)")

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        """텍스트 검증"""
        if not v.strip():
            raise ValueError("텍스트가 비어있습니다")
        return v.strip()

    @field_validator('source_lang', 'target_lang')
    @classmethod
    def validate_lang(cls, v: str) -> str:
        """언어 코드 검증"""
        allowed_langs = ['ko', 'en', 'ja', 'vi', 'zh']
        v_lower = v.lower()
        if v_lower not in allowed_langs:
            raise ValueError(f"지원하지 않는 언어입니다. 지원 언어: {', '.join(allowed_langs)}")
        return v_lower

    class Config:
        json_schema_extra = {
            "example": {
                "text": "클라우드 컴퓨팅은 인터넷을 통해 IT 리소스를 제공합니다",
                "source_lang": "ko",
                "target_lang": "en",
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "project_id": "660e8400-e29b-41d4-a716-446655440000"
            }
        }


class TranslateResponse(BaseModel):
    """번역 응답"""
    translation_id: str = Field(..., description="번역 ID")
    original_text: str = Field(..., description="원문")
    translated_text: str = Field(..., description="번역문")
    source_language: str = Field(..., description="원본 언어")
    target_language: str = Field(..., description="목표 언어")
    context_used: bool = Field(..., description="컨텍스트 사용 여부")
    context_summary: Optional[str] = Field(None, description="사용된 컨텍스트 요약")
    detected_terms: List[DetectedTermResponse] = Field(default_factory=list, description="탐지된 전문용어")
    terms_count: int = Field(..., description="탐지된 용어 개수")

    class Config:
        json_schema_extra = {
            "example": {
                "translation_id": "770e8400-e29b-41d4-a716-446655440000",
                "original_text": "클라우드 컴퓨팅은 인터넷을 통해 IT 리소스를 제공합니다",
                "translated_text": "Cloud computing provides IT resources via the internet",
                "source_language": "ko",
                "target_language": "en",
                "context_used": True,
                "context_summary": "이 프로젝트는 AWS 기반 클라우드 인프라...",
                "detected_terms": [],
                "terms_count": 1
            }
        }
