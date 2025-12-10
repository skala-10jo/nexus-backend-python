"""
Video Translation Pydantic 스키마

영상 번역 관련 요청/응답 스키마 정의
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from uuid import UUID
from datetime import datetime


# ========== Request Schemas ==========

class VideoSTTRequest(BaseModel):
    """STT 처리 요청"""
    video_file_id: UUID = Field(..., description="영상 파일 ID")
    source_language: str = Field(..., description="음성 언어 코드 (ko, en, ja, vi)")

    @validator("source_language")
    def validate_language(cls, v):
        allowed_languages = ["ko", "en", "ja", "vi", "zh"]
        if v not in allowed_languages:
            raise ValueError(f"지원하지 않는 언어입니다. 허용: {', '.join(allowed_languages)}")
        return v


class VideoTranslationRequest(BaseModel):
    """영상 자막 번역 요청"""
    video_file_id: UUID = Field(..., description="영상 파일 ID")
    project_id: Optional[UUID] = Field(
        default=None,
        description="프로젝트 ID (용어집 컨텍스트 자동 조회) - Text.vue 방식과 동일"
    )
    source_language: str = Field(..., description="원본 언어 코드")
    target_language: str = Field(..., description="목표 언어 코드")

    @validator("source_language", "target_language")
    def validate_language(cls, v):
        allowed_languages = ["ko", "en", "ja", "vi", "zh"]
        if v not in allowed_languages:
            raise ValueError(f"지원하지 않는 언어입니다. 허용: {', '.join(allowed_languages)}")
        return v


# ========== Response Schemas ==========

class SubtitleSegmentResponse(BaseModel):
    """자막 세그먼트 응답"""
    sequence_number: int = Field(..., description="세그먼트 순서")
    start_time_ms: int = Field(..., description="시작 시간 (밀리초)")
    end_time_ms: int = Field(..., description="종료 시간 (밀리초)")
    text: str = Field(..., description="자막 텍스트")
    confidence: Optional[float] = Field(None, description="신뢰도 (0.0 ~ 1.0)")

    class Config:
        from_attributes = True


class MultilingualSubtitleSegmentResponse(BaseModel):
    """다국어 자막 세그먼트 응답"""
    sequence_number: int = Field(..., description="세그먼트 순서")
    start_time_ms: int = Field(..., description="시작 시간 (밀리초)")
    end_time_ms: int = Field(..., description="종료 시간 (밀리초)")
    original_text: str = Field(..., description="원본 텍스트")
    translations: dict = Field(default_factory=dict, description="번역 텍스트 (언어별)")
    confidence: Optional[float] = Field(None, description="신뢰도 (0.0 ~ 1.0)")
    detected_terms: Optional[list] = Field(None, description="감지된 전문용어")

    class Config:
        from_attributes = True


class VideoSTTResponse(BaseModel):
    """STT 처리 응답"""
    video_file_id: UUID = Field(..., description="영상 파일 ID")
    language: str = Field(..., description="언어 코드")
    segments: List[SubtitleSegmentResponse] = Field(..., description="자막 세그먼트 리스트")
    total_segments: int = Field(..., description="전체 세그먼트 수")
    created_at: Optional[datetime] = Field(None, description="생성 시간")

    class Config:
        from_attributes = True


class MultilingualSubtitlesResponse(BaseModel):
    """다국어 자막 조회 응답"""
    video_file_id: UUID = Field(..., description="영상 파일 ID")
    original_language: str = Field(..., description="원본 언어 코드")
    available_languages: List[str] = Field(..., description="사용 가능한 언어 목록")
    segments: List[MultilingualSubtitleSegmentResponse] = Field(..., description="자막 세그먼트")
    total_segments: int = Field(..., description="전체 세그먼트 수")

    class Config:
        from_attributes = True


class TranslatedSegmentResponse(BaseModel):
    """번역된 자막 세그먼트 응답"""
    sequence_number: int = Field(..., description="세그먼트 순서")
    start_time_ms: int = Field(..., description="시작 시간 (밀리초)")
    end_time_ms: int = Field(..., description="종료 시간 (밀리초)")
    original_text: str = Field(..., description="원본 텍스트")
    translated_text: str = Field(..., description="번역된 텍스트")
    confidence: Optional[float] = Field(None, description="신뢰도 (0.0 ~ 1.0)")

    class Config:
        from_attributes = True


class VideoTranslationResponse(BaseModel):
    """영상 자막 번역 응답"""
    video_file_id: UUID = Field(..., description="영상 파일 ID")
    source_language: str = Field(..., description="원본 언어")
    target_language: str = Field(..., description="목표 언어")
    segments: List[TranslatedSegmentResponse] = Field(..., description="번역된 자막 세그먼트")
    total_segments: int = Field(..., description="전체 세그먼트 수")
    context_used: bool = Field(..., description="컨텍스트 사용 여부")
    context_document_count: int = Field(..., description="사용된 컨텍스트 문서 수")
    created_at: Optional[datetime] = Field(None, description="생성 시간")

    class Config:
        from_attributes = True


class SubtitleDownloadResponse(BaseModel):
    """자막 다운로드 정보 응답"""
    subtitle_id: UUID = Field(..., description="자막 ID")
    file_path: str = Field(..., description="SRT 파일 경로")
    language: str = Field(..., description="언어 코드")
    subtitle_type: str = Field(..., description="자막 타입")
    file_size_bytes: int = Field(..., description="파일 크기 (바이트)")
