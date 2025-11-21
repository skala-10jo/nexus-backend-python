"""
Expression Speech API Schemas

발음 평가 및 TTS API의 요청/응답 스키마
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID


class WordScore(BaseModel):
    """단어별 발음 점수"""
    word: str = Field(..., description="단어")
    accuracy_score: float = Field(..., description="정확도 점수 (0-100)")
    error_type: str = Field(..., description="오류 유형 (None, Mispronunciation, Omission, Insertion)")


class PronunciationAssessmentResponse(BaseModel):
    """발음 평가 응답"""
    pronunciation_score: float = Field(..., description="전체 발음 점수 (0-100)")
    accuracy_score: float = Field(..., description="정확도 점수 (0-100)")
    fluency_score: float = Field(..., description="유창성 점수 (0-100)")
    completeness_score: float = Field(..., description="완성도 점수 (0-100)")
    recognized_text: str = Field(..., description="인식된 텍스트")
    words: List[WordScore] = Field(default_factory=list, description="단어별 상세 평가")
    error: Optional[str] = Field(None, description="에러 메시지 (있을 경우)")


class TTSRequest(BaseModel):
    """TTS 요청 (expression_id 사용)"""
    expression_id: UUID = Field(..., description="표현 ID")
    voice_name: str = Field(
        default="en-US-JennyNeural",
        description="음성 이름 (en-US-JennyNeural, en-US-GuyNeural 등)"
    )


class TTSTextRequest(BaseModel):
    """TTS 요청 (직접 텍스트 입력)"""
    text: str = Field(..., description="음성으로 변환할 텍스트", min_length=1)
    voice_name: str = Field(
        default="en-US-JennyNeural",
        description="음성 이름 (en-US-JennyNeural, en-US-GuyNeural 등)"
    )


class TTSResponse(BaseModel):
    """TTS 응답 (오디오 파일은 바이너리로 반환)"""
    duration_ms: int = Field(..., description="재생 시간 (밀리초)")
    audio_format: str = Field(..., description="오디오 형식 (mp3, wav)")
