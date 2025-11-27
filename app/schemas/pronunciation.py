"""
Pronunciation Assessment API 스키마

발음 평가 요청/응답을 위한 Pydantic 모델
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class PhonemeAssessment(BaseModel):
    """음소 단위 평가 결과"""
    phoneme: str = Field(..., description="음소 기호 (IPA)")
    accuracy_score: float = Field(..., description="정확도 점수 (0-100)", ge=0, le=100)
    NBestPhonemes: Optional[List[dict]] = Field(default=[], description="대안 음소 후보")


class WordAssessment(BaseModel):
    """단어 단위 평가 결과"""
    word: str = Field(..., description="평가된 단어")
    accuracy_score: float = Field(..., description="정확도 점수 (0-100)", ge=0, le=100)
    error_type: str = Field(
        default="None",
        description="오류 유형 (None/Mispronunciation/Omission/Insertion)"
    )
    phonemes: List[PhonemeAssessment] = Field(default=[], description="음소별 평가")


class PronunciationAssessmentRequest(BaseModel):
    """발음 평가 요청"""
    audio_data: str = Field(..., description="Base64 인코딩된 오디오 데이터")
    reference_text: str = Field(
        ...,
        description="평가 기준 텍스트 (사용자가 읽어야 할 텍스트)",
        min_length=1,
        max_length=1000
    )
    language: str = Field(
        default="en-US",
        description="언어 코드 (BCP-47 형식)",
        pattern=r"^[a-z]{2}-[A-Z]{2}$"
    )
    granularity: str = Field(
        default="Phoneme",
        description="평가 세분화 수준 (Phoneme/Word/FullText)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "audio_data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
                "reference_text": "Hello, how are you today?",
                "language": "en-US",
                "granularity": "Phoneme"
            }
        }


class PronunciationAssessmentResponse(BaseModel):
    """발음 평가 응답"""
    accuracy_score: float = Field(
        ...,
        description="발음 정확도 점수 (0-100)",
        ge=0,
        le=100
    )
    fluency_score: float = Field(
        ...,
        description="유창성 점수 (0-100)",
        ge=0,
        le=100
    )
    completeness_score: float = Field(
        ...,
        description="완성도 점수 (0-100)",
        ge=0,
        le=100
    )
    prosody_score: float = Field(
        ...,
        description="운율 점수 (강세, 억양) (0-100)",
        ge=0,
        le=100
    )
    pronunciation_score: float = Field(
        ...,
        description="전체 발음 점수 (0-100)",
        ge=0,
        le=100
    )
    recognized_text: str = Field(..., description="실제 인식된 텍스트")
    reference_text: str = Field(..., description="기준 텍스트")
    words: List[WordAssessment] = Field(
        default=[],
        description="단어별 상세 평가"
    )
    error: Optional[str] = Field(default=None, description="에러 메시지 (있는 경우)")

    class Config:
        json_schema_extra = {
            "example": {
                "accuracy_score": 85.5,
                "fluency_score": 90.0,
                "completeness_score": 100.0,
                "prosody_score": 88.0,
                "pronunciation_score": 87.2,
                "recognized_text": "Hello, how are you today?",
                "reference_text": "Hello, how are you today?",
                "words": [
                    {
                        "word": "Hello",
                        "accuracy_score": 90.0,
                        "error_type": "None",
                        "phonemes": [
                            {
                                "phoneme": "h",
                                "accuracy_score": 95.0,
                                "NBestPhonemes": []
                            },
                            {
                                "phoneme": "ə",
                                "accuracy_score": 85.0,
                                "NBestPhonemes": []
                            }
                        ]
                    }
                ]
            }
        }


class PronunciationFeedbackSummary(BaseModel):
    """발음 피드백 요약 (프론트엔드용)"""
    overall_score: float = Field(..., description="전체 점수 (0-100)")
    accuracy: float = Field(..., description="정확도")
    fluency: float = Field(..., description="유창성")
    prosody: float = Field(..., description="운율")

    # 문제 단어 (정확도 70% 이하)
    problem_words: List[str] = Field(default=[], description="발음 개선 필요 단어")

    # 음소 문제 (정확도 70% 이하)
    problem_phonemes: List[dict] = Field(
        default=[],
        description="발음 개선 필요 음소"
    )

    # 전반적인 피드백 메시지
    feedback_message: str = Field(..., description="피드백 메시지")

    class Config:
        json_schema_extra = {
            "example": {
                "overall_score": 87.2,
                "accuracy": 85.5,
                "fluency": 90.0,
                "prosody": 88.0,
                "problem_words": ["world", "today"],
                "problem_phonemes": [
                    {"word": "world", "phoneme": "r", "accuracy": 65.0}
                ],
                "feedback_message": "Good pronunciation! Focus on improving 'r' sound in 'world'."
            }
        }
