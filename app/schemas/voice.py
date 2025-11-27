"""
Voice API 스키마 (요청/응답 검증)

STT, Translation, TTS API의 요청/응답을 검증하는 Pydantic 스키마
"""
from pydantic import BaseModel, Field
from typing import Optional, List


# ========== STT 스키마 ==========

class STTRequest(BaseModel):
    """STT 요청 스키마 (단일 파일 업로드)"""

    language: str = Field(
        default="ko-KR",
        description="BCP-47 언어 코드 (예: ko-KR, en-US, ja-JP)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "language": "ko-KR"
            }
        }


class STTResponse(BaseModel):
    """STT 응답 스키마"""

    text: str = Field(..., description="인식된 텍스트")
    confidence: float = Field(..., description="신뢰도 (0.0 ~ 1.0)")
    language: str = Field(..., description="인식된 언어 (BCP-47)")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "안녕하세요. 반갑습니다.",
                "confidence": 0.95,
                "language": "ko-KR"
            }
        }


class TranslationItem(BaseModel):
    """번역 결과 아이템"""

    lang: str = Field(..., description="목표 언어 (BCP-47 코드, 예: en-US, ja-JP)")
    text: str = Field(..., description="번역된 텍스트")

    class Config:
        json_schema_extra = {
            "example": {
                "lang": "en-US",
                "text": "Hello"
            }
        }


class STTStreamMessage(BaseModel):
    """WebSocket STT 스트리밍 메시지"""

    type: str = Field(..., description="메시지 타입 (recognizing, recognized, error, end)")
    text: Optional[str] = Field(None, description="인식된 텍스트")
    confidence: Optional[float] = Field(None, description="신뢰도")
    error: Optional[str] = Field(None, description="에러 메시지")
    detected_language: Optional[str] = Field(
        None,
        description="자동 감지된 언어 (BCP-47 코드, 예: ko-KR, en-US, ja-JP, vi-VN)"
    )
    translations: Optional[List[TranslationItem]] = Field(
        None,
        description="번역 결과 리스트"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "type": "recognized",
                "text": "안녕하세요",
                "confidence": 0.95,
                "detected_language": "ko-KR",
                "translations": [
                    {"lang": "en-US", "text": "Hello"},
                    {"lang": "ja-JP", "text": "こんにちは"}
                ],
                "error": None
            }
        }


# ========== Translation 스키마 ==========

class TranslationRequest(BaseModel):
    """번역 요청 스키마"""

    text: str = Field(..., description="번역할 텍스트", min_length=1)
    source_lang: str = Field(
        default="ko",
        description="원본 언어 (ISO 639-1 코드, 예: ko, en, ja, zh-Hans)"
    )
    target_lang: str = Field(
        ...,
        description="목표 언어 (ISO 639-1 코드)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "안녕하세요. 반갑습니다.",
                "source_lang": "ko",
                "target_lang": "en"
            }
        }


class TranslationResponse(BaseModel):
    """번역 응답 스키마"""

    original_text: str = Field(..., description="원본 텍스트")
    translated_text: str = Field(..., description="번역된 텍스트")
    source_lang: str = Field(..., description="원본 언어 (ISO 639-1)")
    target_lang: str = Field(..., description="목표 언어 (ISO 639-1)")

    class Config:
        json_schema_extra = {
            "example": {
                "original_text": "안녕하세요. 반갑습니다.",
                "translated_text": "Hello. Nice to meet you.",
                "source_lang": "ko",
                "target_lang": "en"
            }
        }


class BatchTranslationRequest(BaseModel):
    """일괄 번역 요청 스키마"""

    texts: List[str] = Field(
        ...,
        description="번역할 텍스트 리스트 (최대 100개)",
        max_length=100
    )
    source_lang: str = Field(
        default="ko",
        description="원본 언어 (ISO 639-1 코드)"
    )
    target_lang: str = Field(
        ...,
        description="목표 언어 (ISO 639-1 코드)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "texts": ["안녕하세요", "반갑습니다", "감사합니다"],
                "source_lang": "ko",
                "target_lang": "en"
            }
        }


class BatchTranslationResponse(BaseModel):
    """일괄 번역 응답 스키마"""

    translations: List[TranslationResponse] = Field(
        ...,
        description="번역 결과 리스트"
    )
    total_count: int = Field(..., description="총 번역 개수")

    class Config:
        json_schema_extra = {
            "example": {
                "translations": [
                    {
                        "original_text": "안녕하세요",
                        "translated_text": "Hello",
                        "source_lang": "ko",
                        "target_lang": "en"
                    }
                ],
                "total_count": 3
            }
        }


# ========== TTS 스키마 ==========

class TTSRequest(BaseModel):
    """TTS 요청 스키마"""

    text: str = Field(..., description="음성으로 변환할 텍스트", min_length=1)
    voice_name: str = Field(
        default="ko-KR-SunHiNeural",
        description="Azure 뉴럴 음성 이름 (예: ko-KR-SunHiNeural, en-US-JennyNeural)"
    )
    rate: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="말하기 속도 (0.5 ~ 2.0)"
    )
    pitch: int = Field(
        default=0,
        ge=-50,
        le=50,
        description="음높이 (-50% ~ +50%)"
    )
    volume: int = Field(
        default=100,
        ge=0,
        le=100,
        description="음량 (0 ~ 100)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "안녕하세요. 반갑습니다.",
                "voice_name": "ko-KR-SunHiNeural",
                "rate": 1.0,
                "pitch": 0,
                "volume": 100
            }
        }


class TTSResponse(BaseModel):
    """TTS 응답 스키마 (메타데이터만, 실제 오디오는 바이너리 응답)"""

    text: str = Field(..., description="변환된 텍스트")
    voice_name: str = Field(..., description="사용된 음성 이름")
    audio_format: str = Field(default="audio/wav", description="오디오 형식")
    audio_size: int = Field(..., description="오디오 데이터 크기 (bytes)")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "안녕하세요. 반갑습니다.",
                "voice_name": "ko-KR-SunHiNeural",
                "audio_format": "audio/wav",
                "audio_size": 102400
            }
        }


# ========== 공통 응답 스키마 ==========

class VoiceApiResponse(BaseModel):
    """Voice API 공통 응답 래퍼"""

    success: bool = Field(..., description="성공 여부")
    message: str = Field(..., description="응답 메시지")
    data: Optional[dict] = Field(None, description="응답 데이터")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "처리 완료",
                "data": {}
            }
        }
