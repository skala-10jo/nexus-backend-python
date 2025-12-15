"""
Voice STT REST API 엔드포인트

Azure Speech SDK를 사용한 음성-텍스트 변환 REST API
아키텍처: API → Service → Agent
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import logging
from app.services.voice_stt_service import VoiceSTTService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Voice STT REST"])

# Service 인스턴스 (싱글톤)
_stt_service: VoiceSTTService | None = None


def get_stt_service() -> VoiceSTTService:
    """VoiceSTTService 싱글톤 인스턴스 반환"""
    global _stt_service
    if _stt_service is None:
        _stt_service = VoiceSTTService()
    return _stt_service


@router.post(
    "/stt",
    response_model=dict,
    summary="음성을 텍스트로 변환 (STT)",
    description="""
    Azure Speech SDK를 사용하여 음성 파일을 텍스트로 변환합니다.

    - 지원 형식: WAV, MP3, OGG, WebM
    - BCP-47 언어 코드 사용 (예: ko-KR, en-US, ja-JP)
    - 최대 파일 크기: 25MB
    """
)
async def speech_to_text(
    file: UploadFile = File(..., description="음성 파일 (WAV/MP3/OGG/WebM)"),
    language: str = Form(default="ko-KR", description="BCP-47 언어 코드")
):
    """
    음성을 텍스트로 변환

    Args:
        file: 음성 파일
        language: BCP-47 언어 코드 (기본값: ko-KR)

    Returns:
        dict: STT 결과 { text, confidence, language }
    """
    try:
        # 파일 크기 검증 (25MB 제한)
        MAX_SIZE = 25 * 1024 * 1024
        contents = await file.read()

        if len(contents) > MAX_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"파일 크기가 너무 큽니다. 최대 25MB까지 허용됩니다."
            )

        logger.info(f"STT request: filename={file.filename}, size={len(contents)} bytes, language={language}")

        # Service 계층을 통해 STT 처리 (아키텍처 규칙 준수: API → Service → Agent)
        service = get_stt_service()
        result = await service.transcribe_audio(
            audio_data=contents,
            language=language
        )

        logger.info(f"STT success: text='{result.get('text', '')[:50]}...'")

        return JSONResponse(
            content={
                "success": True,
                "message": "음성 인식 완료",
                "data": {
                    "text": result.get("text", ""),
                    "confidence": result.get("confidence", 0.0),
                    "language": result.get("language", language)
                }
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"음성 인식 실패: {str(e)}")
