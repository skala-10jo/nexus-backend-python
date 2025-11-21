"""
Voice STT API 엔드포인트

음성을 텍스트로 변환하는 API
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse
import logging
from app.services.voice_stt_service import VoiceSTTService
from app.core.audio_converter import convert_to_wav, detect_audio_format
from app.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["Voice STT"])


@router.post(
    "/stt",
    response_model=dict,
    summary="음성을 텍스트로 변환",
    description="""
    음성 파일을 텍스트로 변환합니다.

    - 지원 형식: WAV, MP3, OGG, WEBM
    - 최대 파일 크기: 10MB
    - 지원 언어: ko-KR (한국어), en-US (영어), ja-JP (일본어) 등
    - **인증 필요**: JWT 토큰 (Authorization: Bearer <token>)

    **사용 예시:**
    ```javascript
    const formData = new FormData()
    formData.append('audio', audioBlob, 'recording.webm')
    formData.append('language', 'ko-KR')

    const response = await fetch('/api/ai/voice/stt', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${token}`
        },
        body: formData
    })
    ```
    """
)
async def speech_to_text(
    audio: UploadFile = File(..., description="음성 파일"),
    language: str = Form(default="en-US", description="언어 코드 (현재 en-US만 지원)"),
    user: dict = Depends(get_current_user)
):
    """
    음성을 텍스트로 변환

    Args:
        audio: 음성 파일 (WAV, MP3, OGG, WEBM 등)
        language: BCP-47 언어 코드 (기본값: ko-KR)

    Returns:
        dict: {
            "success": bool,
            "message": str,
            "data": {
                "text": str,           # 인식된 텍스트
                "confidence": float,   # 신뢰도 (0.0 ~ 1.0)
                "language": str        # 언어 코드
            }
        }

    Raises:
        HTTPException: STT 처리 실패 시
    """
    try:
        user_id = user["user_id"]
        username = user.get("username", "unknown")
        logger.info(f"STT 요청 수신: user={username}, filename={audio.filename}, language={language}")

        # 파일 크기 확인 (10MB 제한)
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        audio_data = await audio.read()

        if len(audio_data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"파일 크기가 너무 큽니다. 최대 {MAX_FILE_SIZE / 1024 / 1024}MB까지 지원합니다."
            )

        logger.info(f"파일 크기: {len(audio_data) / 1024:.2f}KB")

        # 오디오 형식 감지 및 WAV로 변환
        audio_format = detect_audio_format(audio.filename)
        logger.info(f"Detected audio format: {audio_format}")

        if audio_format != 'wav':
            try:
                audio_data = convert_to_wav(audio_data, audio_format)
                logger.info(f"Audio converted to WAV: {len(audio_data) / 1024:.2f}KB")
            except Exception as conv_error:
                logger.error(f"Audio conversion failed: {str(conv_error)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"오디오 형식 변환 실패: {str(conv_error)}"
                )

        # Service 계층 호출 (비즈니스 로직 처리)
        service = VoiceSTTService()
        result = await service.transcribe_audio(
            audio_data=audio_data,
            language=language,
            enable_diarization=False,  # 회화 연습에서는 화자 분리 불필요
            user_id=str(user_id)  # 사용자 ID 전달 (향후 사용량 추적용)
        )

        logger.info(
            f"STT 성공: user={username}, "
            f"text_length={len(result['text'])}, "
            f"confidence={result.get('confidence', 0.0)}"
        )

        return JSONResponse(
            content={
                "success": True,
                "message": "음성 인식 완료",
                "data": {
                    "text": result["text"],
                    "confidence": result.get("confidence", 1.0),
                    "language": result["language"]
                }
            }
        )

    except HTTPException:
        # HTTPException은 그대로 re-raise
        raise

    except Exception as e:
        logger.error(f"STT 처리 실패: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"음성 인식 실패: {str(e)}"
        )


@router.get(
    "/supported-languages",
    summary="지원하는 언어 목록 조회",
    description="Azure Speech Service에서 지원하는 언어 코드 목록을 반환합니다"
)
async def get_supported_languages():
    """
    지원하는 언어 목록 조회

    Returns:
        dict: 지원 언어 목록
    """
    # Service 계층 호출
    service = VoiceSTTService()
    supported_languages = await service.get_supported_languages()

    return JSONResponse(
        content={
            "success": True,
            "message": "지원 언어 목록 조회 완료",
            "data": {
                "languages": supported_languages
            }
        }
    )
