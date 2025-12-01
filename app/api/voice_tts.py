"""
Voice TTS API 엔드포인트

Azure Speech SDK를 사용한 텍스트 음성 변환(TTS) API
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, JSONResponse
import logging
from agent.tts.azure_tts_agent import AzureTTSAgent
from app.schemas.voice import TTSRequest, TTSResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tts", tags=["Voice TTS"])


@router.post(
    "",
    summary="텍스트를 음성으로 변환",
    description="""
    Azure Neural Voices를 사용하여 텍스트를 음성으로 변환합니다.

    - SSML 지원 (속도, 음높이, 음량 제어)
    - 다양한 뉴럴 음성 지원 (20+ 언어)
    - WAV 형식 오디오 반환
    """,
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": "WAV 형식 오디오 파일 반환"
        }
    }
)
async def text_to_speech(request: TTSRequest):
    """
    텍스트를 음성으로 변환

    Args:
        request: TTS 요청 (text, voice_name, rate, pitch, volume)

    Returns:
        Response: WAV 형식 오디오 데이터
    """
    try:
        logger.info(f"TTS request: voice={request.voice_name}, text_length={len(request.text)}")

        # TTS Agent 실행
        agent = AzureTTSAgent.get_instance()
        audio_data = await agent.process(
            text=request.text,
            voice_name=request.voice_name,
            rate=request.rate,
            pitch=request.pitch,
            volume=request.volume
        )

        if not audio_data:
            raise HTTPException(status_code=500, detail="음성 생성 실패: 오디오 데이터가 비어있습니다")

        logger.info(f"TTS success: generated {len(audio_data)} bytes")

        # WAV 형식으로 응답 반환
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="tts_output.wav"',
                "Content-Length": str(len(audio_data))
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"음성 생성 실패: {str(e)}")


@router.post(
    "/metadata",
    response_model=dict,
    summary="TTS 메타데이터 (오디오 미포함)",
    description="""
    TTS 메타데이터만 반환하고 실제 오디오는 생성하지 않습니다.

    - 테스트용 엔드포인트
    - 요청 검증 및 음성 정보 확인
    """
)
async def text_to_speech_metadata(request: TTSRequest):
    """
    TTS 메타데이터 반환 (테스트용)

    Args:
        request: TTS 요청

    Returns:
        dict: TTS 메타데이터
    """
    try:
        logger.info(f"TTS metadata request: voice={request.voice_name}")

        # 응답 구성 (실제 TTS는 수행하지 않음)
        response = TTSResponse(
            text=request.text,
            voice_name=request.voice_name,
            audio_format="audio/wav",
            audio_size=0  # 실제 생성하지 않음
        )

        return JSONResponse(
            content={
                "success": True,
                "message": "TTS 메타데이터 조회 완료",
                "data": response.model_dump()
            }
        )

    except Exception as e:
        logger.error(f"TTS metadata failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"메타데이터 조회 실패: {str(e)}")


@router.get(
    "/voices",
    summary="사용 가능한 음성 목록 조회",
    description="Azure Neural Voices 목록을 반환합니다"
)
async def get_available_voices():
    """
    사용 가능한 Azure Neural Voices 목록 조회

    Returns:
        dict: 음성 목록 (언어별)
    """
    # Azure Neural Voices 주요 목록
    voices = {
        "ko-KR": [
            {"name": "ko-KR-SunHiNeural", "gender": "Female", "description": "밝고 친근한 목소리"},
            {"name": "ko-KR-InJoonNeural", "gender": "Male", "description": "차분하고 안정적인 목소리"},
            {"name": "ko-KR-BongJinNeural", "gender": "Male", "description": "명확하고 전문적인 목소리"},
            {"name": "ko-KR-GookMinNeural", "gender": "Male", "description": "자연스러운 대화체"},
            {"name": "ko-KR-JiMinNeural", "gender": "Female", "description": "부드럽고 따뜻한 목소리"}
        ],
        "en-US": [
            {"name": "en-US-JennyNeural", "gender": "Female", "description": "Friendly and conversational"},
            {"name": "en-US-GuyNeural", "gender": "Male", "description": "Clear and professional"},
            {"name": "en-US-AriaNeural", "gender": "Female", "description": "Natural and warm"},
            {"name": "en-US-DavisNeural", "gender": "Male", "description": "Confident and engaging"},
            {"name": "en-US-AmberNeural", "gender": "Female", "description": "Casual and friendly"}
        ],
        "ja-JP": [
            {"name": "ja-JP-NanamiNeural", "gender": "Female", "description": "明るく親しみやすい"},
            {"name": "ja-JP-KeitaNeural", "gender": "Male", "description": "落ち着いた自然な声"},
            {"name": "ja-JP-AoiNeural", "gender": "Female", "description": "柔らかく温かみのある声"}
        ],
        "zh-CN": [
            {"name": "zh-CN-XiaoxiaoNeural", "gender": "Female", "description": "自然亲切的声音"},
            {"name": "zh-CN-YunxiNeural", "gender": "Male", "description": "清晰专业的声音"},
            {"name": "zh-CN-YunyangNeural", "gender": "Male", "description": "温暖友好的声音"}
        ],
        "es-ES": [
            {"name": "es-ES-ElviraNeural", "gender": "Female", "description": "Voz cálida y amigable"},
            {"name": "es-ES-AlvaroNeural", "gender": "Male", "description": "Voz clara y profesional"}
        ],
        "fr-FR": [
            {"name": "fr-FR-DeniseNeural", "gender": "Female", "description": "Voix naturelle et chaleureuse"},
            {"name": "fr-FR-HenriNeural", "gender": "Male", "description": "Voix claire et professionnelle"}
        ],
        "de-DE": [
            {"name": "de-DE-KatjaNeural", "gender": "Female", "description": "Freundliche und natürliche Stimme"},
            {"name": "de-DE-ConradNeural", "gender": "Male", "description": "Klare und professionelle Stimme"}
        ]
    }

    return JSONResponse(
        content={
            "success": True,
            "message": "사용 가능한 음성 목록 조회 완료",
            "data": {
                "voices": voices,
                "total_languages": len(voices),
                "total_voices": sum(len(v) for v in voices.values())
            }
        }
    )
