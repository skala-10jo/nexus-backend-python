"""
Azure Speech Service API 엔드포인트

브라우저에서 Azure Speech SDK를 사용하기 위한 토큰 발급 API
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import logging
from agent.stt_translation.azure_speech_agent import AzureSpeechAgent
from app.schemas.azure_speech import SpeechTokenResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/speech", tags=["Azure Speech"])


@router.options("/token")
async def options_speech_token():
    """브라우저 preflight용 OPTIONS 허용"""
    return JSONResponse(content=None, status_code=200)


@router.get(
    "/token",
    response_model=dict,
    summary="Azure Speech 인증 토큰 발급",
    description="""
    브라우저에서 Azure Speech SDK를 사용하기 위한 인증 토큰을 발급합니다.

    - 토큰 유효 시간: 10분
    - 클라이언트는 토큰을 캐싱하여 재사용 권장
    - STT, Translation, TTS 작업에 사용됨

    **보안**: Azure 구독 키 노출 방지를 위해 백엔드에서 토큰 발급
    """
)
async def get_speech_token(request: Request):
    """
    Azure Speech 인증 토큰 발급

    Returns:
        dict: 토큰 및 리전 정보

    Raises:
        HTTPException: 토큰 발급 실패 시
    """
    try:
        # 싱글톤 Agent 인스턴스 가져오기
        agent = AzureSpeechAgent.get_instance()
        token, region = await agent.process()

        response_data = SpeechTokenResponse(
            token=token,
            region=region
        )

        resp = JSONResponse(
            content={
                "success": True,
                "message": "Speech token issued successfully",
                "data": response_data.model_dump()
            }
        )
        origin = request.headers.get("origin")
        if origin:
            resp.headers["Access-Control-Allow-Origin"] = origin
        else:
            resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "*"

        return resp

    except Exception as e:
        logger.error(f"토큰 발급 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"토큰 발급 실패: {str(e)}"
        )
    



@router.post(
    "/token/refresh",
    response_model=dict,
    summary="Azure Speech 토큰 강제 갱신",
    description="캐시된 토큰을 무효화하고 새 토큰을 발급합니다 (테스트용)"
)
async def refresh_speech_token(request: Request):
    """
    Azure Speech 토큰 강제 갱신 (캐시 무효화)

    Returns:
        dict: 새 토큰 및 리전 정보
    """
    try:
        # 싱글톤 Agent 인스턴스 가져오기
        agent = AzureSpeechAgent.get_instance()
        token, region = await agent.refresh_token()

        response_data = SpeechTokenResponse(
            token=token,
            region=region
        )

        resp = JSONResponse(
            content={
                "success": True,
                "message": "Speech token refreshed successfully",
                "data": response_data.model_dump()
            }
        )
        origin = request.headers.get("origin")
        if origin:
            resp.headers["Access-Control-Allow-Origin"] = origin
        else:
            resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "*"

        return resp

    except Exception as e:
        logger.error(f"토큰 갱신 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"토큰 갱신 실패: {str(e)}"
        )


@router.get(
    "/region",
    summary="Azure Speech 리전 조회",
    description="설정된 Azure Speech 리전을 반환합니다"
)
async def get_speech_region(request: Request):
    """
    Azure Speech 리전 조회

    Returns:
        dict: 리전 정보
    """
    # 싱글톤 Agent 인스턴스 가져오기
    agent = AzureSpeechAgent.get_instance()

    resp = JSONResponse(
        content={
            "success": True,
            "message": "Speech region retrieved",
            "data": {
                "region": agent.get_region()
            }
        }
    )
    origin = request.headers.get("origin")
    if origin:
        resp.headers["Access-Control-Allow-Origin"] = origin
    else:
        resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "*"

    return resp
