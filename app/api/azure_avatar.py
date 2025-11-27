"""
Azure Avatar API 엔드포인트

브라우저에서 Azure Speech Avatar SDK를 사용하기 위한 토큰 발급 API
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import logging
from agent.avatar.azure_avatar_agent import AzureAvatarAgent
from app.schemas.azure_avatar import AvatarTokenResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/avatar", tags=["Azure Avatar"])


@router.options("/token")
async def options_avatar_token():
    """브라우저 preflight용 OPTIONS 허용"""
    return JSONResponse(content=None, status_code=200)


@router.get(
    "/token",
    response_model=dict,
    summary="Azure Avatar 인증 토큰 발급",
    description="""
    브라우저에서 Azure Speech Avatar SDK를 사용하기 위한 인증 토큰을 발급합니다.

    - 토큰 유효 시간: 10분
    - 클라이언트는 토큰을 캐싱하여 재사용 권장
    - Southeast Asia 리전 사용 (Avatar 지원 리전)

    **보안**: Azure 구독 키 노출 방지를 위해 백엔드에서 토큰 발급
    """
)
async def get_avatar_token(request: Request):
    """
    Azure Avatar 인증 토큰 발급

    Returns:
        dict: 토큰 및 리전 정보

    Raises:
        HTTPException: 토큰 발급 실패 시
    """
    try:
        # 싱글톤 Agent 인스턴스 가져오기
        agent = AzureAvatarAgent.get_instance()
        token, region = await agent.process()

        response_data = AvatarTokenResponse(
            token=token,
            region=region
        )

        resp = JSONResponse(
            content={
                "success": True,
                "message": "Avatar token issued successfully",
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
        logger.error(f"Avatar 토큰 발급 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Avatar 토큰 발급 실패: {str(e)}"
        )


@router.post(
    "/token/refresh",
    response_model=dict,
    summary="Azure Avatar 토큰 강제 갱신",
    description="캐시된 토큰을 무효화하고 새 토큰을 발급합니다 (테스트용)"
)
async def refresh_avatar_token(request: Request):
    """
    Azure Avatar 토큰 강제 갱신 (캐시 무효화)

    Returns:
        dict: 새 토큰 및 리전 정보
    """
    try:
        # 싱글톤 Agent 인스턴스 가져오기
        agent = AzureAvatarAgent.get_instance()
        token, region = await agent.refresh_token()

        response_data = AvatarTokenResponse(
            token=token,
            region=region
        )

        resp = JSONResponse(
            content={
                "success": True,
                "message": "Avatar token refreshed successfully",
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
        logger.error(f"Avatar 토큰 갱신 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Avatar 토큰 갱신 실패: {str(e)}"
        )


@router.get(
    "/region",
    summary="Azure Avatar 리전 조회",
    description="설정된 Azure Avatar 리전을 반환합니다"
)
async def get_avatar_region(request: Request):
    """
    Azure Avatar 리전 조회

    Returns:
        dict: 리전 정보
    """
    # 싱글톤 Agent 인스턴스 가져오기
    agent = AzureAvatarAgent.get_instance()

    resp = JSONResponse(
        content={
            "success": True,
            "message": "Avatar region retrieved",
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


@router.get(
    "/ice-servers",
    summary="Azure Avatar ICE 서버 정보 조회",
    description="""
    WebRTC 연결을 위한 TURN 서버 정보를 Azure에서 가져옵니다.

    - RTCPeerConnection에 필요한 ICE 서버 설정 정보 제공
    - TURN URL, Username, Credential 포함
    """
)
async def get_ice_servers(request: Request):
    """
    Azure Avatar ICE 서버 정보 조회 (TURN 서버)

    Returns:
        dict: ICE 서버 정보 (urls, username, credential)

    Raises:
        HTTPException: ICE 서버 정보 조회 실패 시
    """
    try:
        # 싱글톤 Agent 인스턴스 가져오기
        agent = AzureAvatarAgent.get_instance()
        ice_servers = await agent.get_ice_servers()

        resp = JSONResponse(
            content={
                "success": True,
                "message": "ICE servers retrieved successfully",
                "data": ice_servers
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
        logger.error(f"ICE 서버 정보 조회 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"ICE 서버 정보 조회 실패: {str(e)}"
        )


@router.post(
    "/synthesize",
    summary="텍스트 음성 변환 (TTS)",
    description="""
    텍스트를 고품질 음성으로 변환합니다.

    주의: Avatar 비디오 기능은 현재 Azure 구독에서 지원되지 않습니다.
    대신 Neural TTS 음성만 제공됩니다.

    - MP3 형식으로 반환
    - 응답 시간: 1-3초 (텍스트 길이에 따라)
    """
)
async def synthesize_avatar(request: Request):
    """
    텍스트를 음성으로 변환

    Request Body:
        {
            "text": "말할 텍스트",
            "language": "en-US"  // optional
        }

    Returns:
        오디오 파일 (MP3)

    Raises:
        HTTPException: 음성 합성 실패 시
    """
    try:
        body = await request.json()
        text = body.get("text")
        language = body.get("language", "en-US")

        if not text:
            raise HTTPException(status_code=400, detail="텍스트가 필요합니다")

        # 싱글톤 Agent 인스턴스 가져오기
        agent = AzureAvatarAgent.get_instance()
        audio_data = await agent.synthesize_avatar_video(text, language)

        # 오디오 파일로 반환
        from fastapi.responses import Response
        return Response(
            content=audio_data,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=speech.mp3",
                "Access-Control-Allow-Origin": "*"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Avatar 비디오 합성 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Avatar 비디오 합성 실패: {str(e)}"
        )
