"""
Azure Avatar API 엔드포인트

브라우저에서 Azure Speech Avatar SDK를 사용하기 위한 토큰 발급 API
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import logging
import uuid
from agent.avatar.azure_avatar_agent import AzureAvatarAgent
from app.schemas.azure_avatar import (
    AvatarTokenResponse,
    AvatarApplyRequest,
    AvatarApplyResponse,
    VisemeData
)

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


@router.post(
    "/apply",
    response_model=dict,
    summary="TTS 결과를 아바타에 적용",
    description="""
    기존 TTS 결과(텍스트, 오디오URL)를 받아 아바타 시각화에 필요한 정보를 반환합니다.

    - 기존 TTS 파이프라인은 그대로 유지
    - Avatar 모듈은 독립적으로 TTS 결과를 구독하여 동작
    - WebRTC 모드: SDP 교환 정보 제공
    - Viseme 모드: 입모양 타임라인 제공

    **주의**: 기존 STT→LLM→TTS 흐름은 변경하지 않음
    """
)
async def apply_avatar(request: AvatarApplyRequest):
    """
    TTS 결과를 아바타에 적용

    Args:
        request: AvatarApplyRequest (text, audio_url, language, avatar_style)

    Returns:
        dict: 아바타 세션 정보 및 viseme 데이터
    """
    try:
        logger.info(f"Avatar apply request: text={request.text[:50]}..., lang={request.language}")

        # 세션 ID 생성
        session_id = str(uuid.uuid4())

        # Viseme 데이터 생성 (간단한 시뮬레이션)
        # 실제 구현에서는 Azure Speech SDK의 viseme 이벤트를 사용
        visemes = _generate_viseme_timeline(request.text, request.language)

        # 응답 생성
        response_data = AvatarApplyResponse(
            session_id=session_id,
            avatar_type="viseme",  # 현재는 viseme 모드만 지원
            audio_url=request.audio_url,
            visemes=visemes,
            duration_ms=len(request.text) * 80  # 대략적인 음성 길이 추정
        )

        logger.info(f"Avatar session created: {session_id}, visemes={len(visemes)}")

        return {
            "success": True,
            "message": "Avatar apply 성공",
            "data": response_data.model_dump()
        }

    except Exception as e:
        logger.error(f"Avatar apply 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Avatar apply 실패: {str(e)}"
        )


@router.post(
    "/session",
    response_model=dict,
    summary="Azure Avatar WebRTC 세션 생성",
    description="""
    Azure Avatar 실시간 WebRTC 연결을 위한 세션을 생성합니다.

    - 토큰, ICE 서버, WebSocket URL 등 연결에 필요한 모든 정보 제공
    - 프론트엔드에서 RTCPeerConnection 설정에 사용
    """
)
async def create_avatar_session(request: Request):
    """
    Azure Avatar WebRTC 세션 생성

    Request Body (optional):
        {
            "character": "lisa",  // 아바타 캐릭터
            "style": "casual-sitting"  // 아바타 스타일
        }

    Returns:
        dict: 세션 정보 (session_id, ws_url, token, ice_servers 등)
    """
    try:
        body = {}
        try:
            body = await request.json()
        except:
            pass

        character = body.get("character", "lisa")
        style = body.get("style", "casual-sitting")

        # Agent 인스턴스에서 세션 생성
        agent = AzureAvatarAgent.get_instance()
        session_data = await agent.create_avatar_session(character, style)

        return {
            "success": True,
            "message": "Avatar session created",
            "data": session_data
        }

    except Exception as e:
        logger.error(f"Avatar session 생성 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Avatar session 생성 실패: {str(e)}"
        )


@router.get(
    "/config",
    response_model=dict,
    summary="Azure Avatar 연결 설정 조회",
    description="""
    Azure Avatar WebRTC 연결에 필요한 모든 설정 정보를 반환합니다.

    - Speech token
    - ICE servers (TURN)
    - WebSocket endpoint
    - 기본 아바타 캐릭터/스타일
    """
)
async def get_avatar_config(request: Request):
    """
    Azure Avatar 연결 설정 조회

    Returns:
        dict: 연결 설정 정보
    """
    try:
        agent = AzureAvatarAgent.get_instance()
        config = await agent.get_avatar_config()

        return {
            "success": True,
            "message": "Avatar config retrieved",
            "data": config
        }

    except Exception as e:
        logger.error(f"Avatar config 조회 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Avatar config 조회 실패: {str(e)}"
        )


def _generate_viseme_timeline(text: str, language: str) -> list:
    """
    텍스트에서 viseme 타임라인 생성 (간단한 시뮬레이션)

    실제 구현에서는 Azure Speech SDK의 viseme 콜백을 사용해야 함.
    여기서는 데모용으로 기본 패턴 생성.

    Viseme ID (0-21):
    - 0: 입 다물기 (silence)
    - 1-6: 모음 (a, e, i, o, u 등)
    - 7-21: 자음들

    Args:
        text: 변환할 텍스트
        language: 언어 코드

    Returns:
        list: VisemeData 리스트
    """
    visemes = []
    offset = 0
    avg_phoneme_duration = 80  # ms

    # 간단한 문자 → viseme 매핑
    char_to_viseme = {
        'a': 1, 'e': 2, 'i': 3, 'o': 4, 'u': 5,
        'ㅏ': 1, 'ㅓ': 2, 'ㅣ': 3, 'ㅗ': 4, 'ㅜ': 5,
        ' ': 0, '.': 0, ',': 0, '!': 0, '?': 0
    }

    for char in text.lower():
        viseme_id = char_to_viseme.get(char, 10)  # 기본값: 자음
        visemes.append(VisemeData(offset_ms=offset, viseme_id=viseme_id))
        offset += avg_phoneme_duration

    # 마지막에 입 다물기
    visemes.append(VisemeData(offset_ms=offset, viseme_id=0))

    return visemes
