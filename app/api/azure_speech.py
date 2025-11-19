"""
Azure Speech Service API endpoints.
Provides token issuance for client-side Speech SDK usage.
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import logging
from app.services.azure_speech_service import azure_speech_service
from app.schemas.azure_speech import SpeechTokenResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/speech", tags=["Azure Speech"])


# ✅ 브라우저 preflight용 OPTIONS 허용
@router.options("/token")
async def options_speech_token():
    # CORSMiddleware가 실제로 처리하더라도,
    # 여기서 200만 돌려줘도 preflight는 통과한다.
    return JSONResponse(content=None, status_code=200)


@router.get(
    "/token",
    response_model=dict,
    summary="Get Azure Speech authorization token",
    description="""
    Issues an Azure Speech Service authorization token for client-side SDK usage.

    - Token is valid for 10 minutes
    - Client should cache and reuse the token until expiry
    - Used for Speech-to-Text, Translation, and Text-to-Speech operations

    **Security**: Token is issued from backend to prevent exposing Azure subscription key to clients.
    """
)
def get_speech_token(request: Request):
    """
    Get Azure Speech authorization token.

    Returns:
        dict: Success response with token and region

    Raises:
        HTTPException: If token request fails
    """
    try:
        token = azure_speech_service.get_token()
        region = azure_speech_service.get_region()

        response_data = SpeechTokenResponse(
            token=token,
            region=region
        )

        # ✅ 여기서도 안전하게 CORS 헤더 한 번 더 넣어줌
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
        logger.error(f"Failed to issue speech token: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to issue speech token: {str(e)}"
        )
    



@router.post(
    "/token/refresh",
    response_model=dict,
    summary="Force refresh Azure Speech token",
    description="Clears cached token and issues a new one (useful for testing)"
)
def refresh_speech_token(request: Request):
    """
    Force refresh Azure Speech token by clearing cache.

    Returns:
        dict: Success response with new token and region
    """
    try:
        # 캐시 초기화 후 새 토큰 발급
        azure_speech_service.clear_cache()
        token = azure_speech_service.get_token()
        region = azure_speech_service.get_region()

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
        logger.error(f"Failed to refresh speech token: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh speech token: {str(e)}"
        )


@router.get(
    "/region",
    summary="Get Azure Speech region",
    description="Returns the configured Azure Speech region"
)
def get_speech_region(request: Request):
    """
    Get Azure Speech region.

    Returns:
        dict: Response with region info
    """
    resp = JSONResponse(
        content={
            "success": True,
            "message": "Speech region retrieved",
            "data": {
                "region": azure_speech_service.get_region()
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
