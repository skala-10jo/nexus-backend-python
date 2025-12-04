"""
Small Talk API 엔드포인트

스몰토크 대화 API.
DB 없이 메모리 기반으로 동작합니다.
"""
from fastapi import APIRouter, Depends, HTTPException
import logging

from app.auth import get_current_user
from app.services.small_talk_service import SmallTalkService
from app.schemas.small_talk import (
    StartSmallTalkResponse,
    SendSmallTalkMessageRequest,
    SendSmallTalkMessageResponse,
    SmallTalkFeedbackRequest,
    SmallTalkFeedbackResponse,
    SmallTalkHintRequest,
    SmallTalkHintResponse
)

router = APIRouter(prefix="/api/ai/small-talk", tags=["Small Talk"])
logger = logging.getLogger(__name__)

small_talk_service = SmallTalkService()


@router.post("/start", response_model=StartSmallTalkResponse)
async def start_small_talk(user: dict = Depends(get_current_user)):
    """
    스몰토크 대화 시작

    대시보드 진입 시 자동으로 호출됩니다.
    AI가 먼저 인사를 건넵니다.

    Returns:
        초기 AI 인사 메시지
    """
    try:
        result = await small_talk_service.start_conversation()

        return StartSmallTalkResponse(
            initial_message=result["initial_message"]
        )

    except Exception as e:
        logger.error(f"Failed to start small talk: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message", response_model=SendSmallTalkMessageResponse)
async def send_message(
    request: SendSmallTalkMessageRequest,
    user: dict = Depends(get_current_user)
):
    """
    메시지 전송 및 AI 응답 생성

    Args:
        request: 메시지, 대화 히스토리

    Returns:
        AI 응답 메시지
    """
    try:
        result = await small_talk_service.send_message(
            user_message=request.message,
            conversation_history=request.history
        )

        return SendSmallTalkMessageResponse(
            ai_message=result["ai_message"]
        )

    except Exception as e:
        logger.error(f"Failed to send message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback", response_model=SmallTalkFeedbackResponse)
async def get_feedback(
    request: SmallTalkFeedbackRequest,
    user: dict = Depends(get_current_user)
):
    """
    사용자 메시지에 대한 피드백 생성

    Args:
        request: 메시지, 히스토리, 오디오 데이터(선택)

    Returns:
        문법 교정, 제안, 점수 등
    """
    try:
        feedback = await small_talk_service.generate_feedback(
            user_message=request.message,
            conversation_history=request.history,
            audio_data=request.audio_data
        )

        return SmallTalkFeedbackResponse(**feedback)

    except Exception as e:
        logger.error(f"Failed to generate feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hint", response_model=SmallTalkHintResponse)
async def get_hint(
    request: SmallTalkHintRequest,
    user: dict = Depends(get_current_user)
):
    """
    응답 힌트 생성

    Args:
        request: 히스토리, 마지막 AI 메시지

    Returns:
        힌트 목록과 설명
    """
    try:
        result = await small_talk_service.generate_hint(
            conversation_history=request.history,
            last_ai_message=request.last_ai_message,
            hint_count=request.hint_count
        )

        return SmallTalkHintResponse(
            hints=result.get("hints", []),
            explanations=result.get("explanations", [])
        )

    except Exception as e:
        logger.error(f"Failed to generate hint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
