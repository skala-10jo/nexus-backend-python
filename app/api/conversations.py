"""
회화 연습 API 엔드포인트
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from uuid import UUID
import logging

from app.auth import get_current_user
from app.services.conversation_service import ConversationService

router = APIRouter(prefix="/api/ai/conversations", tags=["Conversations"])
logger = logging.getLogger(__name__)

conversation_service = ConversationService()


# Request/Response Models
class StartConversationRequest(BaseModel):
    scenarioId: str


class SendMessageRequest(BaseModel):
    scenarioId: str
    message: str
    history: List[Dict[str, str]] = []


class EndConversationRequest(BaseModel):
    scenarioId: str
    history: List[Dict[str, str]] = []


class MessageFeedbackRequest(BaseModel):
    scenarioId: str
    message: str
    detectedTerms: List[str] = []


@router.post("/start")
async def start_conversation(
    request: StartConversationRequest,
    user: dict = Depends(get_current_user)
):
    """
    대화 시작

    Args:
        request: 시나리오 ID
        user: 현재 사용자 정보

    Returns:
        시나리오 정보 및 초기 AI 메시지
    """
    try:
        user_id = user["user_id"]

        result = await conversation_service.start_conversation(
            scenario_id=request.scenarioId,
            user_id=user_id
        )

        return {
            "success": True,
            **result
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start conversation: {str(e)}")


@router.post("/message")
async def send_message(
    request: SendMessageRequest,
    user: dict = Depends(get_current_user)
):
    """
    사용자 메시지 전송 및 AI 응답 생성

    Args:
        request: 메시지 요청 (시나리오 ID, 메시지, 히스토리)
        user: 현재 사용자 정보

    Returns:
        AI 응답 및 감지된 전문용어
    """
    try:
        user_id = user["user_id"]

        result = await conversation_service.send_message(
            scenario_id=request.scenarioId,
            user_message=request.message,
            conversation_history=request.history,
            user_id=user_id
        )

        return {
            "success": True,
            **result
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to send message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")


@router.post("/feedback")
async def get_message_feedback(
    request: MessageFeedbackRequest,
    user: dict = Depends(get_current_user)
):
    """
    사용자 메시지에 대한 피드백 생성

    Args:
        request: 메시지 피드백 요청
        user: 현재 사용자 정보

    Returns:
        문법 교정, 용어 사용, 제안, 점수
    """
    try:
        user_id = user["user_id"]

        feedback = await conversation_service.generate_message_feedback(
            scenario_id=request.scenarioId,
            user_message=request.message,
            detected_terms=request.detectedTerms,
            user_id=user_id
        )

        return {
            "success": True,
            "feedback": feedback
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate feedback: {str(e)}")


@router.post("/end")
async def end_conversation(
    request: EndConversationRequest,
    user: dict = Depends(get_current_user)
):
    """
    대화 종료 (선택 사항)

    Args:
        request: 시나리오 ID 및 대화 히스토리
        user: 현재 사용자 정보

    Returns:
        종합 피드백 (추후 구현)
    """
    try:
        # 추후 종합 피드백 기능 구현 예정
        return {
            "success": True,
            "message": "Conversation ended successfully"
        }

    except Exception as e:
        logger.error(f"Failed to end conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to end conversation: {str(e)}")
