"""
회화 연습 API 엔드포인트
"""
from fastapi import APIRouter, Depends, HTTPException
import logging

from app.auth import get_current_user
from app.services.conversation_service import ConversationService
from app.schemas.conversation import (
    StartConversationRequest,
    SendMessageRequest,
    EndConversationRequest,
    MessageFeedbackRequest,
    TranslateMessageRequest,
    HintRequest
)

router = APIRouter(prefix="/api/ai/conversations", tags=["Conversations"])
logger = logging.getLogger(__name__)

conversation_service = ConversationService()


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
            user_id=user_id,
            current_step_index=request.currentStepIndex
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
        request: 메시지 피드백 요청 (현재 단계 정보 포함 가능)
        user: 현재 사용자 정보

    Returns:
        문법 교정, 용어 사용, 제안, 점수, 단계별 표현 피드백
    """
    try:
        user_id = user["user_id"]

        # audioData를 올바르게 전달 (None이 아닌 빈 문자열 체크)
        audio_data_to_send = request.audioData if request.audioData and len(request.audioData) > 0 else None

        # currentStep을 dict로 변환 (있는 경우)
        current_step = None
        if request.currentStep:
            current_step = {
                "name": request.currentStep.name,
                "title": request.currentStep.title,
                "guide": request.currentStep.guide,
                "terminology": request.currentStep.terminology
            }

        feedback = await conversation_service.generate_message_feedback(
            scenario_id=request.scenarioId,
            user_message=request.message,
            detected_terms=request.detectedTerms,
            user_id=user_id,
            audio_data=audio_data_to_send,
            current_step=current_step
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


@router.post("/translate")
async def translate_message(
    request: TranslateMessageRequest,
    user: dict = Depends(get_current_user)
):
    """
    메시지 번역 (컨텍스트 기반 GPT-4o 사용)

    시나리오의 맥락과 전문용어를 활용하여 일관성 있는 번역을 제공합니다.

    Args:
        request: 번역 요청 (시나리오 ID, 메시지, 원본/목표 언어)
        user: 현재 사용자 정보

    Returns:
        번역된 텍스트
    """
    try:
        user_id = user["user_id"]

        translated_text = await conversation_service.translate_message(
            scenario_id=request.scenarioId,
            message=request.message,
            source_language=request.sourceLanguage,
            target_language=request.targetLanguage,
            user_id=user_id
        )

        return {
            "success": True,
            "translatedText": translated_text
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to translate message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to translate message: {str(e)}")


@router.post("/reset")
async def reset_conversation(
    request: StartConversationRequest,
    user: dict = Depends(get_current_user)
):
    """
    대화 초기화 - 해당 시나리오의 모든 세션 및 메시지 삭제

    Args:
        request: 시나리오 ID
        user: 현재 사용자 정보

    Returns:
        초기화 성공 메시지
    """
    try:
        user_id = user["user_id"]

        await conversation_service.reset_conversation(
            scenario_id=request.scenarioId,
            user_id=user_id
        )

        return {
            "success": True,
            "message": "Conversation reset successfully"
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to reset conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reset conversation: {str(e)}")


@router.get("/history/{scenario_id}")
async def get_conversation_history(
    scenario_id: str,
    user: dict = Depends(get_current_user)
):
    """
    저장된 대화 히스토리 조회

    Args:
        scenario_id: 시나리오 ID
        user: 현재 사용자 정보

    Returns:
        세션 정보 및 메시지 목록
    """
    try:
        user_id = user["user_id"]

        history = await conversation_service.get_conversation_history(
            scenario_id=scenario_id,
            user_id=user_id
        )

        return {
            "success": True,
            **history
        }

    except Exception as e:
        logger.error(f"Failed to get conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation history: {str(e)}")


@router.post("/hint")
async def generate_hint(
    request: HintRequest,
    user: dict = Depends(get_current_user)
):
    """
    2단계 대화 힌트 생성

    시나리오 맥락과 현재 step의 terminology를 기반으로
    단어 → 문장 순서의 2단계 힌트를 생성합니다.

    Args:
        request: 힌트 요청 (시나리오 ID, 히스토리, 마지막 AI 메시지, 현재 단계)
        user: 현재 사용자 정보

    Returns:
        2단계 힌트 구조:
        - targetExpression: 목표 표현 (원어민이 쓸 문장)
        - wordHints: 핵심 단어 리스트 (Level 0: 단어 힌트)
        - fullSentence: 완전한 문장 (Level 1: 전체 문장)
        - explanation: 한국어 설명
        - stepInfo: 현재 단계 정보
    """
    try:
        user_id = user["user_id"]

        # currentStep을 dict로 변환 (있는 경우)
        current_step = None
        if request.currentStep:
            current_step = {
                "name": request.currentStep.name,
                "title": request.currentStep.title,
                "guide": request.currentStep.guide,
                "terminology": request.currentStep.terminology
            }

        result = await conversation_service.generate_hint(
            scenario_id=request.scenarioId,
            conversation_history=request.history,
            last_ai_message=request.lastAiMessage,
            user_id=user_id,
            current_step=current_step
        )

        return {
            "success": True,
            **result
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate hint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate hint: {str(e)}")
