"""
Slack Agent API 엔드포인트

Slack 메시지 번역 및 초안 작성 API
비즈니스 로직은 SlackAgentService에서 처리
"""

import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.slack_agent import (
    SlackTranslateRequest,
    SlackTranslateResponse,
    SlackDraftRequest,
    SlackDraftResponse,
    SlackChatRequest,
    SlackChatResponse,
    BizGuideSuggestion,
)
from app.services.slack_agent_service import SlackAgentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/slack", tags=["Slack Agent"])

# Service 인스턴스 (싱글톤)
slack_service = SlackAgentService()


@router.post("/translate", response_model=SlackTranslateResponse, status_code=status.HTTP_200_OK)
async def translate_slack_message(request: SlackTranslateRequest):
    """
    Slack 메시지 번역 API

    사용자가 받은 메시지를 원하는 언어로 번역합니다.

    Args:
        request: 번역 요청
            - text: 번역할 텍스트
            - source_lang: 원본 언어 (auto, ko, en, ja, vi, zh)
            - target_lang: 목표 언어

    Returns:
        SlackTranslateResponse: 번역 결과
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="번역할 텍스트가 비어있습니다"
            )

        result = await slack_service.translate(
            text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang
        )

        return SlackTranslateResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Slack 번역 실패: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"번역 처리 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/draft", response_model=SlackDraftResponse, status_code=status.HTTP_200_OK)
async def create_slack_draft(request: SlackDraftRequest):
    """
    Slack 메시지 초안 작성 API

    BizGuide RAG를 활용하여 비즈니스 메시지 초안을 작성합니다.

    Args:
        request: 초안 작성 요청
            - message: 작성하고 싶은 내용/의도
            - language: 목표 언어 (ko, en)
            - keywords: RAG 검색 키워드 (선택)

    Returns:
        SlackDraftResponse: 초안 작성 결과
    """
    try:
        if not request.message or not request.message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="작성할 내용이 비어있습니다"
            )

        result = await slack_service.create_draft(
            message=request.message,
            language=request.language,
            keywords=request.keywords
        )

        # suggestions를 BizGuideSuggestion 모델로 변환
        suggestions = [
            BizGuideSuggestion(**s) for s in result.get("suggestions", [])
        ]

        return SlackDraftResponse(
            draft=result["draft"],
            suggestions=suggestions,
            status=result["status"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Slack 초안 작성 실패: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"초안 작성 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/chat", response_model=SlackChatResponse, status_code=status.HTTP_200_OK)
async def slack_chat(request: SlackChatRequest):
    """
    Slack 챗봇 API (세션 기반 대화)

    세션 내에서 연속적인 대화를 지원합니다:
    - 초안 작성 → 번역 요청 → 수정 요청 등 연속 대화 가능
    - 세션 ID로 대화 컨텍스트 유지
    - "영어로 번역해줘" 등의 요청 자동 감지

    Args:
        request: 챗봇 요청
            - message: 사용자 메시지
            - session_id: 세션 ID (없으면 새 세션 생성)
            - language: 기본 언어 (ko, en)

    Returns:
        SlackChatResponse: 챗봇 응답
    """
    try:
        if not request.message or not request.message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="메시지가 비어있습니다"
            )

        result = await slack_service.chat(
            message=request.message,
            session_id=request.session_id,
            language=request.language
        )

        # suggestions를 BizGuideSuggestion 모델로 변환
        suggestions = [
            BizGuideSuggestion(**s) for s in result.get("suggestions", [])
        ]

        return SlackChatResponse(
            session_id=result["session_id"],
            message=result["message"],
            draft=result.get("draft"),
            action_type=result["action_type"],
            suggestions=suggestions
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Slack 챗 실패: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"처리 중 오류가 발생했습니다: {str(e)}"
        )


@router.delete("/session/{session_id}", status_code=status.HTTP_200_OK)
async def delete_slack_session(session_id: str):
    """
    Slack 세션 삭제 API

    Args:
        session_id: 삭제할 세션 ID

    Returns:
        삭제 결과
    """
    session = slack_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="세션을 찾을 수 없습니다"
        )

    slack_service.delete_session(session_id)

    return {"success": True, "message": "세션이 삭제되었습니다"}
