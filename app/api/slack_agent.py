"""
Slack Agent API 엔드포인트

Slack 메시지 번역 및 초안 작성 기능
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from agent.translate.simple_translation_agent import SimpleTranslationAgent
from agent.rag.bizguide_rag_agent import BizGuideRAGAgent
from agent.slack.draft_agent import SlackDraftAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/slack", tags=["Slack Agent"])


# ============== Schemas ==============

class SlackTranslateRequest(BaseModel):
    """Slack 메시지 번역 요청"""
    text: str = Field(..., description="번역할 텍스트")
    source_lang: str = Field(default="auto", description="원본 언어 (auto: 자동 감지)")
    target_lang: str = Field(..., description="목표 언어 (ko, en, ja, vi, zh)")


class SlackTranslateResponse(BaseModel):
    """Slack 메시지 번역 응답"""
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str


class SlackDraftRequest(BaseModel):
    """Slack 초안 작성 요청"""
    message: str = Field(..., description="작성하고 싶은 내용/의도")
    language: str = Field(default="ko", description="목표 언어 (ko, en)")
    keywords: Optional[List[str]] = Field(default=None, description="RAG 검색 키워드 (선택)")


class BizGuideSuggestion(BaseModel):
    """비즈니스 표현 제안"""
    text: str
    chapter: Optional[str] = None
    section: Optional[str] = None
    score: float


class SlackDraftResponse(BaseModel):
    """Slack 초안 작성 응답"""
    draft: str = Field(..., description="작성된 초안")
    suggestions: List[BizGuideSuggestion] = Field(default_factory=list, description="참고된 비즈니스 표현")
    status: str


# ============== Agents (Singleton) ==============

_translation_agent: Optional[SimpleTranslationAgent] = None
_rag_agent: Optional[BizGuideRAGAgent] = None
_draft_agent: Optional[SlackDraftAgent] = None


def get_translation_agent() -> SimpleTranslationAgent:
    global _translation_agent
    if _translation_agent is None:
        _translation_agent = SimpleTranslationAgent()
    return _translation_agent


def get_rag_agent() -> BizGuideRAGAgent:
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = BizGuideRAGAgent()
    return _rag_agent


def get_draft_agent() -> SlackDraftAgent:
    global _draft_agent
    if _draft_agent is None:
        _draft_agent = SlackDraftAgent()
    return _draft_agent


# ============== Endpoints ==============

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

        # 자동 언어 감지 시 기본값 사용 (간단히 처리)
        source_lang = request.source_lang
        if source_lang == "auto":
            # 간단한 휴리스틱: 한글이 있으면 ko, 일본어 문자가 있으면 ja, 그 외 en
            text = request.text
            if any('\uac00' <= c <= '\ud7a3' for c in text):
                source_lang = "ko"
            elif any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' for c in text):
                # 히라가나/카타카나가 있으면 ja, 그 외 중국어일 수 있음
                if any('\u3040' <= c <= '\u30ff' for c in text):
                    source_lang = "ja"
                else:
                    source_lang = "zh"
            else:
                source_lang = "en"

        if source_lang == request.target_lang:
            # 같은 언어면 그대로 반환
            return SlackTranslateResponse(
                original_text=request.text,
                translated_text=request.text,
                source_lang=source_lang,
                target_lang=request.target_lang
            )

        logger.info(f"🌐 Slack 번역 요청: {source_lang} → {request.target_lang}, len={len(request.text)}")

        agent = get_translation_agent()
        translated_text = await agent.process(
            text=request.text,
            source_lang=source_lang,
            target_lang=request.target_lang
        )

        logger.info(f"✅ Slack 번역 완료")

        return SlackTranslateResponse(
            original_text=request.text,
            translated_text=translated_text,
            source_lang=source_lang,
            target_lang=request.target_lang
        )

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

        logger.info(f"📝 Slack 초안 작성 요청: lang={request.language}, keywords={request.keywords}")

        # 1. BizGuide RAG로 관련 비즈니스 표현 검색
        rag_agent = get_rag_agent()
        rag_results = await rag_agent.process(
            query=request.message,
            keywords=request.keywords,
            top_k=5
        )

        # RAG 결과를 컨텍스트로 변환
        rag_context = [r.get("text", "") for r in rag_results if r.get("text")]

        suggestions = [
            BizGuideSuggestion(
                text=r.get("text", ""),
                chapter=r.get("chapter"),
                section=r.get("section"),
                score=r.get("score", 0.0)
            )
            for r in rag_results
        ]

        logger.info(f"🔍 RAG 검색 완료: {len(rag_results)}개 결과")

        # 2. SlackDraftAgent로 초안 작성 (EmailDraftAgent와 동일, recipient/subject만 없음)
        draft_agent = get_draft_agent()
        draft_result = await draft_agent.process(
            original_message=request.message,
            rag_context=rag_context if rag_context else None,
            target_language=request.language
        )

        logger.info(f"✅ Slack 초안 작성 완료")

        return SlackDraftResponse(
            draft=draft_result.get("draft", ""),
            suggestions=suggestions,
            status=draft_result.get("status", "success")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Slack 초안 작성 실패: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"초안 작성 중 오류가 발생했습니다: {str(e)}"
        )
