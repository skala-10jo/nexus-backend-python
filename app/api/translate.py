"""
번역 API 엔드포인트

텍스트 번역 RESTful API
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.translate import TranslateRequest, TranslateResponse, DetectedTermResponse
from app.services.translation_service import TranslationService
from agent.stt_translation.translation_agent import TranslationAgent
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai/translate", tags=["Translation AI"])


# 다중 타겟 번역 스키마
class MultiTranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_langs: list[str]


class MultiTranslationResult(BaseModel):
    lang: str
    text: str


class MultiTranslateResponse(BaseModel):
    translations: list[MultiTranslationResult]


@router.post("", response_model=TranslateResponse, status_code=status.HTTP_200_OK)
async def translate_text(
    request: TranslateRequest,
    db: Session = Depends(get_db)
):
    """
    텍스트 번역 API

    프로젝트 선택 여부에 따라 다른 번역 전략을 사용합니다:
    - 프로젝트 없음: 기본 번역 (SimpleTranslationAgent)
    - 프로젝트 있음: 컨텍스트 기반 번역 (ContextEnhancedTranslationAgent + 용어집)

    Args:
        request: 번역 요청
            - text: 번역할 텍스트
            - source_lang: 원본 언어 (ko, en, ja, vi)
            - target_lang: 목표 언어
            - user_id: 사용자 ID
            - project_id: 프로젝트 ID (선택)
        db: DB 세션

    Returns:
        TranslateResponse: 번역 결과
            - translation_id: 번역 ID
            - original_text: 원문
            - translated_text: 번역문
            - source_language: 원본 언어
            - target_language: 목표 언어
            - context_used: 컨텍스트 사용 여부
            - context_summary: 사용된 컨텍스트 요약
            - detected_terms: 탐지된 전문용어 리스트
            - terms_count: 탐지된 용어 개수

    Raises:
        HTTPException 400: 잘못된 요청 (동일한 언어, 빈 텍스트 등)
        HTTPException 500: 번역 처리 중 오류
    """
    try:
        logger.info(f"📥 번역 요청 수신: user={request.user_id}, project={request.project_id}")

        service = TranslationService()

        result = await service.translate_text(
            text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            user_id=request.user_id,
            project_id=request.project_id,
            db=db
        )

        # DetectedTermResponse로 변환
        detected_terms = [
            DetectedTermResponse(
                matched_text=term["matched_text"],
                position_start=term["position_start"],
                position_end=term["position_end"],
                glossary_term_id=term.get("glossary_term_id"),
                korean_term=term["korean_term"],
                english_term=term.get("english_term"),
                vietnamese_term=term.get("vietnamese_term"),
                definition=term.get("definition"),
                domain=term.get("domain")
            )
            for term in result["detected_terms"]
        ]

        response = TranslateResponse(
            translation_id=result["translation_id"],
            original_text=result["original_text"],
            translated_text=result["translated_text"],
            source_language=result["source_language"],
            target_language=result["target_language"],
            context_used=result["context_used"],
            context_summary=result.get("context_summary"),
            detected_terms=detected_terms,
            terms_count=result["terms_count"]
        )

        logger.info(f"✅ 번역 완료: translation_id={response.translation_id}")

        return response

    except ValueError as e:
        logger.error(f"❌ 잘못된 요청: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"❌ 번역 처리 실패: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"번역 처리 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/multi", response_model=MultiTranslateResponse, status_code=status.HTTP_200_OK)
async def translate_multi_target(request: MultiTranslateRequest):
    """
    다중 타겟 번역 API (실시간 음성 번역용)

    하나의 텍스트를 여러 언어로 동시 번역합니다.
    Azure Translator API의 multi-target 기능을 사용하여
    한 번의 API 호출로 여러 언어로 번역합니다.

    Args:
        request: 다중 번역 요청
            - text: 번역할 텍스트
            - source_lang: 원본 언어 (ISO 639-1: ko, en, ja, vi 등)
            - target_langs: 목표 언어 리스트 (ISO 639-1)

    Returns:
        MultiTranslateResponse: 번역 결과 리스트
            - translations: [
                {"lang": "en", "text": "Hello"},
                {"lang": "ja", "text": "こんにちは"}
              ]

    Raises:
        HTTPException 400: 잘못된 요청 (빈 텍스트, 빈 타겟 언어 등)
        HTTPException 500: 번역 처리 중 오류

    Example:
        >>> POST /api/ai/translate/multi
        >>> {
        >>>   "text": "안녕하세요",
        >>>   "source_lang": "ko",
        >>>   "target_langs": ["en", "ja", "vi"]
        >>> }
        >>>
        >>> Response:
        >>> {
        >>>   "translations": [
        >>>     {"lang": "en", "text": "Hello"},
        >>>     {"lang": "ja", "text": "こんにちは"},
        >>>     {"lang": "vi", "text": "Xin chào"}
        >>>   ]
        >>> }
    """
    try:
        # 입력 검증
        if not request.text or not request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="번역할 텍스트가 비어있습니다"
            )

        if not request.target_langs or len(request.target_langs) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="목표 언어가 지정되지 않았습니다"
            )

        logger.info(
            f"📥 다중 번역 요청: {request.source_lang} -> {request.target_langs}, "
            f"text_len={len(request.text)}"
        )

        # TranslationAgent 사용 (싱글톤)
        agent = TranslationAgent.get_instance()

        # 다중 타겟 번역 수행
        translations = await agent.process_multi(
            text=request.text,
            source_lang=request.source_lang,
            target_langs=request.target_langs
        )

        # 응답 구성
        result = MultiTranslateResponse(
            translations=[
                MultiTranslationResult(lang=t["lang"], text=t["text"])
                for t in translations
            ]
        )

        logger.info(f"✅ 다중 번역 완료: {len(result.translations)}개 언어")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 다중 번역 실패: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"다중 번역 처리 중 오류가 발생했습니다: {str(e)}"
        )
