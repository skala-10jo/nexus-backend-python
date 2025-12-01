"""
Voice Translation API 엔드포인트

Azure Translator를 사용한 텍스트 번역 API
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import logging
from agent.stt_translation.translation_agent import TranslationAgent
from app.schemas.voice import (
    TranslationRequest,
    TranslationResponse,
    BatchTranslationRequest,
    BatchTranslationResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/translate", tags=["Voice Translation"])


@router.post(
    "",
    response_model=dict,
    summary="텍스트 번역",
    description="""
    Azure Translator를 사용하여 텍스트를 번역합니다.

    - ISO 639-1 언어 코드 사용 (예: ko, en, ja, zh-Hans)
    - 90개 이상의 언어 지원
    - 실시간 번역
    """
)
async def translate_text(request: TranslationRequest):
    """
    텍스트 번역

    Args:
        request: 번역 요청 (text, source_lang, target_lang)

    Returns:
        dict: 번역 결과
    """
    try:
        logger.info(f"Translation request: {request.source_lang} -> {request.target_lang}, length={len(request.text)}")

        # Translation Agent 실행
        agent = TranslationAgent.get_instance()
        translated_text = await agent.process(
            text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang
        )

        # 응답 구성
        response = TranslationResponse(
            original_text=request.text,
            translated_text=translated_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang
        )

        return JSONResponse(
            content={
                "success": True,
                "message": "번역 완료",
                "data": response.model_dump()
            }
        )

    except Exception as e:
        logger.error(f"Translation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"번역 실패: {str(e)}")


@router.post(
    "/batch",
    response_model=dict,
    summary="일괄 번역",
    description="""
    여러 텍스트를 한 번에 번역합니다.

    - 최대 100개의 텍스트 동시 번역
    - ISO 639-1 언어 코드 사용
    - 효율적인 배치 처리
    """
)
async def translate_batch(request: BatchTranslationRequest):
    """
    일괄 번역

    Args:
        request: 일괄 번역 요청 (texts, source_lang, target_lang)

    Returns:
        dict: 번역 결과 리스트
    """
    try:
        logger.info(f"Batch translation request: {request.source_lang} -> {request.target_lang}, count={len(request.texts)}")

        # Translation Agent 실행 (배치)
        agent = TranslationAgent.get_instance()
        translated_texts = await agent.process_batch(
            texts=request.texts,
            source_lang=request.source_lang,
            target_lang=request.target_lang
        )

        # 응답 구성
        translations = []
        for original, translated in zip(request.texts, translated_texts):
            translations.append(
                TranslationResponse(
                    original_text=original,
                    translated_text=translated,
                    source_lang=request.source_lang,
                    target_lang=request.target_lang
                )
            )

        response = BatchTranslationResponse(
            translations=translations,
            total_count=len(translations)
        )

        return JSONResponse(
            content={
                "success": True,
                "message": f"{len(translations)}개 번역 완료",
                "data": response.model_dump()
            }
        )

    except Exception as e:
        logger.error(f"Batch translation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"일괄 번역 실패: {str(e)}")


@router.get(
    "/languages",
    summary="지원 언어 목록 조회",
    description="Azure Translator가 지원하는 언어 목록을 반환합니다"
)
async def get_supported_languages():
    """
    지원 언어 목록 조회

    Returns:
        dict: 지원 언어 목록 (ISO 639-1 코드)
    """
    # Azure Translator 주요 지원 언어
    supported_languages = {
        "ko": "한국어",
        "en": "English",
        "ja": "日本語",
        "zh-Hans": "简体中文",
        "zh-Hant": "繁體中文",
        "es": "Español",
        "fr": "Français",
        "de": "Deutsch",
        "it": "Italiano",
        "pt": "Português",
        "ru": "Русский",
        "ar": "العربية",
        "hi": "हिन्दी",
        "th": "ไทย",
        "vi": "Tiếng Việt",
        "id": "Bahasa Indonesia",
        "ms": "Bahasa Melayu",
        "tr": "Türkçe",
        "pl": "Polski",
        "nl": "Nederlands"
    }

    return JSONResponse(
        content={
            "success": True,
            "message": "지원 언어 목록 조회 완료",
            "data": {
                "languages": supported_languages,
                "total_count": len(supported_languages)
            }
        }
    )
