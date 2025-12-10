"""
Slack Agent API 엔드포인트

Slack 메시지 번역 및 초안 작성 기능
세션 기반 대화 컨텍스트 및 번역 기능 지원
"""

import logging
import uuid
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from agent.translate.simple_translation_agent import SimpleTranslationAgent
from agent.rag.bizguide_rag_agent import BizGuideRAGAgent
from agent.slack.draft_agent import SlackDraftAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/slack", tags=["Slack Agent"])


# ============== Session Store (In-Memory) ==============

class SessionStore:
    """
    세션별 대화 히스토리 저장소 (인메모리)

    Features:
        - 세션 ID 기반 대화 히스토리 관리
        - 자동 만료 (30분 비활성)
        - 마지막 생성된 초안 저장 (번역/수정 요청 시 사용)
    """

    def __init__(self, expiry_minutes: int = 30):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._expiry_minutes = expiry_minutes

    def create_session(self) -> str:
        """새 세션 생성 및 ID 반환"""
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "history": [],
            "last_draft": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 조회 (만료 체크 포함)"""
        self._cleanup_expired()
        session = self._sessions.get(session_id)
        if session:
            session["updated_at"] = datetime.now()
        return session

    def add_message(self, session_id: str, role: str, content: str):
        """대화 히스토리에 메시지 추가"""
        session = self.get_session(session_id)
        if session:
            session["history"].append({"role": role, "content": content})
            session["updated_at"] = datetime.now()

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """대화 히스토리 반환"""
        session = self.get_session(session_id)
        return session["history"] if session else []

    def set_last_draft(self, session_id: str, draft: str):
        """마지막 생성 초안 저장"""
        session = self.get_session(session_id)
        if session:
            session["last_draft"] = draft

    def get_last_draft(self, session_id: str) -> Optional[str]:
        """마지막 생성 초안 반환"""
        session = self.get_session(session_id)
        return session["last_draft"] if session else None

    def delete_session(self, session_id: str):
        """세션 삭제"""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def _cleanup_expired(self):
        """만료된 세션 정리"""
        now = datetime.now()
        expired = [
            sid for sid, data in self._sessions.items()
            if now - data["updated_at"] > timedelta(minutes=self._expiry_minutes)
        ]
        for sid in expired:
            del self._sessions[sid]


# 전역 세션 저장소
session_store = SessionStore()


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


class SlackChatMessage(BaseModel):
    """대화 메시지"""
    role: str = Field(..., description="역할 (user 또는 assistant)")
    content: str = Field(..., description="메시지 내용")


class SlackChatRequest(BaseModel):
    """Slack 챗봇 요청 (세션 기반)"""
    message: str = Field(..., description="사용자 메시지")
    session_id: Optional[str] = Field(default=None, description="세션 ID (없으면 새 세션 생성)")
    language: str = Field(default="ko", description="기본 언어 (ko, en)")


class SlackChatResponse(BaseModel):
    """Slack 챗봇 응답"""
    session_id: str = Field(..., description="세션 ID")
    message: str = Field(..., description="AI 응답 메시지")
    draft: Optional[str] = Field(default=None, description="생성된 초안 (있는 경우)")
    action_type: str = Field(..., description="수행된 작업 (draft, translate, refine, general)")
    suggestions: List[BizGuideSuggestion] = Field(default_factory=list, description="참고된 비즈니스 표현")


# ============== Intent Detection ==============

def detect_intent(message: str, has_draft: bool = False) -> Dict[str, Any]:
    """
    사용자 메시지에서 의도를 감지합니다.

    Args:
        message: 사용자 메시지
        has_draft: 이전에 생성된 초안이 있는지 여부

    Returns:
        {
            "intent": "draft" | "translate" | "refine" | "general",
            "target_language": "ko" | "en" | None,
            "refinement_instruction": str | None
        }
    """
    message_lower = message.lower().strip()

    # 번역 요청 감지 패턴
    translate_to_en_patterns = [
        r"영어로\s*(번역|바꿔|변환|작성)",
        r"english로",
        r"translate\s*(to|into)?\s*english",
        r"in english",
        r"영문으로",
        r"영어\s*버전",
    ]

    translate_to_ko_patterns = [
        r"한글로\s*(번역|바꿔|변환|작성)",
        r"한국어로",
        r"korean으로",
        r"translate\s*(to|into)?\s*korean",
        r"in korean",
        r"한국어\s*버전",
    ]

    # 수정/개선 요청 감지 패턴
    refine_patterns = [
        r"(좀 더|더)\s*(친절|공손|격식|간결|짧게|길게|자세히|상세히)",
        r"(수정|고쳐|바꿔|변경)",
        r"(톤|어조|분위기).*?(바꿔|변경|수정)",
        r"다시\s*(작성|써줘|만들어)",
        r"~(하게|하도록)\s*(바꿔|수정|고쳐)",
    ]

    # 번역 요청 확인
    for pattern in translate_to_en_patterns:
        if re.search(pattern, message_lower):
            return {
                "intent": "translate",
                "target_language": "en",
                "refinement_instruction": None
            }

    for pattern in translate_to_ko_patterns:
        if re.search(pattern, message_lower):
            return {
                "intent": "translate",
                "target_language": "ko",
                "refinement_instruction": None
            }

    # 수정 요청 확인 (이전 초안이 있는 경우에만)
    if has_draft:
        for pattern in refine_patterns:
            if re.search(pattern, message_lower):
                return {
                    "intent": "refine",
                    "target_language": None,
                    "refinement_instruction": message
                }

    # 기본: 새 초안 작성
    return {
        "intent": "draft",
        "target_language": None,
        "refinement_instruction": None
    }


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

        # 1. 세션 관리
        session_id = request.session_id
        if not session_id or not session_store.get_session(session_id):
            session_id = session_store.create_session()
            logger.info(f"🆕 새 세션 생성: {session_id}")
        else:
            logger.info(f"📎 기존 세션 사용: {session_id}")

        # 2. 이전 초안 존재 여부 확인
        last_draft = session_store.get_last_draft(session_id)
        has_draft = last_draft is not None and len(last_draft) > 0

        # 3. 의도 감지
        intent_result = detect_intent(request.message, has_draft)
        intent = intent_result["intent"]
        target_language = intent_result["target_language"] or request.language

        logger.info(f"🎯 의도 감지: intent={intent}, target_lang={target_language}, has_draft={has_draft}")

        # 4. 대화 히스토리 가져오기
        conversation_history = session_store.get_history(session_id)

        # 5. 의도별 처리
        draft = None
        suggestions = []
        response_message = ""

        if intent == "translate" and has_draft:
            # 이전 초안을 번역
            logger.info(f"🌐 초안 번역 요청: {target_language}")

            # 원본 언어 감지
            source_lang = "ko" if target_language == "en" else "en"

            translation_agent = get_translation_agent()
            translated_text = await translation_agent.process(
                text=last_draft,
                source_lang=source_lang,
                target_lang=target_language
            )

            draft = translated_text
            session_store.set_last_draft(session_id, draft)
            response_message = f"초안을 {'영어' if target_language == 'en' else '한국어'}로 번역했습니다."

        elif intent == "refine" and has_draft:
            # 이전 초안을 수정
            logger.info(f"✏️ 초안 수정 요청")

            # 대화 히스토리에 이전 초안 컨텍스트 추가
            refine_history = conversation_history.copy()
            refine_history.append({
                "role": "assistant",
                "content": f"이전에 작성한 초안:\n\n{last_draft}"
            })

            # RAG 검색
            rag_agent = get_rag_agent()
            rag_results = await rag_agent.process(
                query=request.message,
                top_k=3
            )
            rag_context = [r.get("text", "") for r in rag_results if r.get("text")]

            # 수정 요청으로 초안 재작성
            draft_agent = get_draft_agent()
            draft_result = await draft_agent.process(
                original_message=f"다음 초안을 수정해주세요: {request.message}\n\n기존 초안:\n{last_draft}",
                rag_context=rag_context,
                target_language=request.language,
                conversation_history=refine_history
            )

            draft = draft_result.get("draft", "")
            session_store.set_last_draft(session_id, draft)
            response_message = "초안을 수정했습니다."

            suggestions = [
                BizGuideSuggestion(
                    text=r.get("text", ""),
                    chapter=r.get("chapter"),
                    section=r.get("section"),
                    score=r.get("score", 0.0)
                )
                for r in rag_results
            ]

        else:
            # 새 초안 작성
            logger.info(f"📝 새 초안 작성 요청")

            # RAG 검색
            rag_agent = get_rag_agent()
            rag_results = await rag_agent.process(
                query=request.message,
                top_k=5
            )
            rag_context = [r.get("text", "") for r in rag_results if r.get("text")]

            # 초안 작성
            draft_agent = get_draft_agent()
            draft_result = await draft_agent.process(
                original_message=request.message,
                rag_context=rag_context,
                target_language=target_language,
                conversation_history=conversation_history
            )

            draft = draft_result.get("draft", "")
            session_store.set_last_draft(session_id, draft)
            response_message = "비즈니스 메시지 초안을 작성했습니다."

            suggestions = [
                BizGuideSuggestion(
                    text=r.get("text", ""),
                    chapter=r.get("chapter"),
                    section=r.get("section"),
                    score=r.get("score", 0.0)
                )
                for r in rag_results
            ]

        # 6. 대화 히스토리 업데이트
        session_store.add_message(session_id, "user", request.message)
        session_store.add_message(session_id, "assistant", draft or response_message)

        logger.info(f"✅ Slack 챗 응답 완료: intent={intent}")

        return SlackChatResponse(
            session_id=session_id,
            message=response_message,
            draft=draft,
            action_type=intent,
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
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="세션을 찾을 수 없습니다"
        )

    session_store.delete_session(session_id)
    logger.info(f"🗑️ 세션 삭제: {session_id}")

    return {"success": True, "message": "세션이 삭제되었습니다"}
