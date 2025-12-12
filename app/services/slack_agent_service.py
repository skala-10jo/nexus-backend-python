"""
Slack Agent Service

Slack 메시지 번역 및 초안 작성 비즈니스 로직
여러 Agent를 조율하여 세션 기반 대화 처리
"""

import logging
import uuid
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from agent.translate.simple_translation_agent import SimpleTranslationAgent
from agent.rag.bizguide_rag_agent import BizGuideRAGAgent
from agent.slack.draft_agent import SlackDraftAgent

logger = logging.getLogger(__name__)


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


# 전역 세션 저장소 (싱글톤)
_session_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    """세션 저장소 싱글톤 반환"""
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store


class SlackAgentService:
    """
    Slack Agent 비즈니스 로직 서비스

    책임:
        - 여러 Agent 조율 (번역, RAG, 초안 작성)
        - 세션 기반 대화 관리
        - 사용자 의도 감지 및 분기 처리
    """

    def __init__(self):
        """Agent 인스턴스 초기화"""
        self._translation_agent: Optional[SimpleTranslationAgent] = None
        self._rag_agent: Optional[BizGuideRAGAgent] = None
        self._draft_agent: Optional[SlackDraftAgent] = None
        self._session_store = get_session_store()

    @property
    def translation_agent(self) -> SimpleTranslationAgent:
        """번역 Agent (lazy initialization)"""
        if self._translation_agent is None:
            self._translation_agent = SimpleTranslationAgent()
        return self._translation_agent

    @property
    def rag_agent(self) -> BizGuideRAGAgent:
        """RAG Agent (lazy initialization)"""
        if self._rag_agent is None:
            self._rag_agent = BizGuideRAGAgent()
        return self._rag_agent

    @property
    def draft_agent(self) -> SlackDraftAgent:
        """초안 작성 Agent (lazy initialization)"""
        if self._draft_agent is None:
            self._draft_agent = SlackDraftAgent()
        return self._draft_agent

    def detect_intent(self, message: str, has_draft: bool = False) -> Dict[str, Any]:
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

    def detect_source_language(self, text: str) -> str:
        """
        텍스트에서 언어를 자동 감지합니다.

        Args:
            text: 감지할 텍스트

        Returns:
            언어 코드 (ko, en, ja, zh)
        """
        if any('\uac00' <= c <= '\ud7a3' for c in text):
            return "ko"
        elif any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' for c in text):
            if any('\u3040' <= c <= '\u30ff' for c in text):
                return "ja"
            else:
                return "zh"
        else:
            return "en"

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Dict[str, str]:
        """
        텍스트 번역

        Args:
            text: 번역할 텍스트
            source_lang: 원본 언어 (auto면 자동 감지)
            target_lang: 목표 언어

        Returns:
            번역 결과 딕셔너리
        """
        # 자동 언어 감지
        actual_source_lang = source_lang
        if source_lang == "auto":
            actual_source_lang = self.detect_source_language(text)

        # 같은 언어면 그대로 반환
        if actual_source_lang == target_lang:
            return {
                "original_text": text,
                "translated_text": text,
                "source_lang": actual_source_lang,
                "target_lang": target_lang
            }

        logger.info(f"🌐 번역 요청: {actual_source_lang} → {target_lang}, len={len(text)}")

        translated_text = await self.translation_agent.process(
            text=text,
            source_lang=actual_source_lang,
            target_lang=target_lang
        )

        logger.info(f"✅ 번역 완료")

        return {
            "original_text": text,
            "translated_text": translated_text,
            "source_lang": actual_source_lang,
            "target_lang": target_lang
        }

    async def create_draft(
        self,
        message: str,
        language: str = "ko",
        keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        비즈니스 메시지 초안 작성

        Args:
            message: 작성하고 싶은 내용/의도
            language: 목표 언어
            keywords: RAG 검색 키워드 (선택)

        Returns:
            초안 작성 결과 딕셔너리
        """
        logger.info(f"📝 초안 작성 요청: lang={language}, keywords={keywords}")

        # 1. BizGuide RAG로 관련 비즈니스 표현 검색
        rag_results = await self.rag_agent.process(
            query=message,
            keywords=keywords,
            top_k=5
        )

        rag_context = [r.get("text", "") for r in rag_results if r.get("text")]

        suggestions = [
            {
                "text": r.get("text", ""),
                "chapter": r.get("chapter"),
                "section": r.get("section"),
                "score": r.get("score", 0.0)
            }
            for r in rag_results
        ]

        logger.info(f"🔍 RAG 검색 완료: {len(rag_results)}개 결과")

        # 2. SlackDraftAgent로 초안 작성
        draft_result = await self.draft_agent.process(
            original_message=message,
            rag_context=rag_context if rag_context else None,
            target_language=language
        )

        logger.info(f"✅ 초안 작성 완료")

        return {
            "draft": draft_result.get("draft", ""),
            "suggestions": suggestions,
            "status": draft_result.get("status", "success")
        }

    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        language: str = "ko"
    ) -> Dict[str, Any]:
        """
        세션 기반 대화 처리

        Args:
            message: 사용자 메시지
            session_id: 세션 ID (없으면 새 세션 생성)
            language: 기본 언어

        Returns:
            챗봇 응답 딕셔너리
        """
        # 1. 세션 관리
        if not session_id or not self._session_store.get_session(session_id):
            session_id = self._session_store.create_session()
            logger.info(f"🆕 새 세션 생성: {session_id}")
        else:
            logger.info(f"📎 기존 세션 사용: {session_id}")

        # 2. 이전 초안 존재 여부 확인
        last_draft = self._session_store.get_last_draft(session_id)
        has_draft = last_draft is not None and len(last_draft) > 0

        # 3. 의도 감지
        intent_result = self.detect_intent(message, has_draft)
        intent = intent_result["intent"]
        target_language = intent_result["target_language"] or language

        logger.info(f"🎯 의도 감지: intent={intent}, target_lang={target_language}, has_draft={has_draft}")

        # 4. 대화 히스토리 가져오기
        conversation_history = self._session_store.get_history(session_id)

        # 5. 의도별 처리
        draft = None
        suggestions = []
        response_message = ""

        if intent == "translate" and has_draft:
            # 이전 초안을 번역
            result = await self._handle_translate(last_draft, target_language)
            draft = result["translated_text"]
            response_message = result["response_message"]

        elif intent == "refine" and has_draft:
            # 이전 초안을 수정
            result = await self._handle_refine(
                message, last_draft, language, conversation_history
            )
            draft = result["draft"]
            suggestions = result["suggestions"]
            response_message = result["response_message"]

        else:
            # 새 초안 작성
            result = await self._handle_new_draft(
                message, target_language, conversation_history
            )
            draft = result["draft"]
            suggestions = result["suggestions"]
            response_message = result["response_message"]

        # 6. 세션 업데이트
        if draft:
            self._session_store.set_last_draft(session_id, draft)
        self._session_store.add_message(session_id, "user", message)
        self._session_store.add_message(session_id, "assistant", draft or response_message)

        logger.info(f"✅ 챗 응답 완료: intent={intent}")

        return {
            "session_id": session_id,
            "message": response_message,
            "draft": draft,
            "action_type": intent,
            "suggestions": suggestions
        }

    async def _handle_translate(
        self,
        last_draft: str,
        target_language: str
    ) -> Dict[str, Any]:
        """번역 처리"""
        logger.info(f"🌐 초안 번역 요청: {target_language}")

        source_lang = "ko" if target_language == "en" else "en"

        translated_text = await self.translation_agent.process(
            text=last_draft,
            source_lang=source_lang,
            target_lang=target_language
        )

        lang_name = "영어" if target_language == "en" else "한국어"
        return {
            "translated_text": translated_text,
            "response_message": f"초안을 {lang_name}로 번역했습니다."
        }

    async def _handle_refine(
        self,
        message: str,
        last_draft: str,
        language: str,
        conversation_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """수정 처리"""
        logger.info(f"✏️ 초안 수정 요청")

        refine_history = conversation_history.copy()
        refine_history.append({
            "role": "assistant",
            "content": f"이전에 작성한 초안:\n\n{last_draft}"
        })

        rag_results = await self.rag_agent.process(query=message, top_k=3)
        rag_context = [r.get("text", "") for r in rag_results if r.get("text")]

        draft_result = await self.draft_agent.process(
            original_message=f"다음 초안을 수정해주세요: {message}\n\n기존 초안:\n{last_draft}",
            rag_context=rag_context,
            target_language=language,
            conversation_history=refine_history
        )

        suggestions = [
            {
                "text": r.get("text", ""),
                "chapter": r.get("chapter"),
                "section": r.get("section"),
                "score": r.get("score", 0.0)
            }
            for r in rag_results
        ]

        return {
            "draft": draft_result.get("draft", ""),
            "suggestions": suggestions,
            "response_message": "초안을 수정했습니다."
        }

    async def _handle_new_draft(
        self,
        message: str,
        target_language: str,
        conversation_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """새 초안 작성 처리"""
        logger.info(f"📝 새 초안 작성 요청")

        rag_results = await self.rag_agent.process(query=message, top_k=5)
        rag_context = [r.get("text", "") for r in rag_results if r.get("text")]

        draft_result = await self.draft_agent.process(
            original_message=message,
            rag_context=rag_context,
            target_language=target_language,
            conversation_history=conversation_history
        )

        suggestions = [
            {
                "text": r.get("text", ""),
                "chapter": r.get("chapter"),
                "section": r.get("section"),
                "score": r.get("score", 0.0)
            }
            for r in rag_results
        ]

        return {
            "draft": draft_result.get("draft", ""),
            "suggestions": suggestions,
            "response_message": "비즈니스 메시지 초안을 작성했습니다."
        }

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 조회"""
        return self._session_store.get_session(session_id)

    def delete_session(self, session_id: str):
        """세션 삭제"""
        self._session_store.delete_session(session_id)
        logger.info(f"🗑️ 세션 삭제: {session_id}")
