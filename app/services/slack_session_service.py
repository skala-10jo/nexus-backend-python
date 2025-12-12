"""
Slack Agent 세션 관리 서비스

세션 기반 대화 컨텍스트 및 번역 기능 지원을 위한 세션 관리
"""

import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

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


# 전역 세션 저장소 인스턴스 (싱글톤 패턴)
_session_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    """
    SessionStore 싱글톤 인스턴스 반환

    Returns:
        SessionStore: 전역 세션 저장소 인스턴스
    """
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store
