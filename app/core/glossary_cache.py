"""
용어집 캐싱 모듈 (Glossary Cache)

프로젝트별 용어집을 메모리에 캐싱하여 DB 부하를 줄이고
반복 요청 시 성능을 향상시킵니다.

사용 예시:
    >>> from app.core.glossary_cache import glossary_cache
    >>>
    >>> # 캐시 조회
    >>> terms = glossary_cache.get(project_id)
    >>> if terms is None:
    ...     terms = fetch_from_db(project_id)
    ...     glossary_cache.set(project_id, terms)
    >>>
    >>> # 용어집 변경 시 캐시 무효화
    >>> glossary_cache.invalidate(project_id)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from uuid import UUID
import logging
import threading

logger = logging.getLogger(__name__)


class GlossaryCache:
    """
    프로젝트별 용어집 TTL 캐시

    특징:
    - TTL (Time-To-Live) 기반 자동 만료
    - 프로젝트별 독립 캐싱
    - Thread-safe 구현
    - 수동 무효화 지원

    Attributes:
        ttl_seconds: 캐시 유효 시간 (초)
        max_size: 최대 캐시 항목 수 (LRU 방식)
    """

    def __init__(
        self,
        ttl_seconds: int = 300,  # 5분
        max_size: int = 100
    ):
        """
        GlossaryCache 초기화

        Args:
            ttl_seconds: 캐시 유효 시간 (기본값: 300초 = 5분)
            max_size: 최대 캐시 항목 수 (기본값: 100)
        """
        self._cache: Dict[str, tuple] = {}  # {project_id: (data, timestamp)}
        self._access_order: List[str] = []  # LRU 관리용
        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_size = max_size
        self._lock = threading.Lock()

        logger.info(f"📦 GlossaryCache 초기화: TTL={ttl_seconds}s, max_size={max_size}")

    def _normalize_key(self, project_id: UUID | str) -> str:
        """프로젝트 ID를 문자열 키로 정규화"""
        return str(project_id)

    def get(self, project_id: UUID | str) -> Optional[List[Dict[str, Any]]]:
        """
        캐시에서 용어집 조회

        Args:
            project_id: 프로젝트 ID

        Returns:
            캐시된 용어집 리스트 또는 None (캐시 미스 또는 만료)
        """
        key = self._normalize_key(project_id)

        with self._lock:
            if key not in self._cache:
                logger.debug(f"❌ 캐시 미스: project={key}")
                return None

            data, timestamp = self._cache[key]

            # TTL 확인
            if datetime.now() - timestamp > self._ttl:
                # 만료된 캐시 제거
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                logger.debug(f"⏰ 캐시 만료: project={key}")
                return None

            # LRU: 접근 순서 업데이트
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            logger.debug(f"✅ 캐시 히트: project={key}, terms={len(data)}개")
            return data

    def set(
        self,
        project_id: UUID | str,
        glossary_terms: List[Dict[str, Any]]
    ) -> None:
        """
        캐시에 용어집 저장

        Args:
            project_id: 프로젝트 ID
            glossary_terms: 용어집 리스트
        """
        key = self._normalize_key(project_id)

        with self._lock:
            # LRU: 캐시 크기 초과 시 가장 오래된 항목 제거
            if len(self._cache) >= self._max_size and key not in self._cache:
                if self._access_order:
                    oldest_key = self._access_order.pop(0)
                    if oldest_key in self._cache:
                        del self._cache[oldest_key]
                    logger.debug(f"🗑️ 캐시 제거 (LRU): project={oldest_key}")

            self._cache[key] = (glossary_terms, datetime.now())

            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            logger.debug(f"💾 캐시 저장: project={key}, terms={len(glossary_terms)}개")

    def invalidate(self, project_id: UUID | str) -> bool:
        """
        특정 프로젝트의 캐시 무효화

        용어집이 변경되었을 때 호출하여 캐시를 강제로 제거합니다.

        Args:
            project_id: 프로젝트 ID

        Returns:
            캐시가 존재하여 제거된 경우 True, 없었던 경우 False
        """
        key = self._normalize_key(project_id)

        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                logger.info(f"🔄 캐시 무효화: project={key}")
                return True
            return False

    def clear(self) -> None:
        """전체 캐시 초기화"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            logger.info("🧹 전체 캐시 초기화 완료")

    def get_stats(self) -> Dict[str, Any]:
        """
        캐시 통계 조회

        Returns:
            캐시 상태 정보 딕셔너리
        """
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl.total_seconds(),
                "cached_projects": list(self._cache.keys())
            }


# 전역 싱글톤 인스턴스
glossary_cache = GlossaryCache(ttl_seconds=300, max_size=100)


def get_glossary_cache() -> GlossaryCache:
    """
    GlossaryCache 싱글톤 인스턴스 반환

    Returns:
        전역 GlossaryCache 인스턴스
    """
    return glossary_cache
