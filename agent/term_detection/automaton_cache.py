"""
Aho-Corasick Automaton 캐시

프로젝트별 용어집에 대한 automaton을 캐싱하여
반복적인 automaton 빌드 비용을 절감합니다.
"""

from typing import List, Dict, Optional, Any
import logging
import hashlib

logger = logging.getLogger(__name__)


class AutomatonCache:
    """
    Aho-Corasick Automaton 캐시

    프로젝트별 용어집에 대한 automaton을 캐싱하여
    반복적인 automaton 빌드 비용을 절감합니다.

    캐시 키는 용어집 해시값을 사용하여 용어집 변경 시 자동 무효화됩니다.
    """

    def __init__(self, max_size: int = 100):
        """
        Args:
            max_size: 최대 캐시 크기 (LRU 방식으로 관리)
        """
        self._cache: Dict[str, Any] = {}
        self._access_order: List[str] = []
        self._max_size = max_size

    def _generate_cache_key(
        self,
        glossary_terms: List[Dict],
        lang_field: str,
        normalize_mode: bool = False
    ) -> str:
        """
        용어집 기반 캐시 키 생성

        용어집 내용이 변경되면 다른 해시가 생성되어
        자동으로 캐시가 무효화됩니다.

        Args:
            glossary_terms: 용어집 리스트
            lang_field: 언어 필드명
            normalize_mode: 정규화 모드 여부 (True면 '_norm' 접미사 추가)

        Returns:
            캐시 키 문자열
        """
        terms_str = "|".join([
            str(t.get(lang_field, ""))
            for t in glossary_terms
            if t.get(lang_field)
        ])
        mode_suffix = "_norm" if normalize_mode else ""
        hash_value = hashlib.md5(f"{lang_field}:{terms_str}".encode()).hexdigest()[:16]
        return f"{lang_field}{mode_suffix}_{len(glossary_terms)}_{hash_value}"

    def get(
        self,
        glossary_terms: List[Dict],
        lang_field: str,
        normalize_mode: bool = False
    ) -> Optional[Any]:
        """캐시에서 automaton 조회"""
        key = self._generate_cache_key(glossary_terms, lang_field, normalize_mode)
        if key in self._cache:
            # LRU: 접근 순서 업데이트
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            logger.debug(f"Automaton cache hit: {key}")
            return self._cache[key]
        return None

    def set(
        self,
        glossary_terms: List[Dict],
        lang_field: str,
        automaton: Any,
        normalize_mode: bool = False
    ) -> None:
        """캐시에 automaton 저장"""
        key = self._generate_cache_key(glossary_terms, lang_field, normalize_mode)

        # LRU: 캐시 크기 초과 시 가장 오래된 항목 제거
        if len(self._cache) >= self._max_size and key not in self._cache:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
            logger.debug(f"Automaton cache evicted (LRU): {oldest_key}")

        self._cache[key] = automaton
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        logger.debug(f"Automaton cache stored: {key}")

    def clear(self) -> None:
        """캐시 전체 초기화"""
        self._cache.clear()
        self._access_order.clear()
        logger.info("Automaton cache cleared")


# 전역 캐시 인스턴스 (싱글톤)
_automaton_cache = AutomatonCache(max_size=50)


def get_automaton_cache() -> AutomatonCache:
    """전역 Automaton 캐시 인스턴스 반환"""
    return _automaton_cache
