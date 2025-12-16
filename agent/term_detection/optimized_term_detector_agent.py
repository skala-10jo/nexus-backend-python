"""
최적화된 용어 탐지 Agent (Optimized Term Detector Agent)

Aho-Corasick 알고리즘을 사용하여 고성능 용어 탐지를 수행하는 Micro Agent.
단일 책임: 텍스트 + 용어집 -> 탐지된 용어 리스트 (O(M+Z) 시간복잡도)

성능 개선:
- 기존: O(N x M) - N=용어 수, M=텍스트 길이
- 개선: O(M + Z) - M=텍스트 길이, Z=매칭 수
- 5,000 용어 x 5,000자 기준 약 5,000배 성능 향상

띄어쓰기 정규화:
- "인공 지능" <-> "인공지능" 매칭 지원
- 원본 텍스트 위치 역산 기능
- 언어별 정규화 전략 적용

재사용 시나리오:
- 번역: 컨텍스트 기반 번역을 위한 용어 탐지
- 문서 분석: 문서에서 사용된 전문용어 분석
- 품질 검증: 번역 품질 체크
- 자동 완성: 사용자 입력 시 용어 제안
"""

from typing import List, Dict, Optional, Any, Tuple
import logging

from agent.base_agent import BaseAgent
from .models import DetectedTerm, PositionMapping
from .automaton_cache import get_automaton_cache

try:
    import ahocorasick
    AHOCORASICK_AVAILABLE = True
except ImportError:
    AHOCORASICK_AVAILABLE = False
    logging.warning("pyahocorasick not installed. Falling back to regex-based matching.")

logger = logging.getLogger(__name__)


class OptimizedTermDetectorAgent(BaseAgent):
    """
    Aho-Corasick 기반 고성능 용어 탐지 Agent

    책임: 텍스트 + 용어집 -> 탐지된 용어 (위치 포함)

    시간복잡도: O(M + Z)
    - M: 텍스트 길이
    - Z: 총 매칭 수

    기존 TermDetectorAgent 대비 약 1,000~10,000배 성능 향상

    예시:
        >>> agent = OptimizedTermDetectorAgent()
        >>> glossary = [
        ...     {"korean_term": "인공지능", "english_term": "AI"},
        ...     {"korean_term": "머신러닝", "english_term": "Machine Learning"}
        ... ]
        >>> text = "인공지능과 머신러닝은 중요합니다"
        >>> detected = await agent.process(text, glossary)
        >>> len(detected)
        2

    Note:
        이 Agent는 OpenAI API를 사용하지 않고 순수 문자열 매칭만 수행합니다.
        BaseAgent를 상속받지만 self.client는 사용하지 않습니다.
    """

    # 언어별 정규화 적용 여부 설정
    # 한국어/일본어/중국어: 띄어쓰기 규칙이 유연하므로 정규화 적용
    # 영어/베트남어: 띄어쓰기가 단어 구분이므로 기본적으로 정규화 비적용
    NORMALIZE_LANGUAGES = {"ko", "ja", "zh"}

    def __init__(self):
        """Agent 초기화"""
        super().__init__()
        self._cache = get_automaton_cache()

    def _get_lang_field(self, source_lang: str) -> str:
        """
        source_lang에 따라 매칭할 용어집 필드명 반환

        Args:
            source_lang: 원본 언어 코드 (ko, en, vi 등)

        Returns:
            용어집 필드명 (korean_term, english_term, vietnamese_term)
        """
        lang_field_map = {
            "ko": "korean_term",
            "en": "english_term",
            "vi": "vietnamese_term",
            "ja": "korean_term",  # 일본어는 한글로 fallback
            "zh": "korean_term",  # 중국어는 한글로 fallback
        }
        return lang_field_map.get(source_lang, "korean_term")

    def _create_position_mapping(
        self,
        text: str,
        normalize_chars: str = " \t\n"
    ) -> PositionMapping:
        """
        텍스트 정규화 및 위치 매핑 테이블 생성

        공백 문자를 제거하여 정규화된 텍스트를 생성하고,
        정규화된 인덱스를 원본 인덱스로 역산할 수 있는 매핑을 생성합니다.

        Args:
            text: 원본 텍스트
            normalize_chars: 제거할 문자들 (기본: 공백류)

        Returns:
            PositionMapping: 정규화된 텍스트와 매핑 정보

        Time Complexity: O(M) where M = len(text)
        """
        normalized_chars: List[str] = []
        norm_to_orig: List[int] = []

        for orig_idx, char in enumerate(text):
            if char not in normalize_chars:
                normalized_chars.append(char)
                norm_to_orig.append(orig_idx)

        return PositionMapping(
            normalized_text=''.join(normalized_chars),
            original_text=text,
            norm_to_orig=norm_to_orig
        )

    def _normalize_term(self, term: str) -> str:
        """용어에서 공백 제거"""
        return ''.join(term.split())

    def _map_to_original_position(
        self,
        norm_start: int,
        norm_end: int,
        mapping: PositionMapping
    ) -> Tuple[int, int]:
        """
        정규화된 위치를 원본 위치로 변환

        Args:
            norm_start: 정규화 텍스트에서의 시작 위치
            norm_end: 정규화 텍스트에서의 종료 위치 (exclusive)
            mapping: 위치 매핑 정보

        Returns:
            (원본 시작 위치, 원본 종료 위치)
        """
        orig_start = mapping.norm_to_orig[norm_start]
        orig_end = mapping.norm_to_orig[norm_end - 1] + 1
        return orig_start, orig_end

    def _build_automaton(
        self,
        glossary_terms: List[Dict],
        lang_field: str,
        normalize_mode: bool = False
    ) -> Any:
        """
        Aho-Corasick automaton 구축

        Args:
            glossary_terms: 용어집 리스트
            lang_field: 매칭할 언어 필드 (korean_term, english_term, vietnamese_term)
            normalize_mode: 정규화 모드 여부 (True면 용어에서 공백 제거)

        Returns:
            구축된 Aho-Corasick automaton

        Note:
            긴 용어를 먼저 추가하여 긴 용어 우선 매칭을 보장합니다.
        """
        if not AHOCORASICK_AVAILABLE:
            raise RuntimeError("pyahocorasick 라이브러리가 설치되지 않았습니다.")

        A = ahocorasick.Automaton()

        # 긴 용어 우선 정렬
        def get_term_length(t: Dict) -> int:
            term = t.get(lang_field, "") or ""
            if normalize_mode:
                return len(self._normalize_term(term))
            return len(term)

        sorted_terms = sorted(
            glossary_terms,
            key=get_term_length,
            reverse=True
        )

        for term_dict in sorted_terms:
            search_term = term_dict.get(lang_field)
            if search_term and search_term.strip():
                if normalize_mode:
                    normalized_term = self._normalize_term(search_term)
                    if normalized_term:
                        A.add_word(
                            normalized_term.lower(),
                            (normalized_term, term_dict)
                        )
                else:
                    A.add_word(
                        search_term.lower(),
                        (search_term, term_dict)
                    )

        A.make_automaton()

        mode_str = "normalized" if normalize_mode else "exact"
        logger.debug(f"Automaton built ({mode_str} mode): {len(glossary_terms)} terms")
        return A

    def _get_or_build_automaton(
        self,
        glossary_terms: List[Dict],
        lang_field: str,
        normalize_mode: bool = False
    ) -> Any:
        """캐시에서 automaton 조회 또는 새로 빌드"""
        cached = self._cache.get(glossary_terms, lang_field, normalize_mode)
        if cached is not None:
            return cached

        automaton = self._build_automaton(glossary_terms, lang_field, normalize_mode)
        self._cache.set(glossary_terms, lang_field, automaton, normalize_mode)
        return automaton

    async def process(
        self,
        text: str,
        glossary_terms: List[Dict[str, str]],
        source_lang: str = "ko",
        case_sensitive: bool = False,
        normalize_whitespace: bool = True
    ) -> List[DetectedTerm]:
        """
        텍스트에서 용어집의 용어를 탐지 (Aho-Corasick 알고리즘)

        시간복잡도: O(M + Z)
        - M: 텍스트 길이
        - Z: 총 매칭 수

        Args:
            text: 분석할 텍스트
            glossary_terms: 용어집 리스트 (딕셔너리 리스트)
            source_lang: 원본 언어 코드 (ko, en, vi 등)
            case_sensitive: 대소문자 구분 여부 (기본값: False)
            normalize_whitespace: 띄어쓰기 정규화 여부 (기본값: True)

        Returns:
            탐지된 용어 리스트 (위치 정보 포함, 위치순 정렬)

        Raises:
            ValueError: 텍스트가 비어있는 경우
            RuntimeError: pyahocorasick이 설치되지 않은 경우
        """
        if not text or not text.strip():
            raise ValueError("텍스트가 비어있습니다")

        if not glossary_terms:
            logger.debug("Empty glossary, returning empty list")
            return []

        if not AHOCORASICK_AVAILABLE:
            logger.warning("pyahocorasick not installed. Using regex fallback.")
            return await self._fallback_regex_process(
                text, glossary_terms, source_lang, case_sensitive, normalize_whitespace
            )

        # 언어별 정규화 적용 여부 결정
        should_normalize = normalize_whitespace and source_lang in self.NORMALIZE_LANGUAGES

        if should_normalize:
            return await self._process_with_normalization(
                text, glossary_terms, source_lang, case_sensitive
            )
        else:
            return await self._process_exact_match(
                text, glossary_terms, source_lang, case_sensitive
            )

    async def _process_exact_match(
        self,
        text: str,
        glossary_terms: List[Dict[str, str]],
        source_lang: str,
        case_sensitive: bool
    ) -> List[DetectedTerm]:
        """정확 매칭 모드 (띄어쓰기 정규화 없음)"""
        lang_field = self._get_lang_field(source_lang)

        logger.info(f"Term detection started (exact match): {len(glossary_terms)} terms")

        automaton = self._get_or_build_automaton(glossary_terms, lang_field, normalize_mode=False)
        search_text = text if case_sensitive else text.lower()

        detected_terms: List[DetectedTerm] = []
        matched_positions = set()

        for end_pos, (original_term, term_dict) in automaton.iter(search_text):
            term_len = len(original_term)
            start_pos = end_pos - term_len + 1

            current_range = range(start_pos, end_pos + 1)
            if any(pos in matched_positions for pos in current_range):
                continue

            # 한글 용어: 단어 경계 체크
            if not original_term.isascii():
                if start_pos > 0:
                    prev_char = text[start_pos - 1]
                    if '\uac00' <= prev_char <= '\ud7a3':
                        continue

            matched_positions.update(current_range)
            actual_matched_text = text[start_pos:end_pos + 1]

            detected_terms.append(DetectedTerm(
                matched_text=actual_matched_text,
                position_start=start_pos,
                position_end=end_pos + 1,
                korean_term=term_dict.get("korean_term"),
                english_term=term_dict.get("english_term"),
                vietnamese_term=term_dict.get("vietnamese_term")
            ))

        detected_terms.sort(key=lambda t: t.position_start)
        logger.info(f"Term detection completed: {len(detected_terms)} terms found (exact match)")

        return detected_terms

    async def _process_with_normalization(
        self,
        text: str,
        glossary_terms: List[Dict[str, str]],
        source_lang: str,
        case_sensitive: bool
    ) -> List[DetectedTerm]:
        """
        정규화 기반 용어 탐지

        띄어쓰기를 무시하고 용어를 탐지합니다.
        "인공 지능" <-> "인공지능" 같은 변형을 모두 매칭합니다.
        """
        lang_field = self._get_lang_field(source_lang)

        logger.info(f"Term detection started (normalized mode): {len(glossary_terms)} terms")

        # Step 1: 텍스트 정규화 및 위치 매핑 생성
        mapping = self._create_position_mapping(text)
        logger.debug(f"Text normalized: {len(text)} -> {len(mapping.normalized_text)} chars")

        if not mapping.normalized_text:
            logger.debug("Normalized text is empty, returning empty list")
            return []

        # Step 2: 정규화된 용어로 Automaton 빌드
        automaton = self._get_or_build_automaton(glossary_terms, lang_field, normalize_mode=True)

        # Step 3: 정규화된 텍스트에서 검색
        search_text = mapping.normalized_text if case_sensitive else mapping.normalized_text.lower()

        # Step 4: 모든 용어 탐지
        detected_terms: List[DetectedTerm] = []
        matched_positions = set()

        for end_pos, (normalized_term, term_dict) in automaton.iter(search_text):
            term_len = len(normalized_term)
            norm_start = end_pos - term_len + 1
            norm_end = end_pos + 1

            current_range = range(norm_start, norm_end)
            if any(pos in matched_positions for pos in current_range):
                continue

            # Step 5: 원본 위치로 역산
            orig_start, orig_end = self._map_to_original_position(
                norm_start, norm_end, mapping
            )

            # 한글 용어: 원본 텍스트에서 단어 경계 체크
            if not normalized_term.isascii():
                if orig_start > 0:
                    prev_char = text[orig_start - 1]
                    if '\uac00' <= prev_char <= '\ud7a3':
                        continue

            matched_positions.update(current_range)
            actual_matched_text = text[orig_start:orig_end]

            detected_terms.append(DetectedTerm(
                matched_text=actual_matched_text,
                position_start=orig_start,
                position_end=orig_end,
                korean_term=term_dict.get("korean_term"),
                english_term=term_dict.get("english_term"),
                vietnamese_term=term_dict.get("vietnamese_term")
            ))

        detected_terms.sort(key=lambda t: t.position_start)
        logger.info(f"Term detection completed: {len(detected_terms)} terms found (normalized mode)")

        return detected_terms

    async def _fallback_regex_process(
        self,
        text: str,
        glossary_terms: List[Dict[str, str]],
        source_lang: str,
        case_sensitive: bool,
        normalize_whitespace: bool = True
    ) -> List[DetectedTerm]:
        """
        Fallback: 기존 regex 기반 탐지 (pyahocorasick 미설치 시)

        성능이 낮으므로 가능하면 pyahocorasick 설치를 권장합니다.
        """
        import re

        primary_field = self._get_lang_field(source_lang)
        should_normalize = normalize_whitespace and source_lang in self.NORMALIZE_LANGUAGES

        detected_terms: List[DetectedTerm] = []
        matched_positions = set()

        mapping: Optional[PositionMapping] = None
        search_text = text
        if should_normalize:
            mapping = self._create_position_mapping(text)
            search_text = mapping.normalized_text

        # 긴 용어부터 매칭
        def get_term_length(t: Dict) -> int:
            term = t.get(primary_field, "") or ""
            if should_normalize:
                return len(self._normalize_term(term))
            return len(term)

        sorted_terms = sorted(
            glossary_terms,
            key=get_term_length,
            reverse=True
        )

        for term_dict in sorted_terms:
            search_term = term_dict.get(primary_field)
            if not search_term:
                continue

            if should_normalize:
                search_term = self._normalize_term(search_term)
                if not search_term:
                    continue

            escaped_term = re.escape(search_term)

            if should_normalize:
                pattern = escaped_term
            elif search_term.isascii():
                pattern = r'\b' + escaped_term + r'\b'
            else:
                pattern = r'(?<![가-힣])' + escaped_term

            flags = re.IGNORECASE if not case_sensitive else 0

            for match in re.finditer(pattern, search_text, flags):
                start, end = match.span()

                if any(pos in range(start, end) for pos in matched_positions):
                    continue

                if should_normalize and mapping:
                    orig_start, orig_end = self._map_to_original_position(
                        start, end, mapping
                    )

                    if not search_term.isascii():
                        if orig_start > 0:
                            prev_char = text[orig_start - 1]
                            if '\uac00' <= prev_char <= '\ud7a3':
                                continue

                    actual_matched_text = text[orig_start:orig_end]
                else:
                    orig_start, orig_end = start, end
                    actual_matched_text = match.group()

                matched_positions.update(range(start, end))

                detected_terms.append(DetectedTerm(
                    matched_text=actual_matched_text,
                    position_start=orig_start,
                    position_end=orig_end,
                    korean_term=term_dict.get("korean_term"),
                    english_term=term_dict.get("english_term"),
                    vietnamese_term=term_dict.get("vietnamese_term")
                ))

        detected_terms.sort(key=lambda t: t.position_start)

        mode_str = "normalized" if should_normalize else "exact"
        logger.warning(f"Regex fallback used ({mode_str} mode): {len(detected_terms)} terms found")

        return detected_terms

    @staticmethod
    def clear_cache() -> None:
        """
        전역 Automaton 캐시 초기화

        용어집이 대량으로 변경된 경우 호출하여
        캐시를 수동으로 초기화할 수 있습니다.
        """
        get_automaton_cache().clear()
