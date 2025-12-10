"""
최적화된 용어 탐지 Agent (Optimized Term Detector Agent)

Aho-Corasick 알고리즘을 사용하여 고성능 용어 탐지를 수행하는 Micro Agent.
단일 책임: 텍스트 + 용어집 → 탐지된 용어 리스트 (O(M+Z) 시간복잡도)

성능 개선:
- 기존: O(N × M) - N=용어 수, M=텍스트 길이
- 개선: O(M + Z) - M=텍스트 길이, Z=매칭 수
- 5,000 용어 × 5,000자 기준 약 5,000배 성능 향상

띄어쓰기 정규화:
- "인공 지능" ↔ "인공지능" 매칭 지원
- 원본 텍스트 위치 역산 기능
- 언어별 정규화 전략 적용

재사용 시나리오:
- 번역: 컨텍스트 기반 번역을 위한 용어 탐지
- 문서 분석: 문서에서 사용된 전문용어 분석
- 품질 검증: 번역 품질 체크
- 자동 완성: 사용자 입력 시 용어 제안
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import logging
import hashlib
from agent.base_agent import BaseAgent

try:
    import ahocorasick
    AHOCORASICK_AVAILABLE = True
except ImportError:
    AHOCORASICK_AVAILABLE = False
    logging.warning("⚠️ pyahocorasick not installed. Falling back to regex-based matching.")

logger = logging.getLogger(__name__)


@dataclass
class DetectedTerm:
    """
    탐지된 용어 정보

    Attributes:
        matched_text: 문서에서 실제 매칭된 텍스트
        position_start: 원문에서의 시작 위치 (인덱스)
        position_end: 원문에서의 종료 위치 (인덱스)
        korean_term: 용어집의 한글 용어
        english_term: 용어집의 영어 용어 (있는 경우)
        vietnamese_term: 용어집의 베트남어 용어 (있는 경우)
    """
    matched_text: str
    position_start: int
    position_end: int
    korean_term: str
    english_term: Optional[str] = None
    vietnamese_term: Optional[str] = None


@dataclass
class PositionMapping:
    """
    정규화된 텍스트와 원본 텍스트 간의 위치 매핑

    띄어쓰기 정규화 시 원본 텍스트의 위치를 역산하기 위한 매핑 정보.

    Attributes:
        normalized_text: 공백이 제거된 정규화 텍스트
        original_text: 원본 텍스트
        norm_to_orig: 정규화 인덱스 → 원본 인덱스 매핑 리스트
            norm_to_orig[i] = 정규화 텍스트의 i번째 문자가 원본의 몇 번째 위치인지

    Example:
        >>> text = "인공 지능과 머신 러닝"
        >>> mapping = PositionMapping(...)
        >>> mapping.normalized_text
        '인공지능과머신러닝'
        >>> mapping.norm_to_orig
        [0, 1, 3, 4, 5, 7, 8, 10, 11]
    """
    normalized_text: str
    original_text: str
    norm_to_orig: List[int] = field(default_factory=list)


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
        # 용어집의 해당 언어 필드만 추출하여 해시 생성
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
            logger.debug(f"🎯 Automaton 캐시 히트: {key}")
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
            logger.debug(f"🗑️ Automaton 캐시 제거 (LRU): {oldest_key}")

        self._cache[key] = automaton
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        logger.debug(f"💾 Automaton 캐시 저장: {key}")

    def clear(self) -> None:
        """캐시 전체 초기화"""
        self._cache.clear()
        self._access_order.clear()
        logger.info("🧹 Automaton 캐시 초기화 완료")


# 전역 캐시 인스턴스 (싱글톤)
_automaton_cache = AutomatonCache(max_size=50)


class OptimizedTermDetectorAgent(BaseAgent):
    """
    Aho-Corasick 기반 고성능 용어 탐지 Agent

    책임: 텍스트 + 용어집 → 탐지된 용어 (위치 포함)

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
        self._cache = _automaton_cache

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

        Example:
            >>> text = "인공 지능과 머신 러닝"
            >>> mapping = self._create_position_mapping(text)
            >>> mapping.normalized_text
            '인공지능과머신러닝'
            >>> mapping.norm_to_orig
            [0, 1, 3, 4, 5, 7, 8, 10, 11]

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
        """
        용어에서 공백 제거

        Args:
            term: 용어 문자열

        Returns:
            공백이 제거된 용어

        Example:
            >>> self._normalize_term("인공 지능")
            '인공지능'
        """
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

        Example:
            >>> # 원본: "인공 지능과" → 정규화: "인공지능과"
            >>> # "인공지능" 매칭 시 norm_start=0, norm_end=4
            >>> orig_start, orig_end = self._map_to_original_position(0, 4, mapping)
            >>> orig_start, orig_end
            (0, 5)  # 원본에서 "인공 지능"의 범위
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
            normalize_mode=True일 때는 정규화된 용어(공백 제거)로 automaton을 빌드합니다.
        """
        if not AHOCORASICK_AVAILABLE:
            raise RuntimeError("pyahocorasick 라이브러리가 설치되지 않았습니다.")

        A = ahocorasick.Automaton()

        # 긴 용어 우선 정렬 (긴 용어가 짧은 용어를 포함할 수 있으므로)
        # 정규화 모드에서는 정규화된 길이 기준으로 정렬
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
                # 정규화 모드: 공백 제거
                if normalize_mode:
                    normalized_term = self._normalize_term(search_term)
                    if normalized_term:
                        # 소문자로 저장 (대소문자 무시 매칭용)
                        # value: (정규화된 텍스트, 전체 용어 정보)
                        A.add_word(
                            normalized_term.lower(),
                            (normalized_term, term_dict)
                        )
                else:
                    # 일반 모드: 원본 그대로
                    A.add_word(
                        search_term.lower(),
                        (search_term, term_dict)
                    )

        A.make_automaton()

        mode_str = "정규화" if normalize_mode else "일반"
        logger.debug(f"🔧 Automaton 빌드 완료 ({mode_str} 모드): {len(glossary_terms)}개 용어")
        return A

    def _get_or_build_automaton(
        self,
        glossary_terms: List[Dict],
        lang_field: str,
        normalize_mode: bool = False
    ) -> Any:
        """
        캐시에서 automaton 조회 또는 새로 빌드

        Args:
            glossary_terms: 용어집 리스트
            lang_field: 매칭할 언어 필드
            normalize_mode: 정규화 모드 여부

        Returns:
            Aho-Corasick automaton
        """
        # 캐시 확인 (정규화 모드별 별도 캐시)
        cached = self._cache.get(glossary_terms, lang_field, normalize_mode)
        if cached is not None:
            return cached

        # 캐시 미스: 새로 빌드
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
                각 딕셔너리는 다음 키를 포함해야 함:
                - korean_term (필수): 한글 용어
                - english_term (선택): 영어 용어
                - vietnamese_term (선택): 베트남어 용어
            source_lang: 원본 언어 코드 (ko, en, vi 등) - 해당 언어 용어 매칭
            case_sensitive: 대소문자 구분 여부 (기본값: False)
            normalize_whitespace: 띄어쓰기 정규화 여부 (기본값: True)
                - True: "인공 지능" == "인공지능" 으로 매칭 (한국어/일본어/중국어)
                - False: 정확한 문자열만 매칭

        Returns:
            탐지된 용어 리스트 (위치 정보 포함, 위치순 정렬)

        Raises:
            ValueError: 텍스트가 비어있는 경우
            RuntimeError: pyahocorasick이 설치되지 않은 경우

        Example:
            >>> agent = OptimizedTermDetectorAgent()
            >>> text = "클라우드 환경에서 컨테이너를 배포합니다"
            >>> glossary = [
            ...     {"korean_term": "클라우드", "english_term": "Cloud"},
            ...     {"korean_term": "컨테이너", "english_term": "Container"}
            ... ]
            >>> detected = await agent.process(text, glossary, source_lang="ko")
            >>> for term in detected:
            ...     print(f"{term.matched_text} at [{term.position_start}:{term.position_end}]")
            클라우드 at [0:4]
            컨테이너 at [10:14]

            # 띄어쓰기 정규화 예시
            >>> glossary = [{"korean_term": "인공 지능", "english_term": "AI"}]
            >>> text = "인공지능을 활용합니다"  # 띄어쓰기 없음
            >>> detected = await agent.process(text, glossary, source_lang="ko")
            >>> detected[0].matched_text
            '인공지능'  # 원본 텍스트 기준
        """
        if not text or not text.strip():
            raise ValueError("텍스트가 비어있습니다")

        if not glossary_terms:
            logger.debug("용어집이 비어있어 빈 리스트 반환")
            return []

        if not AHOCORASICK_AVAILABLE:
            logger.warning("⚠️ pyahocorasick 미설치. 기존 regex 방식으로 폴백합니다.")
            return await self._fallback_regex_process(
                text, glossary_terms, source_lang, case_sensitive, normalize_whitespace
            )

        # 언어별 정규화 적용 여부 결정
        # 한국어/일본어/중국어: 띄어쓰기 규칙이 유연하므로 정규화 적용
        # 영어/베트남어: 띄어쓰기가 단어 구분이므로 기본적으로 정규화 비적용
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
        """
        정확 매칭 모드 (기존 로직)

        띄어쓰기 정규화 없이 정확한 문자열 매칭을 수행합니다.
        """
        lang_field = self._get_lang_field(source_lang)

        logger.info(f"🔍 Aho-Corasick 용어 탐지 시작 (정확 매칭): {len(glossary_terms)}개 용어")

        # Automaton 가져오기 (캐시 또는 빌드)
        automaton = self._get_or_build_automaton(glossary_terms, lang_field, normalize_mode=False)

        # 검색용 텍스트 (대소문자 처리)
        search_text = text if case_sensitive else text.lower()

        # O(M) 단일 패스로 모든 용어 탐지
        detected_terms: List[DetectedTerm] = []
        matched_positions = set()  # 중복 매칭 방지

        for end_pos, (original_term, term_dict) in automaton.iter(search_text):
            term_len = len(original_term)
            start_pos = end_pos - term_len + 1

            # 이미 매칭된 위치와 겹치는지 확인 (긴 용어 우선)
            current_range = range(start_pos, end_pos + 1)
            if any(pos in matched_positions for pos in current_range):
                continue

            # 한글 용어의 경우: 앞에 한글이 있으면 스킵 (단어 경계 체크)
            if not original_term.isascii():
                if start_pos > 0:
                    prev_char = text[start_pos - 1]
                    # 앞 문자가 한글이면 스킵
                    if '\uac00' <= prev_char <= '\ud7a3':
                        continue

            # 매칭된 위치 기록
            matched_positions.update(current_range)

            # 원문에서 실제 매칭된 텍스트 추출 (대소문자 원본 유지)
            actual_matched_text = text[start_pos:end_pos + 1]

            detected_terms.append(DetectedTerm(
                matched_text=actual_matched_text,
                position_start=start_pos,
                position_end=end_pos + 1,  # 기존 API와 호환 (exclusive end)
                korean_term=term_dict.get("korean_term"),
                english_term=term_dict.get("english_term"),
                vietnamese_term=term_dict.get("vietnamese_term")
            ))

        # 위치순으로 정렬
        detected_terms.sort(key=lambda t: t.position_start)

        logger.info(f"✅ 용어 탐지 완료: {len(detected_terms)}개 탐지 (정확 매칭)")

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
        "인공 지능" ↔ "인공지능" 같은 변형을 모두 매칭합니다.

        시간복잡도: O(M + Z) 유지 (정규화에 O(M) 추가)

        Args:
            text: 원본 텍스트
            glossary_terms: 용어집 리스트
            source_lang: 원본 언어 코드
            case_sensitive: 대소문자 구분 여부

        Returns:
            탐지된 용어 리스트 (원본 텍스트 기준 위치 정보 포함)
        """
        lang_field = self._get_lang_field(source_lang)

        logger.info(f"🔍 Aho-Corasick 용어 탐지 시작 (정규화 모드): {len(glossary_terms)}개 용어")

        # Step 1: 텍스트 정규화 및 위치 매핑 생성
        mapping = self._create_position_mapping(text)
        logger.debug(f"📝 텍스트 정규화: {len(text)}자 → {len(mapping.normalized_text)}자")

        if not mapping.normalized_text:
            logger.debug("정규화된 텍스트가 비어있어 빈 리스트 반환")
            return []

        # Step 2: 정규화된 용어로 Automaton 빌드
        automaton = self._get_or_build_automaton(glossary_terms, lang_field, normalize_mode=True)

        # Step 3: 정규화된 텍스트에서 검색
        search_text = mapping.normalized_text if case_sensitive else mapping.normalized_text.lower()

        # Step 4: O(M) 단일 패스로 모든 용어 탐지
        detected_terms: List[DetectedTerm] = []
        matched_positions = set()  # 정규화된 위치 기준 중복 방지

        for end_pos, (normalized_term, term_dict) in automaton.iter(search_text):
            term_len = len(normalized_term)
            norm_start = end_pos - term_len + 1
            norm_end = end_pos + 1  # exclusive

            # 이미 매칭된 위치와 겹치는지 확인 (긴 용어 우선)
            current_range = range(norm_start, norm_end)
            if any(pos in matched_positions for pos in current_range):
                continue

            # Step 5: 원본 위치로 역산 (단어 경계 체크보다 먼저 수행)
            orig_start, orig_end = self._map_to_original_position(
                norm_start, norm_end, mapping
            )

            # 한글 용어의 경우: 원본 텍스트에서 앞에 한글이 있으면 스킵 (단어 경계 체크)
            # 정규화된 텍스트가 아닌 원본 텍스트 기준으로 체크해야 함
            # (정규화 시 공백이 제거되어 원래 분리된 단어가 붙어버릴 수 있으므로)
            if not normalized_term.isascii():
                if orig_start > 0:
                    prev_char = text[orig_start - 1]
                    # 앞 문자가 한글이면 스킵 (공백, 특수문자 등은 OK)
                    if '\uac00' <= prev_char <= '\ud7a3':
                        continue

            # 매칭된 위치 기록 (정규화된 위치)
            matched_positions.update(current_range)

            # 원본 텍스트에서 실제 매칭된 문자열 추출
            actual_matched_text = text[orig_start:orig_end]

            detected_terms.append(DetectedTerm(
                matched_text=actual_matched_text,
                position_start=orig_start,
                position_end=orig_end,
                korean_term=term_dict.get("korean_term"),
                english_term=term_dict.get("english_term"),
                vietnamese_term=term_dict.get("vietnamese_term")
            ))

        # 위치순으로 정렬
        detected_terms.sort(key=lambda t: t.position_start)

        logger.info(f"✅ 용어 탐지 완료: {len(detected_terms)}개 탐지 (정규화 모드)")

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

        Args:
            text: 검색할 텍스트
            glossary_terms: 용어집 리스트
            source_lang: 원본 언어 코드
            case_sensitive: 대소문자 구분 여부
            normalize_whitespace: 띄어쓰기 정규화 여부

        Returns:
            탐지된 용어 리스트
        """
        import re

        primary_field = self._get_lang_field(source_lang)

        # 정규화 모드 결정
        should_normalize = normalize_whitespace and source_lang in self.NORMALIZE_LANGUAGES

        detected_terms: List[DetectedTerm] = []
        matched_positions = set()

        # 정규화 모드일 때 매핑 생성
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

            # 정규화 모드일 때 용어도 정규화
            if should_normalize:
                search_term = self._normalize_term(search_term)
                if not search_term:
                    continue

            escaped_term = re.escape(search_term)

            # 정규화 모드에서는 단어 경계 체크를 나중에 원본 텍스트 기준으로 수행
            if should_normalize:
                # 정규화 모드: 단어 경계 없이 매칭 (나중에 원본에서 체크)
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

                # 정규화 모드일 때 원본 위치로 역산
                if should_normalize and mapping:
                    orig_start, orig_end = self._map_to_original_position(
                        start, end, mapping
                    )

                    # 원본 텍스트에서 단어 경계 체크 (한글 용어)
                    if not search_term.isascii():
                        if orig_start > 0:
                            prev_char = text[orig_start - 1]
                            # 앞 문자가 한글이면 스킵
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

        mode_str = "정규화" if should_normalize else "정확"
        logger.warning(f"⚠️ Regex fallback 사용 ({mode_str} 모드): {len(detected_terms)}개 탐지")

        return detected_terms

    @staticmethod
    def clear_cache() -> None:
        """
        전역 Automaton 캐시 초기화

        용어집이 대량으로 변경된 경우 호출하여
        캐시를 수동으로 초기화할 수 있습니다.
        """
        _automaton_cache.clear()
