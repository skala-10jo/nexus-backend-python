"""
용어 매칭 Agent (Glossary Matcher Agent)

탐지된 용어를 용어집과 매칭하여 상세 정보를 제공하는 Micro Agent.
단일 책임: 탐지된 용어 + 용어집 → 상세 용어 정보

재사용 시나리오:
- 번역: 용어 설명 표시
- 용어 자동 완성: 사용자 입력 시 용어 제안
- 용어 일관성 검증: 번역/문서 품질 체크
- 용어 툴팁: 문서 읽기 시 용어 설명 표시
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging
from agent.base_agent import BaseAgent
from .models import DetectedTerm

logger = logging.getLogger(__name__)


@dataclass
class MatchedTerm:
    """
    용어집과 매칭된 용어 상세 정보

    Attributes:
        matched_text: 문서에서 실제 매칭된 텍스트
        position_start: 원문에서의 시작 위치
        position_end: 원문에서의 종료 위치
        glossary_term_id: 용어집 용어 ID (UUID)
        korean_term: 한글 용어
        english_term: 영어 용어
        vietnamese_term: 베트남어 용어
        definition: 용어 정의
        context: 사용 맥락
        example_sentence: 예문
        note: 추가 설명
        domain: 분야
        confidence_score: 신뢰도 점수
    """
    matched_text: str
    position_start: int
    position_end: int
    glossary_term_id: str
    korean_term: str
    english_term: Optional[str] = None
    vietnamese_term: Optional[str] = None
    definition: Optional[str] = None
    context: Optional[str] = None
    example_sentence: Optional[str] = None
    note: Optional[str] = None
    domain: Optional[str] = None
    confidence_score: Optional[float] = None


class GlossaryMatcherAgent(BaseAgent):
    """
    탐지된 용어를 용어집과 매칭하는 Agent

    책임: 탐지된 용어 + 용어집 → 상세 용어 정보

    OptimizedTermDetectorAgent가 탐지한 용어를 용어집 데이터베이스와
    매칭하여 정의, 맥락, 예문 등 상세 정보를 제공합니다.

    예시:
        >>> agent = GlossaryMatcherAgent()
        >>> detected_terms = [...]  # OptimizedTermDetectorAgent 결과
        >>> glossary_full = [...]    # DB에서 조회한 전체 용어집
        >>> matched = await agent.process(detected_terms, glossary_full)
        >>> print(matched[0].definition)
        "인터넷을 통해 IT 리소스를 제공하는 서비스"
    """

    async def process(
        self,
        detected_terms: List[DetectedTerm],
        glossary_terms: List[Dict[str, Any]]
    ) -> List[MatchedTerm]:
        """
        탐지된 용어를 용어집과 매칭하여 상세 정보 제공

        Args:
            detected_terms: OptimizedTermDetectorAgent에서 탐지된 용어 리스트
            glossary_terms: 전체 용어집 (DB에서 조회, id와 상세 정보 포함)
                각 딕셔너리는 다음 키를 포함해야 함:
                - id (필수): 용어 UUID
                - korean_term (필수): 한글 용어
                - english_term (선택): 영어 용어
                - vietnamese_term (선택): 베트남어 용어
                - definition (선택): 정의
                - context (선택): 맥락
                - example_sentence (선택): 예문
                - note (선택): 추가 설명
                - domain (선택): 분야
                - confidence_score (선택): 신뢰도

        Returns:
            매칭된 용어 리스트 (상세 정보 포함)

        Raises:
            ValueError: 입력이 유효하지 않은 경우
        """
        if not detected_terms:
            logger.info("탐지된 용어가 없습니다")
            return []

        if not glossary_terms:
            logger.warning("⚠️ 용어집이 비어있습니다")
            return []

        logger.info(f"🔍 용어 매칭 시작: {len(detected_terms)}개 탐지 → {len(glossary_terms)}개 용어집")

        # 빠른 조회를 위한 용어집 맵 생성
        glossary_map = {
            term.get("korean_term"): term
            for term in glossary_terms
            if term.get("korean_term")
        }

        matched_terms: List[MatchedTerm] = []

        for detected in detected_terms:
            # 용어집에서 매칭되는 용어 찾기
            glossary_entry = glossary_map.get(detected.korean_term)

            if glossary_entry:
                matched_terms.append(MatchedTerm(
                    matched_text=detected.matched_text,
                    position_start=detected.position_start,
                    position_end=detected.position_end,
                    glossary_term_id=str(glossary_entry.get("id", "")),
                    korean_term=detected.korean_term,
                    english_term=glossary_entry.get("english_term"),
                    vietnamese_term=glossary_entry.get("vietnamese_term"),
                    definition=glossary_entry.get("definition"),
                    context=glossary_entry.get("context"),
                    example_sentence=glossary_entry.get("example_sentence"),
                    note=glossary_entry.get("note"),
                    domain=glossary_entry.get("domain"),
                    confidence_score=glossary_entry.get("confidence_score")
                ))
            else:
                logger.warning(f"⚠️ 용어집에서 매칭 실패: {detected.korean_term}")

        logger.info(f"✅ 매칭 완료: {len(matched_terms)}/{len(detected_terms)}개 성공")

        return matched_terms
