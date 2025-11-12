"""
용어 탐지 Agent (Term Detector Agent)

텍스트에서 용어집의 용어를 탐지하는 Micro Agent.
단일 책임: 텍스트 + 용어집 → 탐지된 용어 리스트 (위치 포함)

재사용 시나리오:
- 번역: 컨텍스트 기반 번역을 위한 용어 탐지
- 문서 분석: 문서에서 사용된 전문용어 분석
- 품질 검증: 번역 품질 체크
- 자동 완성: 사용자 입력 시 용어 제안
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import re
from agent.base_agent import BaseAgent


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


class TermDetectorAgent(BaseAgent):
    """
    텍스트에서 용어를 탐지하는 Agent

    책임: 텍스트 + 용어집 → 탐지된 용어 (위치 포함)

    이 Agent는 간단한 문자열 매칭을 통해 용어집의 용어를
    텍스트에서 찾아냅니다. 빠르고 재사용 가능하도록 설계되었습니다.

    예시:
        >>> agent = TermDetectorAgent()
        >>> glossary = [
        ...     {"korean_term": "인공지능", "english_term": "AI"},
        ...     {"korean_term": "머신러닝", "english_term": "Machine Learning"}
        ... ]
        >>> text = "인공지능과 머신러닝은 중요합니다"
        >>> detected = await agent.process(text, glossary)
        >>> len(detected)
        2
    """

    async def process(
        self,
        text: str,
        glossary_terms: List[Dict[str, str]],
        case_sensitive: bool = False
    ) -> List[DetectedTerm]:
        """
        텍스트에서 용어집의 용어를 탐지

        Args:
            text: 분석할 텍스트
            glossary_terms: 용어집 리스트 (딕셔너리 리스트)
                각 딕셔너리는 다음 키를 포함해야 함:
                - korean_term (필수): 한글 용어
                - english_term (선택): 영어 용어
                - vietnamese_term (선택): 베트남어 용어
            case_sensitive: 대소문자 구분 여부 (기본값: False)

        Returns:
            탐지된 용어 리스트 (위치 정보 포함)

        Raises:
            ValueError: 텍스트가 비어있거나 용어집이 유효하지 않은 경우
        """
        if not text or not text.strip():
            raise ValueError("텍스트가 비어있습니다")

        if not glossary_terms:
            return []

        detected_terms: List[DetectedTerm] = []

        # 길이가 긴 용어부터 매칭 (긴 용어 우선)
        sorted_terms = sorted(
            glossary_terms,
            key=lambda t: len(t.get("korean_term", "")),
            reverse=True
        )

        # 이미 매칭된 위치 추적 (중복 매칭 방지)
        matched_positions = set()

        for term_dict in sorted_terms:
            korean_term = term_dict.get("korean_term")
            if not korean_term:
                continue

            english_term = term_dict.get("english_term")
            vietnamese_term = term_dict.get("vietnamese_term")

            # 정규식 패턴 생성
            pattern = re.escape(korean_term)
            if not case_sensitive:
                pattern = f"(?i){pattern}"  # 대소문자 구분 안 함

            # 모든 매칭 찾기
            for match in re.finditer(pattern, text):
                start, end = match.span()

                # 이미 매칭된 위치와 겹치는지 확인
                if any(pos in range(start, end) for pos in matched_positions):
                    continue

                # 매칭된 위치 기록
                matched_positions.update(range(start, end))

                detected_terms.append(DetectedTerm(
                    matched_text=match.group(),
                    position_start=start,
                    position_end=end,
                    korean_term=korean_term,
                    english_term=english_term,
                    vietnamese_term=vietnamese_term
                ))

        # 위치순으로 정렬
        detected_terms.sort(key=lambda t: t.position_start)

        return detected_terms
