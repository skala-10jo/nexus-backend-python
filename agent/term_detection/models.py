"""
용어 탐지 데이터 모델

탐지된 용어 정보와 위치 매핑을 위한 데이터 클래스들.
"""

from dataclasses import dataclass, field
from typing import List, Optional


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
        norm_to_orig: 정규화 인덱스 -> 원본 인덱스 매핑 리스트
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
