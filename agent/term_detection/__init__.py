"""
용어 탐지 Agent 모듈

텍스트에서 전문용어를 탐지하고 매칭하는 Micro Agent들

- TermDetectorAgent: 기존 regex 기반 용어 탐지 (O(N x M))
- OptimizedTermDetectorAgent: Aho-Corasick 기반 고성능 용어 탐지 (O(M+Z))
  - 띄어쓰기 정규화 지원: "인공 지능" <-> "인공지능" 매칭
- GlossaryMatcherAgent: 탐지된 용어와 용어집 상세 정보 매칭
"""

from .term_detector_agent import TermDetectorAgent, DetectedTerm
from .optimized_term_detector_agent import OptimizedTermDetectorAgent
from .models import DetectedTerm as OptimizedDetectedTerm, PositionMapping
from .automaton_cache import AutomatonCache, get_automaton_cache
from .glossary_matcher_agent import GlossaryMatcherAgent, MatchedTerm

__all__ = [
    # 기존 Agent (하위 호환)
    "TermDetectorAgent",
    "DetectedTerm",
    # 최적화된 Agent (권장)
    "OptimizedTermDetectorAgent",
    "OptimizedDetectedTerm",
    "PositionMapping",
    "AutomatonCache",
    "get_automaton_cache",
    # 용어 매칭
    "GlossaryMatcherAgent",
    "MatchedTerm",
]
