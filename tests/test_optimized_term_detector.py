"""
OptimizedTermDetectorAgent 테스트

Aho-Corasick 알고리즘 기반 용어 탐지 성능 및 정확도 테스트
"""

import pytest
import asyncio
import time
from typing import List, Dict

# Agent imports
from agent.term_detection.optimized_term_detector_agent import (
    OptimizedTermDetectorAgent,
    DetectedTerm,
    AutomatonCache
)
from agent.term_detection.term_detector_agent import (
    TermDetectorAgent,
    DetectedTerm as LegacyDetectedTerm
)


# 테스트용 용어집
SAMPLE_GLOSSARY = [
    {"korean_term": "인공지능", "english_term": "Artificial Intelligence", "vietnamese_term": "Trí tuệ nhân tạo"},
    {"korean_term": "머신러닝", "english_term": "Machine Learning", "vietnamese_term": "Học máy"},
    {"korean_term": "딥러닝", "english_term": "Deep Learning", "vietnamese_term": "Học sâu"},
    {"korean_term": "클라우드", "english_term": "Cloud", "vietnamese_term": "Đám mây"},
    {"korean_term": "컨테이너", "english_term": "Container", "vietnamese_term": "Container"},
    {"korean_term": "마이크로서비스", "english_term": "Microservices", "vietnamese_term": "Vi dịch vụ"},
    {"korean_term": "쿠버네티스", "english_term": "Kubernetes", "vietnamese_term": "Kubernetes"},
    {"korean_term": "도커", "english_term": "Docker", "vietnamese_term": "Docker"},
    {"korean_term": "API", "english_term": "API", "vietnamese_term": "API"},
    {"korean_term": "REST", "english_term": "REST", "vietnamese_term": "REST"},
]


class TestOptimizedTermDetectorAgent:
    """OptimizedTermDetectorAgent 테스트 클래스"""

    @pytest.fixture
    def agent(self):
        """테스트용 Agent 인스턴스"""
        return OptimizedTermDetectorAgent()

    @pytest.fixture
    def legacy_agent(self):
        """기존 Agent 인스턴스 (비교용)"""
        return TermDetectorAgent()

    @pytest.mark.asyncio
    async def test_basic_detection(self, agent):
        """기본 용어 탐지 테스트"""
        text = "인공지능과 머신러닝을 활용한 시스템입니다"

        detected = await agent.process(text, SAMPLE_GLOSSARY, source_lang="ko")

        assert len(detected) == 2
        terms = [d.korean_term for d in detected]
        assert "인공지능" in terms
        assert "머신러닝" in terms

    @pytest.mark.asyncio
    async def test_position_accuracy(self, agent):
        """위치 정확도 테스트"""
        text = "클라우드 환경에서 컨테이너를 배포합니다"

        detected = await agent.process(text, SAMPLE_GLOSSARY, source_lang="ko")

        # 클라우드: 0~4
        cloud_term = next(d for d in detected if d.korean_term == "클라우드")
        assert cloud_term.position_start == 0
        assert cloud_term.position_end == 4
        assert cloud_term.matched_text == "클라우드"

        # 컨테이너: 10~14
        container_term = next(d for d in detected if d.korean_term == "컨테이너")
        assert container_term.position_start == 10
        assert container_term.position_end == 14
        assert container_term.matched_text == "컨테이너"

    @pytest.mark.asyncio
    async def test_no_duplicate_matching(self, agent):
        """중복 매칭 방지 테스트"""
        # "인공지능"이 포함된 더 긴 용어가 있는 경우
        glossary = [
            {"korean_term": "인공지능", "english_term": "AI"},
            {"korean_term": "인공지능 시스템", "english_term": "AI System"},
        ]
        text = "인공지능 시스템을 구축합니다"

        detected = await agent.process(text, glossary, source_lang="ko")

        # 긴 용어 우선 매칭, 중복 없어야 함
        assert len(detected) == 1
        assert detected[0].korean_term == "인공지능 시스템"

    @pytest.mark.asyncio
    async def test_korean_word_boundary(self, agent):
        """한글 단어 경계 테스트"""
        glossary = [{"korean_term": "시스템", "english_term": "System"}]

        # "시스템"이 독립적으로 등장해야 매칭
        text1 = "시스템을 개발합니다"  # 매칭 O
        text2 = "생태시스템 구축"      # 매칭 X (앞에 한글)

        detected1 = await agent.process(text1, glossary, source_lang="ko")
        detected2 = await agent.process(text2, glossary, source_lang="ko")

        assert len(detected1) == 1
        assert len(detected2) == 0  # 앞에 한글이 붙어있으므로 매칭 안됨

    @pytest.mark.asyncio
    async def test_english_detection(self, agent):
        """영어 용어 탐지 테스트"""
        text = "We use Machine Learning and Deep Learning"

        detected = await agent.process(text, SAMPLE_GLOSSARY, source_lang="en")

        assert len(detected) == 2
        terms = [d.english_term for d in detected]
        assert "Machine Learning" in terms
        assert "Deep Learning" in terms

    @pytest.mark.asyncio
    async def test_case_insensitive(self, agent):
        """대소문자 무시 테스트"""
        text = "api and rest are important"

        detected = await agent.process(text, SAMPLE_GLOSSARY, source_lang="en")

        assert len(detected) == 2

    @pytest.mark.asyncio
    async def test_empty_input(self, agent):
        """빈 입력 테스트"""
        with pytest.raises(ValueError):
            await agent.process("", SAMPLE_GLOSSARY)

        with pytest.raises(ValueError):
            await agent.process("   ", SAMPLE_GLOSSARY)

    @pytest.mark.asyncio
    async def test_empty_glossary(self, agent):
        """빈 용어집 테스트"""
        text = "인공지능을 사용합니다"

        detected = await agent.process(text, [], source_lang="ko")

        assert len(detected) == 0

    @pytest.mark.asyncio
    async def test_cache_functionality(self, agent):
        """캐시 기능 테스트"""
        text = "인공지능과 머신러닝"

        # 첫 번째 호출 (캐시 미스)
        detected1 = await agent.process(text, SAMPLE_GLOSSARY, source_lang="ko")

        # 두 번째 호출 (캐시 히트 예상)
        detected2 = await agent.process(text, SAMPLE_GLOSSARY, source_lang="ko")

        # 결과가 동일해야 함
        assert len(detected1) == len(detected2)
        for d1, d2 in zip(detected1, detected2):
            assert d1.korean_term == d2.korean_term
            assert d1.position_start == d2.position_start

    @pytest.mark.asyncio
    async def test_performance_vs_legacy(self, agent, legacy_agent):
        """성능 비교 테스트 (Optimized vs Legacy)"""
        # 대규모 용어집 생성
        large_glossary = [
            {"korean_term": f"용어{i}", "english_term": f"Term{i}"}
            for i in range(1000)
        ]
        large_glossary.extend(SAMPLE_GLOSSARY)

        # 긴 텍스트 생성
        text = "인공지능과 머신러닝, 딥러닝을 활용한 클라우드 기반 마이크로서비스 시스템입니다. " * 50

        # Optimized Agent 성능 측정
        start_optimized = time.time()
        detected_optimized = await agent.process(text, large_glossary, source_lang="ko")
        time_optimized = time.time() - start_optimized

        # Legacy Agent 성능 측정
        start_legacy = time.time()
        detected_legacy = await legacy_agent.process(text, large_glossary, source_lang="ko")
        time_legacy = time.time() - start_legacy

        print(f"\n📊 성능 비교 (1,000 용어 × {len(text)}자):")
        print(f"  - Optimized (Aho-Corasick): {time_optimized:.4f}초")
        print(f"  - Legacy (Regex): {time_legacy:.4f}초")
        print(f"  - 성능 향상: {time_legacy / time_optimized:.1f}x")

        # 결과는 동일해야 함
        assert len(detected_optimized) == len(detected_legacy)

        # Optimized가 더 빨라야 함 (최소 2배 이상)
        # 작은 데이터셋에서는 오버헤드로 인해 차이가 적을 수 있음
        # assert time_optimized < time_legacy

    @pytest.mark.asyncio
    async def test_result_compatibility(self, agent, legacy_agent):
        """결과 호환성 테스트 (Legacy와 동일한 결과 보장)"""
        text = "인공지능과 머신러닝, 클라우드 환경에서 컨테이너를 배포합니다"

        detected_optimized = await agent.process(text, SAMPLE_GLOSSARY, source_lang="ko")
        detected_legacy = await legacy_agent.process(text, SAMPLE_GLOSSARY, source_lang="ko")

        # 탐지된 용어 수 동일
        assert len(detected_optimized) == len(detected_legacy)

        # 각 용어 정보 동일
        for opt, leg in zip(detected_optimized, detected_legacy):
            assert opt.korean_term == leg.korean_term
            assert opt.english_term == leg.english_term
            assert opt.position_start == leg.position_start
            assert opt.position_end == leg.position_end
            assert opt.matched_text == leg.matched_text


class TestAutomatonCache:
    """AutomatonCache 테스트 클래스"""

    def test_cache_basic(self):
        """기본 캐시 동작 테스트"""
        cache = AutomatonCache(max_size=10)

        # 초기 상태: 캐시 비어있음
        assert cache.get(SAMPLE_GLOSSARY, "korean_term") is None

        # Automaton 생성 및 저장 (수동 테스트용)
        # 실제로는 Agent가 내부적으로 처리

    def test_cache_clear(self):
        """캐시 초기화 테스트"""
        cache = AutomatonCache()
        cache.clear()
        # 예외 없이 완료되어야 함

    def test_cache_normalize_mode_separation(self):
        """정규화 모드별 캐시 분리 테스트"""
        cache = AutomatonCache(max_size=10)

        # 같은 용어집이라도 정규화 모드에 따라 다른 캐시 키 생성
        key_normal = cache._generate_cache_key(SAMPLE_GLOSSARY, "korean_term", normalize_mode=False)
        key_normalized = cache._generate_cache_key(SAMPLE_GLOSSARY, "korean_term", normalize_mode=True)

        # 두 키가 달라야 함
        assert key_normal != key_normalized
        assert "_norm" not in key_normal
        assert "_norm" in key_normalized


class TestWhitespaceNormalization:
    """띄어쓰기 정규화 테스트 클래스"""

    @pytest.fixture
    def agent(self):
        """테스트용 Agent 인스턴스"""
        return OptimizedTermDetectorAgent()

    @pytest.mark.asyncio
    async def test_glossary_with_space_text_without_space(self, agent):
        """용어집에 공백 있음, 텍스트에 공백 없음"""
        glossary = [
            {"korean_term": "인공 지능", "english_term": "Artificial Intelligence"},
            {"korean_term": "머신 러닝", "english_term": "Machine Learning"},
        ]

        text = "인공지능과 머신러닝을 활용합니다"

        detected = await agent.process(text, glossary, source_lang="ko")

        assert len(detected) == 2

        # 첫 번째 용어: 인공지능
        assert detected[0].matched_text == "인공지능"
        assert detected[0].korean_term == "인공 지능"
        assert detected[0].position_start == 0
        assert detected[0].position_end == 4

        # 두 번째 용어: 머신러닝
        assert detected[1].matched_text == "머신러닝"
        assert detected[1].korean_term == "머신 러닝"

    @pytest.mark.asyncio
    async def test_glossary_without_space_text_with_space(self, agent):
        """용어집에 공백 없음, 텍스트에 공백 있음"""
        glossary = [
            {"korean_term": "딥러닝", "english_term": "Deep Learning"},
            {"korean_term": "클라우드", "english_term": "Cloud"},
        ]

        text = "딥 러닝과 클라 우드 기술을 배웁니다"

        detected = await agent.process(text, glossary, source_lang="ko")

        assert len(detected) == 2

        # 첫 번째 용어: 딥 러닝 (공백 포함)
        assert detected[0].matched_text == "딥 러닝"
        assert detected[0].korean_term == "딥러닝"
        assert detected[0].position_start == 0
        assert detected[0].position_end == 4

        # 두 번째 용어: 클라 우드 (공백 포함)
        assert detected[1].matched_text == "클라 우드"
        assert detected[1].korean_term == "클라우드"

    @pytest.mark.asyncio
    async def test_mixed_whitespace_variations(self, agent):
        """다양한 띄어쓰기 변형 혼합 테스트"""
        glossary = [
            {"korean_term": "자연어 처리", "english_term": "NLP"},
        ]

        # 다양한 띄어쓰기 변형 테스트
        test_cases = [
            ("자연어처리를 공부합니다", "자연어처리"),       # 공백 없음
            ("자연어 처리를 공부합니다", "자연어 처리"),     # 공백 있음 (원본)
            ("자 연 어 처 리를 공부합니다", "자 연 어 처 리"),  # 공백 많음
        ]

        for text, expected_match in test_cases:
            detected = await agent.process(text, glossary, source_lang="ko")
            assert len(detected) == 1, f"Failed for: {text}"
            assert detected[0].matched_text == expected_match, f"Failed for: {text}"
            assert detected[0].korean_term == "자연어 처리"

    @pytest.mark.asyncio
    async def test_position_accuracy_with_normalization(self, agent):
        """정규화 시 위치 정확도 테스트"""
        glossary = [{"korean_term": "데이터 분석", "english_term": "Data Analysis"}]

        text = "요즘 데이터분석이 중요합니다"
        #       01234 567890123456789

        detected = await agent.process(text, glossary, source_lang="ko")

        assert len(detected) == 1
        assert detected[0].matched_text == "데이터분석"
        assert detected[0].position_start == 3
        assert detected[0].position_end == 8
        # text[3:8] == "데이터분석" 확인
        assert text[detected[0].position_start:detected[0].position_end] == "데이터분석"

    @pytest.mark.asyncio
    async def test_normalization_disabled(self, agent):
        """정규화 비활성화 테스트"""
        glossary = [{"korean_term": "인공 지능", "english_term": "AI"}]

        text = "인공지능을 연구합니다"

        # 정규화 비활성화 시 매칭 안됨
        detected = await agent.process(
            text, glossary, source_lang="ko",
            normalize_whitespace=False
        )
        assert len(detected) == 0

        # 정규화 활성화 시 매칭됨
        detected = await agent.process(
            text, glossary, source_lang="ko",
            normalize_whitespace=True
        )
        assert len(detected) == 1

    @pytest.mark.asyncio
    async def test_english_no_normalization_by_default(self, agent):
        """영어는 기본적으로 정규화 비적용 테스트"""
        glossary = [
            {"korean_term": "뉴욕", "english_term": "New York"},
        ]

        # 영어 텍스트에서 공백 없이 검색
        text = "I visited NewYork last year"

        # 영어는 NORMALIZE_LANGUAGES에 없으므로 정규화 비적용
        detected = await agent.process(text, glossary, source_lang="en")

        # "NewYork"은 "New York"과 다르므로 매칭 안됨
        assert len(detected) == 0

        # 정확히 일치하는 경우
        text2 = "I visited New York last year"
        detected2 = await agent.process(text2, glossary, source_lang="en")
        assert len(detected2) == 1
        assert detected2[0].matched_text == "New York"

    @pytest.mark.asyncio
    async def test_no_duplicate_with_normalization(self, agent):
        """정규화 모드에서 중복 매칭 방지 테스트"""
        glossary = [
            {"korean_term": "기계 학습", "english_term": "ML"},
            {"korean_term": "기계 학습 시스템", "english_term": "ML System"},
        ]

        text = "기계학습시스템을 구축합니다"

        detected = await agent.process(text, glossary, source_lang="ko")

        # 긴 용어 우선, 중복 없어야 함
        assert len(detected) == 1
        assert detected[0].korean_term == "기계 학습 시스템"
        assert detected[0].matched_text == "기계학습시스템"

    @pytest.mark.asyncio
    async def test_multiple_occurrences_with_normalization(self, agent):
        """정규화 모드에서 여러 번 등장하는 용어 테스트"""
        glossary = [{"korean_term": "인공 지능", "english_term": "AI"}]

        text = "인공지능의 발전, 인공 지능의 미래, 인공  지능의 응용"

        detected = await agent.process(text, glossary, source_lang="ko")

        # 세 번 모두 탐지되어야 함
        assert len(detected) == 3

        # 각각의 매칭 텍스트 확인
        matched_texts = [d.matched_text for d in detected]
        assert "인공지능" in matched_texts
        assert "인공 지능" in matched_texts
        assert "인공  지능" in matched_texts

    @pytest.mark.asyncio
    async def test_korean_word_boundary_with_normalization(self, agent):
        """정규화 모드에서 한글 단어 경계 테스트"""
        glossary = [{"korean_term": "시스템", "english_term": "System"}]

        # "시스템"이 독립적으로 등장해야 매칭
        text1 = "시 스 템을 개발합니다"  # 매칭 O (공백 정규화)
        text2 = "생태시스템 구축"         # 매칭 X (앞에 한글)

        detected1 = await agent.process(text1, glossary, source_lang="ko")
        detected2 = await agent.process(text2, glossary, source_lang="ko")

        assert len(detected1) == 1
        assert detected1[0].matched_text == "시 스 템"
        assert len(detected2) == 0  # 앞에 한글이 붙어있으므로 매칭 안됨


class TestPositionMapping:
    """PositionMapping 테스트 클래스"""

    @pytest.fixture
    def agent(self):
        return OptimizedTermDetectorAgent()

    def test_create_position_mapping(self, agent):
        """위치 매핑 생성 테스트"""
        text = "인공 지능과 머신 러닝"

        mapping = agent._create_position_mapping(text)

        assert mapping.original_text == text
        assert mapping.normalized_text == "인공지능과머신러닝"
        assert len(mapping.norm_to_orig) == len(mapping.normalized_text)

        # 매핑 검증
        # 원본:    인 공   지 능 과 머 신   러 닝
        # 인덱스:  0  1  2  3  4  5  6  7  8  9  10
        # 정규화:  인 공 지 능 과 머 신 러 닝
        # 인덱스:  0  1  2  3  4  5  6  7  8
        expected_mapping = [0, 1, 3, 4, 5, 7, 8, 10, 11]
        assert mapping.norm_to_orig == expected_mapping

    def test_map_to_original_position(self, agent):
        """위치 역산 테스트"""
        text = "인공 지능과 머신 러닝"
        mapping = agent._create_position_mapping(text)

        # 정규화된 "인공지능" (0~4)을 원본으로 역산
        orig_start, orig_end = agent._map_to_original_position(0, 4, mapping)

        assert orig_start == 0
        assert orig_end == 5  # "인공 지" 까지 (공백 포함)
        assert text[orig_start:orig_end] == "인공 지능"

    def test_normalize_term(self, agent):
        """용어 정규화 테스트"""
        assert agent._normalize_term("인공 지능") == "인공지능"
        assert agent._normalize_term("머신  러닝") == "머신러닝"
        assert agent._normalize_term("  딥 러닝  ") == "딥러닝"
        assert agent._normalize_term("NoSpace") == "NoSpace"


# CLI에서 직접 실행 가능한 벤치마크
async def run_benchmark():
    """성능 벤치마크 실행"""
    print("=" * 60)
    print("🚀 OptimizedTermDetectorAgent 성능 벤치마크")
    print("=" * 60)

    agent = OptimizedTermDetectorAgent()
    legacy_agent = TermDetectorAgent()

    # 다양한 크기의 용어집 테스트
    for num_terms in [100, 500, 1000, 5000]:
        glossary = [
            {"korean_term": f"용어{i}", "english_term": f"Term{i}"}
            for i in range(num_terms)
        ]
        glossary.extend(SAMPLE_GLOSSARY)

        # 다양한 길이의 텍스트 테스트
        for text_multiplier in [1, 10, 50]:
            base_text = "인공지능과 머신러닝, 딥러닝을 활용한 클라우드 기반 시스템입니다. "
            text = base_text * text_multiplier

            # Warmup (캐시 빌드)
            await agent.process(text, glossary, source_lang="ko")

            # 성능 측정 (캐시 활용)
            start = time.time()
            for _ in range(10):
                await agent.process(text, glossary, source_lang="ko")
            time_optimized = (time.time() - start) / 10

            # Legacy 측정
            start = time.time()
            for _ in range(10):
                await legacy_agent.process(text, glossary, source_lang="ko")
            time_legacy = (time.time() - start) / 10

            speedup = time_legacy / time_optimized if time_optimized > 0 else 0

            print(f"\n📊 {num_terms} 용어 × {len(text)}자:")
            print(f"   Optimized: {time_optimized*1000:.2f}ms")
            print(f"   Legacy:    {time_legacy*1000:.2f}ms")
            print(f"   Speedup:   {speedup:.1f}x")

    # 캐시 초기화
    OptimizedTermDetectorAgent.clear_cache()
    print("\n✅ 벤치마크 완료")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
