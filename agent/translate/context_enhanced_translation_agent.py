"""
컨텍스트 기반 번역 Agent (Context-Enhanced Translation Agent)

프로젝트 컨텍스트와 용어집을 활용한 고품질 번역 Micro Agent.
단일 책임: 텍스트 + 컨텍스트 + 용어집 → 정확한 전문 번역

재사용 시나리오:
- 프로젝트 기반 번역: 프로젝트 문서와 용어집 활용
- 문서 번역: 전문 분야 문서의 정확한 번역
- 전문 분야 번역: 의료, 법률, 기술 등 도메인 특화 번역
- 일관성 있는 번역: 용어 통일 및 스타일 유지
"""

import logging
from typing import List, Dict
from agent.base_agent import BaseAgent
from agent.term_detection.term_detector_agent import DetectedTerm

logger = logging.getLogger(__name__)


class ContextEnhancedTranslationAgent(BaseAgent):
    """
    컨텍스트 기반 고급 번역 Agent

    책임: 텍스트 + 컨텍스트 + 용어집 + 탐지된 용어 → 정확한 번역

    프로젝트 문서의 맥락과 전문용어사전을 활용하여
    일반 번역보다 훨씬 정확하고 일관성 있는 번역을 제공합니다.

    예시:
        >>> agent = ContextEnhancedTranslationAgent()
        >>> result = await agent.process(
        ...     text="클라우드 환경에서 컨테이너를 배포합니다",
        ...     source_lang="ko",
        ...     target_lang="en",
        ...     context="이 프로젝트는 AWS 기반 마이크로서비스...",
        ...     glossary_terms=[{"korean_term": "컨테이너", "english_term": "Container"}],
        ...     detected_terms=[...]
        ... )
        >>> print(result)
        "Deploy containers in the cloud environment"
    """

    def _create_system_prompt(
        self,
        source_lang: str,
        target_lang: str,
        context: str,
        glossary_terms: List[Dict[str, str]],
        detected_terms: List[DetectedTerm]
    ) -> str:
        """
        시스템 프롬프트 생성

        Args:
            source_lang: 원본 언어 코드
            target_lang: 목표 언어 코드
            context: 프로젝트/문서 컨텍스트
            glossary_terms: 전체 용어집
            detected_terms: 텍스트에서 탐지된 용어

        Returns:
            시스템 프롬프트 문자열
        """
        lang_names = {
            "ko": "한국어",
            "en": "영어",
            "ja": "일본어",
            "vi": "베트남어",
            "zh": "중국어"
        }

        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)

        # 탐지된 용어 포맷팅
        detected_terms_text = ""
        if detected_terms:
            detected_terms_text = "\n**번역할 텍스트에 포함된 전문용어**:\n"
            for term in detected_terms:
                english = term.english_term or ""
                vietnamese = term.vietnamese_term or ""
                detected_terms_text += f"- {term.korean_term}"
                if english:
                    detected_terms_text += f" → {english}"
                if vietnamese:
                    detected_terms_text += f" (베트남어: {vietnamese})"
                detected_terms_text += "\n"

        # 주요 용어 포맷팅 (최대 20개)
        glossary_text = ""
        if glossary_terms:
            glossary_text = "\n**전문용어사전 (참고용)**:\n"
            for i, term in enumerate(glossary_terms[:20]):
                korean = term.get("korean_term", "")
                english = term.get("english_term", "")
                vietnamese = term.get("vietnamese_term", "")
                if korean:
                    glossary_text += f"- {korean}"
                    if english:
                        glossary_text += f" → {english}"
                    if vietnamese:
                        glossary_text += f" (베트남어: {vietnamese})"
                    glossary_text += "\n"

            if len(glossary_terms) > 20:
                glossary_text += f"... 외 {len(glossary_terms) - 20}개 용어\n"

        return f"""당신은 전문 분야 번역 전문가입니다.
주어진 프로젝트 맥락과 전문용어사전을 참고하여 {source_name} 텍스트를 {target_name}로 정확하고 자연스럽게 번역하세요.

**프로젝트 맥락**:
{context}
{detected_terms_text}{glossary_text}

**번역 원칙**:
1. **전문용어 우선 적용**: 위에 표시된 전문용어는 용어집의 번역을 **반드시** 사용하세요
2. **맥락 고려**: 프로젝트의 기술 스택과 도메인을 고려하여 번역
3. **일관성 유지**: 같은 용어는 항상 같은 번역 사용
4. **자연스러운 표현**: 목표 언어의 관용적 표현 사용
5. **격식 유지**: 원문의 톤과 격식 수준 유지

**중요**:
- 탐지된 전문용어는 **반드시** 용어집의 번역을 사용하세요
- 번역문만 출력하고, 추가 설명이나 주석은 포함하지 마세요
- 원문에 없는 내용을 추가하지 마세요
- 전문용어가 아닌 일반 단어는 자연스럽게 번역하세요
"""

    async def process(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str,
        glossary_terms: List[Dict[str, str]],
        detected_terms: List[DetectedTerm]
    ) -> str:
        """
        컨텍스트 기반 번역 수행

        Args:
            text: 번역할 원문
            source_lang: 원본 언어 코드 (ko, en, ja, vi 등)
            target_lang: 목표 언어 코드
            context: 프로젝트/문서 컨텍스트 (DocumentSummarizerAgent 결과)
            glossary_terms: 프로젝트 용어집
            detected_terms: 텍스트에서 탐지된 용어 (TermDetectorAgent 결과)

        Returns:
            번역된 텍스트

        Raises:
            ValueError: 입력이 유효하지 않은 경우
            Exception: GPT API 호출 실패 시
        """
        if not text or not text.strip():
            raise ValueError("번역할 텍스트가 비어있습니다")

        if source_lang == target_lang:
            raise ValueError("원본 언어와 목표 언어가 동일합니다")

        if not context:
            logger.warning("⚠️ 컨텍스트가 비어있습니다. 기본 번역으로 대체를 고려하세요")

        logger.info(f"🌐 컨텍스트 기반 번역 시작: {source_lang} → {target_lang}")
        logger.info(f"📚 용어집: {len(glossary_terms)}개, 탐지된 용어: {len(detected_terms)}개")

        try:
            system_prompt = self._create_system_prompt(
                source_lang,
                target_lang,
                context,
                glossary_terms,
                detected_terms
            )

            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.2,  # 더 일관성 있는 번역
                max_tokens=2000
            )

            translated_text = response.choices[0].message.content.strip()

            logger.info(f"✅ 번역 완료: {len(text)}자 → {len(translated_text)}자")

            return translated_text

        except Exception as e:
            logger.error(f"❌ 번역 실패: {str(e)}")
            raise Exception(f"컨텍스트 기반 번역 중 오류 발생: {str(e)}")
