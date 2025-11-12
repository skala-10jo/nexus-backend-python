"""
기본 번역 Agent (Simple Translation Agent)

컨텍스트 없이 기본적인 텍스트 번역을 수행하는 Micro Agent.
단일 책임: 텍스트 + 언어 쌍 → 번역된 텍스트

재사용 시나리오:
- 프로젝트 없는 일반 번역
- 이메일 본문 번역
- 채팅 메시지 번역
- 짧은 문장 번역
"""

import logging
from agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SimpleTranslationAgent(BaseAgent):
    """
    기본 텍스트 번역 Agent (컨텍스트 없음)

    책임: 텍스트 + 언어 쌍 → 번역된 텍스트

    GPT-4o를 사용하여 간단하고 정확한 번역을 제공합니다.
    프로젝트 컨텍스트나 용어집 없이 순수한 번역만 수행합니다.

    예시:
        >>> agent = SimpleTranslationAgent()
        >>> result = await agent.process(
        ...     text="안녕하세요",
        ...     source_lang="ko",
        ...     target_lang="en"
        ... )
        >>> print(result)
        "Hello"
    """

    def _create_system_prompt(self, source_lang: str, target_lang: str) -> str:
        """
        시스템 프롬프트 생성

        Args:
            source_lang: 원본 언어 코드
            target_lang: 목표 언어 코드

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

        return f"""당신은 전문 번역가입니다.
{source_name} 텍스트를 {target_name}로 정확하고 자연스럽게 번역하세요.

**번역 원칙**:
1. 원문의 의미를 정확하게 전달
2. 목표 언어의 자연스러운 표현 사용
3. 문맥에 맞는 적절한 어휘 선택
4. 격식체/비격식체는 원문의 톤을 유지

**중요**:
- 번역문만 출력하고, 추가 설명이나 주석은 포함하지 마세요
- 원문에 없는 내용을 추가하지 마세요
- 가능한 한 간결하고 명확하게 번역하세요
"""

    async def process(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """
        기본 번역 수행 (컨텍스트 없음)

        Args:
            text: 번역할 원문
            source_lang: 원본 언어 코드 (ko, en, ja, vi 등)
            target_lang: 목표 언어 코드

        Returns:
            번역된 텍스트

        Raises:
            ValueError: 텍스트가 비어있거나 언어 코드가 동일한 경우
            Exception: GPT API 호출 실패 시
        """
        if not text or not text.strip():
            raise ValueError("번역할 텍스트가 비어있습니다")

        if source_lang == target_lang:
            raise ValueError("원본 언어와 목표 언어가 동일합니다")

        logger.info(f"🌐 기본 번역 시작: {source_lang} → {target_lang}")

        try:
            system_prompt = self._create_system_prompt(source_lang, target_lang)

            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,  # 일관성 있는 번역
                max_tokens=2000
            )

            translated_text = response.choices[0].message.content.strip()

            logger.info(f"✅ 번역 완료: {len(text)}자 → {len(translated_text)}자")

            return translated_text

        except Exception as e:
            logger.error(f"❌ 번역 실패: {str(e)}")
            raise Exception(f"번역 처리 중 오류 발생: {str(e)}")
