"""
Learning Agent for Speaking Tutor.

Generates learning content including:
- Grammar explanation for corrections
- Practice exercises
- Key learning points
"""
import logging
import json
from typing import Dict, Any, List, Optional

from agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class LearningAgent(BaseAgent):
    """
    GPT-4o based learning content generator.

    Creates educational content from feedback to help users
    practice and improve their speaking skills.
    """

    def __init__(self):
        """Initialize with OpenAI client from BaseAgent."""
        super().__init__()

    async def process(
        self,
        original_text: str,
        improved_text: str,
        grammar_corrections: List[str],
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Generate learning content for an utterance.

        Args:
            original_text: The original spoken text
            improved_text: The corrected/improved text
            grammar_corrections: List of grammar corrections
            language: Language code (e.g., 'en', 'ko')

        Returns:
            {
                "key_points": ["Present tense vs past tense..."],
                "grammar_explanation": "When describing past events...",
                "practice_tips": ["Pay attention to verb endings..."],
                "alternative_expressions": ["I attended...", "I was at..."]
            }
        """
        if not original_text or not improved_text:
            return self._empty_learning()

        try:
            content = await self._generate_learning_content(
                original_text,
                improved_text,
                grammar_corrections,
                language
            )
            return content

        except Exception as e:
            logger.error(f"Learning content generation failed: {str(e)}")
            return self._error_learning(str(e))

    async def _generate_learning_content(
        self,
        original_text: str,
        improved_text: str,
        grammar_corrections: List[str],
        language: str
    ) -> Dict[str, Any]:
        """Generate learning content using GPT-4o."""

        language_name = self._get_language_name(language)
        corrections_text = "\n".join(f"- {c}" for c in grammar_corrections) if grammar_corrections else "No specific corrections"

        system_prompt = f"""당신은 한국어 학습자를 돕는 전문 {language_name} 언어 튜터입니다.
당신의 역할은 학습자가 교정 내용을 이해하고 연습할 수 있도록 교육 콘텐츠를 만드는 것입니다.

원본 텍스트와 개선된 텍스트의 비교를 바탕으로 다음을 생성하세요:

1. **핵심 학습 포인트**: 집중해야 할 2-3가지 주요 문법/어휘 개념
2. **문법 설명**: 관련된 문법 규칙에 대한 명확한 설명 (한국어로)
3. **연습 팁**: 유사한 실수를 피하기 위한 실용적인 조언
4. **대안 표현**: 같은 의미를 표현하는 2-3가지 다른 방법

격려하고, 명확하며, 실용적이어야 합니다. 한국어 학습자를 위해 설명은 한국어(한국어)로 제공하세요.

중요: 반드시 유효한 JSON 형식으로만 응답하세요."""

        user_prompt = f"""원본 텍스트:
"{original_text}"

개선된 텍스트:
"{improved_text}"

문법 교정 사항:
{corrections_text}

다음 JSON 형식으로 학습 콘텐츠를 생성하세요:
{{
    "key_points": ["핵심 학습 포인트 1", "핵심 학습 포인트 2"],
    "grammar_explanation": "문법 설명 (상세하게)",
    "practice_tips": ["연습 팁 1", "연습 팁 2"],
    "alternative_expressions": ["대안 표현 1", "대안 표현 2"]
}}

JSON 객체만 반환하고, 추가 텍스트는 포함하지 마세요."""

        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4,
            max_tokens=800,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content

        try:
            result = json.loads(content)

            return {
                "key_points": result.get("key_points", []),
                "grammar_explanation": result.get("grammar_explanation", ""),
                "practice_tips": result.get("practice_tips", []),
                "alternative_expressions": result.get("alternative_expressions", [])
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}, content: {content}")
            return self._error_learning("Failed to parse learning content")

    def _get_language_name(self, code: str) -> str:
        """Convert language code to name."""
        names = {
            "en": "English",
            "ko": "Korean",
            "ja": "Japanese",
            "zh": "Chinese",
            "de": "German",
            "fr": "French",
            "es": "Spanish"
        }
        return names.get(code, "English")

    def _empty_learning(self) -> Dict[str, Any]:
        """Return empty learning content."""
        return {
            "key_points": [],
            "grammar_explanation": "",
            "practice_tips": [],
            "alternative_expressions": []
        }

    def _error_learning(self, error: str) -> Dict[str, Any]:
        """Return learning content with error information."""
        return {
            "key_points": [f"학습 콘텐츠 생성 중 오류가 발생했습니다: {error}"],
            "grammar_explanation": "다시 시도해 주세요.",
            "practice_tips": [],
            "alternative_expressions": []
        }

    async def generate_batch_learning(
        self,
        items: List[Dict[str, Any]],
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """
        Generate learning content for multiple items.

        Args:
            items: List of {"id": str, "original": str, "improved": str, "corrections": [...]}
            language: Language code

        Returns:
            List of {"id": str, "learning": {...}}
        """
        results = []

        for item in items:
            learning = await self.process(
                original_text=item.get("original", ""),
                improved_text=item.get("improved", ""),
                grammar_corrections=item.get("corrections", []),
                language=language
            )
            results.append({
                "id": item.get("id"),
                "learning": learning
            })

        return results
