"""
GPT-4o를 사용한 스피킹 피드백 에이전트.

음성 발화에 대한 상세한 피드백을 제공합니다:
- 문법 교정
- 개선 제안
- 점수 세부 사항 (문법, 어휘, 유창성, 명확성)
- 개선된 문장 생성
"""
import logging
import json
from typing import Dict, Any, Optional

from agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SpeakingFeedbackAgent(BaseAgent):
    """
    GPT-4o 기반 음성 발화 피드백 생성기.

    음성 전사 텍스트를 분석하고
    언어 학습을 위한 종합적인 피드백을 제공합니다.
    """

    def __init__(self):
        """BaseAgent의 OpenAI 클라이언트로 초기화합니다."""
        super().__init__()

    async def process(
        self,
        utterance_text: str,
        context: Optional[str] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        발화에 대한 피드백을 생성합니다.

        Args:
            utterance_text: 전사된 음성 텍스트
            context: 선택적 컨텍스트 (예: 'business meeting', 'presentation')
            language: 언어 코드 (예: 'en', 'ko')

        Returns:
            {
                "grammar_corrections": ["시제가 틀렸어요..."],
                "suggestions": ["더 공손한 표현으로는..."],
                "improved_sentence": "I went to the meeting...",
                "score": 7,
                "score_breakdown": {
                    "grammar": 6,
                    "vocabulary": 8,
                    "fluency": 7,
                    "clarity": 7
                }
            }
        """
        if not utterance_text or not utterance_text.strip():
            return self._empty_feedback()

        try:
            feedback = await self._generate_feedback(
                utterance_text,
                context,
                language
            )
            return feedback

        except Exception as e:
            logger.error(f"Feedback generation failed: {str(e)}")
            return self._error_feedback(str(e))

    async def _generate_feedback(
        self,
        text: str,
        context: Optional[str],
        language: str
    ) -> Dict[str, Any]:
        """GPT-4o를 사용하여 피드백을 생성합니다."""

        context_description = f" in a {context} context" if context else ""
        language_name = self._get_language_name(language)

        system_prompt = f"""당신은 비즈니스 커뮤니케이션 전문 {language_name} 언어 튜터입니다.
당신의 역할은 음성 발화를 분석하고 건설적인 피드백을 제공하는 것입니다.

다음 음성 스크립트{context_description}를 분석하고 다음을 제공하세요:

1. **문법 교정**: 구체적인 문법 오류와 설명 (한국어 학습자를 위해 한국어로)
2. **제안 사항**: 표현, 격식, 명확성을 개선하기 위한 실용적인 팁
3. **개선된 문장**: 수정/개선된 버전
4. **점수** (0-10 척도):
   - Grammar: 문법적 정확성
   - Vocabulary: 단어 선택의 적절성
   - Fluency: 자연스러운 흐름과 표현
   - Clarity: 메시지의 명확성

격려하되 솔직하게 피드백하세요. 가장 영향력 있는 개선 사항에 집중하세요.
한국어 학습자가 더 잘 이해할 수 있도록 피드백은 한국어(한국어)로 제공하세요.

중요: 반드시 유효한 JSON 형식으로만 응답하세요."""

        user_prompt = f"""다음 음성 발화를 분석하세요:

"{text}"

다음 JSON 형식으로 응답하세요:
{{
    "grammar_corrections": ["문법 교정 1", "문법 교정 2"],
    "suggestions": ["제안 1", "제안 2"],
    "improved_sentence": "개선된 문장",
    "score": 7,
    "score_breakdown": {{
        "grammar": 6,
        "vocabulary": 8,
        "fluency": 7,
        "clarity": 7
    }}
}}

발화가 이미 완벽한 경우, 격려하는 피드백과 높은 점수를 제공하세요.
JSON 객체만 반환하고, 추가 텍스트는 포함하지 마세요."""

        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content

        try:
            result = json.loads(content)

            # 구조 검증 및 정규화
            return {
                "grammar_corrections": result.get("grammar_corrections", []),
                "suggestions": result.get("suggestions", []),
                "improved_sentence": result.get("improved_sentence", text),
                "score": self._clamp_score(result.get("score", 5)),
                "score_breakdown": self._normalize_breakdown(result.get("score_breakdown", {}))
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}, content: {content}")
            return self._error_feedback("Failed to parse feedback")

    def _get_language_name(self, code: str) -> str:
        """언어 코드를 이름으로 변환합니다."""
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

    def _clamp_score(self, score: Any) -> int:
        """점수가 0-10 범위 내에 있도록 보장합니다."""
        try:
            s = int(score)
            return max(0, min(10, s))
        except (TypeError, ValueError):
            return 5

    def _normalize_breakdown(self, breakdown: Dict) -> Dict[str, int]:
        """점수 세부 사항을 정규화합니다."""
        default = {"grammar": 5, "vocabulary": 5, "fluency": 5, "clarity": 5}

        result = {}
        for key in default.keys():
            result[key] = self._clamp_score(breakdown.get(key, default[key]))

        return result

    def _empty_feedback(self) -> Dict[str, Any]:
        """빈 입력에 대한 빈 피드백을 반환합니다."""
        return {
            "grammar_corrections": [],
            "suggestions": [],
            "improved_sentence": "",
            "score": 0,
            "score_breakdown": {
                "grammar": 0,
                "vocabulary": 0,
                "fluency": 0,
                "clarity": 0
            }
        }

    def _error_feedback(self, error: str) -> Dict[str, Any]:
        """오류 정보를 포함한 피드백을 반환합니다."""
        return {
            "grammar_corrections": [f"피드백 생성 중 오류가 발생했습니다: {error}"],
            "suggestions": ["다시 시도해 주세요."],
            "improved_sentence": "",
            "score": 0,
            "score_breakdown": {
                "grammar": 0,
                "vocabulary": 0,
                "fluency": 0,
                "clarity": 0
            }
        }

    async def generate_batch_feedback(
        self,
        utterances: list,
        context: Optional[str] = None,
        language: str = "en"
    ) -> list:
        """
        여러 발화에 대한 피드백을 생성합니다.

        Args:
            utterances: {"id": str, "text": str} 리스트
            context: 선택적 컨텍스트
            language: 언어 코드

        Returns:
            {"id": str, "feedback": {...}} 리스트
        """
        results = []

        for utt in utterances:
            feedback = await self.process(
                utt.get("text", ""),
                context,
                language
            )
            results.append({
                "id": utt.get("id"),
                "feedback": feedback
            })

        return results
