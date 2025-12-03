"""
Speaking Feedback Agent using GPT-4o.

Provides detailed feedback on spoken utterances including:
- Grammar corrections
- Improvement suggestions
- Score breakdown (grammar, vocabulary, fluency, clarity)
- Improved sentence generation
"""
import logging
import json
from typing import Dict, Any, Optional

from agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SpeakingFeedbackAgent(BaseAgent):
    """
    GPT-4o based feedback generator for spoken utterances.

    Analyzes text from speech transcription and provides
    comprehensive feedback for language learning.
    """

    def __init__(self):
        """Initialize with OpenAI client from BaseAgent."""
        super().__init__()

    async def process(
        self,
        utterance_text: str,
        context: Optional[str] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Generate feedback for an utterance.

        Args:
            utterance_text: The transcribed speech text
            context: Optional context (e.g., 'business meeting', 'presentation')
            language: Language code (e.g., 'en', 'ko')

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
        """Generate feedback using GPT-4o."""

        context_description = f" in a {context} context" if context else ""
        language_name = self._get_language_name(language)

        system_prompt = f"""You are an expert {language_name} language tutor specialized in business communication.
Your task is to analyze spoken utterances and provide constructive feedback.

Analyze the following speech transcript{context_description} and provide:

1. **Grammar Corrections**: List specific grammar errors with explanations (in Korean for Korean users)
2. **Suggestions**: Practical tips to improve expression, formality, or clarity
3. **Improved Sentence**: A corrected/improved version of what was said
4. **Scores** (0-10 scale):
   - Grammar: Grammatical correctness
   - Vocabulary: Word choice appropriateness
   - Fluency: Natural flow and phrasing
   - Clarity: How clear the message is

Be encouraging but honest. Focus on the most impactful improvements.
Provide feedback in Korean (한국어) to help Korean learners understand better.

IMPORTANT: Respond in valid JSON format only."""

        user_prompt = f"""Analyze this spoken utterance:

"{text}"

Respond with a JSON object in this exact format:
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

If the utterance is already perfect, provide encouraging feedback and high scores.
Return ONLY the JSON object, no additional text."""

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

            # Validate and normalize structure
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

    def _clamp_score(self, score: Any) -> int:
        """Ensure score is within 0-10."""
        try:
            s = int(score)
            return max(0, min(10, s))
        except (TypeError, ValueError):
            return 5

    def _normalize_breakdown(self, breakdown: Dict) -> Dict[str, int]:
        """Normalize score breakdown."""
        default = {"grammar": 5, "vocabulary": 5, "fluency": 5, "clarity": 5}

        result = {}
        for key in default.keys():
            result[key] = self._clamp_score(breakdown.get(key, default[key]))

        return result

    def _empty_feedback(self) -> Dict[str, Any]:
        """Return empty feedback for empty input."""
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
        """Return feedback with error information."""
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
        Generate feedback for multiple utterances.

        Args:
            utterances: List of {"id": str, "text": str}
            context: Optional context
            language: Language code

        Returns:
            List of {"id": str, "feedback": {...}}
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
