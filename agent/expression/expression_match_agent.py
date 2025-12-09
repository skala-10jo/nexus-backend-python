"""
ExpressionMatchAgent: 비즈니스 표현과 예문 매칭 Agent

expressions.json의 expression이 예문에서 어디에 해당하는지 GPT로 찾습니다.

문제 케이스:
1. expression: "take (someone) through" → 예문: "take me through"
2. expression: "catch (someone) up on (something)" → 예문: "catch me up on what..."
3. expression: "follow up on ~" → 예문: "follow up on the issue"

Author: NEXUS Team
Date: 2025-01-18
"""
from agent.base_agent import BaseAgent
from typing import Dict, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)


class ExpressionMatchAgent(BaseAgent):
    """
    Expression-Sentence 매칭 Agent

    expression 패턴이 sentence 어디에 해당하는지 GPT로 분석합니다.
    정규식으로 매칭이 안 되는 경우의 Fallback으로 사용됩니다.

    특징:
    - (someone), (something) 같은 변수 처리
    - ~, ... 같은 특수 기호 처리
    - one's → my, your, his 등 대명사 변형 처리

    Example:
        >>> agent = ExpressionMatchAgent()
        >>> result = await agent.process(
        ...     expression="take (someone) through",
        ...     sentence="Can you take me through the budget proposal?"
        ... )
        >>> print(result)
        {
            "matched": True,
            "start_index": 8,
            "end_index": 23,
            "matched_text": "take me through"
        }
    """

    async def process(
        self,
        expression: str,
        sentence: str
    ) -> Dict[str, Any]:
        """
        expression이 sentence 어디에 해당하는지 찾습니다.

        Args:
            expression: 비즈니스 표현 (예: "take (someone) through")
            sentence: 예문 (예: "Can you take me through the budget proposal?")

        Returns:
            Dict[str, Any]: {
                "matched": bool,        # 매칭 성공 여부
                "start_index": int,     # 시작 인덱스 (0-based)
                "end_index": int,       # 끝 인덱스 (exclusive)
                "matched_text": str     # 실제 매칭된 텍스트
            }

        Raises:
            ValueError: expression 또는 sentence가 비어있을 때
        """
        if not expression or not expression.strip():
            raise ValueError("expression is required")
        if not sentence or not sentence.strip():
            raise ValueError("sentence is required")

        try:
            logger.info(f"Finding match: expression='{expression}' in sentence='{sentence[:50]}...'")

            system_prompt = """You are an expert at matching business English expressions to sentences.

Your task is to find where an expression pattern appears in a sentence, even when:
1. Variables like (someone), (something) are replaced with actual words (me, you, him, the project, etc.)
2. Special symbols like ~, ... indicate continuation or variation
3. Possessive pronouns like one's become my, your, his, her, their, etc.

IMPORTANT: Return ONLY valid JSON, no other text.

Response format:
{
  "matched": true/false,
  "start_index": <0-based start position in sentence>,
  "end_index": <exclusive end position>,
  "matched_text": "<exact text from sentence that matches the expression>"
}

Examples:
- Expression: "take (someone) through"
  Sentence: "Can you take me through the budget proposal?"
  Response: {"matched": true, "start_index": 8, "end_index": 23, "matched_text": "take me through"}

- Expression: "catch (someone) up on (something)"
  Sentence: "Can you catch me up on what happened?"
  Response: {"matched": true, "start_index": 8, "end_index": 32, "matched_text": "catch me up on what happened"}

- Expression: "one's responsibility"
  Sentence: "It's your responsibility to finish this."
  Response: {"matched": true, "start_index": 5, "end_index": 25, "matched_text": "your responsibility"}"""

            user_prompt = f"""Find where this expression appears in the sentence:

Expression: "{expression}"
Sentence: "{sentence}"

Return JSON with the match information."""

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # 빠르고 저렴한 모델 사용
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # 일관된 결과를 위해 낮은 temperature
                max_tokens=200
            )

            result_text = response.choices[0].message.content.strip()

            # JSON 파싱 시도
            try:
                # ```json ... ``` 형식 제거
                if result_text.startswith("```"):
                    result_text = result_text.split("```")[1]
                    if result_text.startswith("json"):
                        result_text = result_text[4:]
                    result_text = result_text.strip()

                result = json.loads(result_text)

                # 필수 필드 검증
                if "matched" not in result:
                    result["matched"] = False

                if result.get("matched"):
                    # 인덱스 검증
                    start = result.get("start_index", 0)
                    end = result.get("end_index", 0)
                    matched_text = result.get("matched_text", "")

                    # 실제 문장에서 해당 위치의 텍스트와 비교 검증
                    if 0 <= start < end <= len(sentence):
                        actual_text = sentence[start:end]
                        if actual_text.lower() != matched_text.lower():
                            # GPT가 반환한 텍스트와 실제 인덱스가 맞지 않으면 재계산
                            matched_lower = matched_text.lower()
                            sentence_lower = sentence.lower()
                            if matched_lower in sentence_lower:
                                new_start = sentence_lower.find(matched_lower)
                                new_end = new_start + len(matched_text)
                                result["start_index"] = new_start
                                result["end_index"] = new_end
                                result["matched_text"] = sentence[new_start:new_end]

                logger.info(f"Match result: matched={result.get('matched')}, text='{result.get('matched_text', '')}'")
                return result

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse GPT response as JSON: {result_text}")
                return {
                    "matched": False,
                    "start_index": 0,
                    "end_index": 0,
                    "matched_text": "",
                    "error": f"JSON parse error: {str(e)}"
                }

        except Exception as e:
            logger.error(f"Expression matching failed: {str(e)}", exc_info=True)
            return {
                "matched": False,
                "start_index": 0,
                "end_index": 0,
                "matched_text": "",
                "error": str(e)
            }


# 편의 함수
def get_expression_match_agent() -> ExpressionMatchAgent:
    """
    ExpressionMatchAgent 인스턴스 반환

    Returns:
        ExpressionMatchAgent 인스턴스
    """
    return ExpressionMatchAgent()
