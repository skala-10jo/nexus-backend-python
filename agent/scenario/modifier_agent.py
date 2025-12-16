"""
Scenario modification agent using GPT-4o.
Modifies existing scenarios based on user chat messages.
"""
import json
import logging
from typing import Dict, Any

from agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ScenarioModifierAgent(BaseAgent):
    """
    AI agent for modifying conversation practice scenarios via chat.

    Uses GPT-4o to interpret user modification requests and update
    scenario fields accordingly.

    Example:
        >>> agent = ScenarioModifierAgent()
        >>> result = await agent.process(
        ...     current_scenario={"title": "Meeting", "userRole": "PM", ...},
        ...     user_message="역할을 더 구체적으로 바꿔줘",
        ...     language="en",
        ...     difficulty="intermediate"
        ... )
        >>> print(result["message"])
        "역할을 수정했습니다!"
    """

    # Language mapping
    LANG_MAP = {
        "en": "English",
        "ko": "Korean (한국어)",
        "zh": "Chinese (中文)",
        "ja": "Japanese (日本語)",
        "vi": "Vietnamese (Tiếng Việt)"
    }

    async def process(
        self,
        current_scenario: Dict[str, Any],
        user_message: str,
        language: str,
        difficulty: str
    ) -> Dict[str, Any]:
        """
        Modify scenario based on user chat message.

        Args:
            current_scenario: Current scenario state dictionary
            user_message: User's modification request
            language: Target language code
            difficulty: Difficulty level

        Returns:
            Dictionary with:
                - modifiedScenario: Dict of modified fields
                - message: Response message to user

        Raises:
            Exception: If GPT-4o processing fails
        """
        logger.info(f"🤖 시나리오 수정 요청: message='{user_message[:50]}...'")

        target_lang = self.LANG_MAP.get(language, "English")
        system_prompt = self._create_system_prompt(current_scenario, target_lang, difficulty)

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )

        result = json.loads(response.choices[0].message.content)
        modified_fields = result.get("modifiedFields", {})
        assistant_message = result.get("assistantMessage", "시나리오를 수정했습니다.")

        logger.info(f"✅ 시나리오 수정 완료: {len(modified_fields)} 필드 수정됨")

        return {
            "modifiedScenario": modified_fields,
            "message": assistant_message
        }

    def _create_system_prompt(
        self,
        current_scenario: Dict[str, Any],
        target_lang: str,
        difficulty: str
    ) -> str:
        """Create system prompt for scenario modification."""
        return f"""당신은 비즈니스 회화 시나리오를 수정하는 전문 AI 어시스턴트입니다.

사용자가 현재 시나리오에 대해 수정 요청을 하면, 요청사항을 분석하고 적절하게 시나리오를 수정해야 합니다.

현재 시나리오 상태:
- 제목: {current_scenario.get('title', '(없음)')}
- 설명: {current_scenario.get('description', '(없음)')}
- 시나리오 텍스트: {current_scenario.get('scenarioText', '(없음)')}
- 사용자 역할: {current_scenario.get('userRole', '(없음)')}
- AI 역할: {current_scenario.get('aiRole', '(없음)')}
- 카테고리: {current_scenario.get('category', 'General')}
- 필수 전문용어: {current_scenario.get('requiredTerminology', '(없음)')}

목표 언어: {target_lang}
난이도: {difficulty}

수정 지침:
1. 사용자의 요청을 정확하게 이해하고 해당 필드만 수정하세요
2. 요청하지 않은 필드는 그대로 유지하세요
3. 제목과 설명은 반드시 한글로 작성하세요
4. 시나리오 텍스트는 반드시 개조식으로 작성하세요
5. 역할은 간단하게 1-2 단어로 {target_lang}로 작성하세요
6. AI 역할은 "AI"나 "Assistant"가 아닌 현실적인 역할을 사용하세요
7. 난이도 {difficulty}에 맞는 적절한 복잡도로 조정하세요

다음 JSON 형식으로 응답하세요:
{{
  "modifiedFields": {{
    "title": "수정된 제목 (한글, 변경 시에만)",
    "description": "수정된 설명 (한글, 변경 시에만)",
    "scenarioText": "수정된 시나리오 텍스트 (개조식, 한글, 변경 시에만)",
    "userRole": "수정된 사용자 역할 ({target_lang}, 변경 시에만)",
    "aiRole": "수정된 AI 역할 ({target_lang}, 변경 시에만)",
    "category": "수정된 카테고리 (변경 시에만)",
    "requiredTerminology": "수정된 전문용어 (변경 시에만)"
  }},
  "assistantMessage": "사용자에게 보낼 한글 응답 메시지 (1-2문장)"
}}

중요:
- modifiedFields에는 실제로 변경된 필드만 포함하세요
- 변경하지 않은 필드는 포함하지 마세요
- assistantMessage는 친근하고 간결하게 작성하세요"""
