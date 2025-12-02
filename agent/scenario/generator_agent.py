"""
Scenario generation agent using GPT-4o.
Generates conversation scenarios from context for language practice.
"""
import json
import logging
from typing import List, Dict, Any

from agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ScenarioGeneratorAgent(BaseAgent):
    """
    AI agent for generating conversation practice scenarios.

    Uses GPT-4o to create realistic business or everyday conversation scenarios
    based on provided context (projects, schedules, documents).

    Example:
        >>> agent = ScenarioGeneratorAgent()
        >>> scenarios = await agent.process(
        ...     context="Project: E-commerce Platform...",
        ...     language="en",
        ...     difficulty="intermediate",
        ...     count=3,
        ...     is_everyday=False
        ... )
        >>> print(f"Generated {len(scenarios)} scenarios")
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
        context: str,
        language: str,
        difficulty: str,
        count: int,
        is_everyday: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate conversation scenarios using GPT-4o.

        Args:
            context: Context from projects/schedules/documents
            language: Target language code (en, ko, zh, ja, vi)
            difficulty: Difficulty level (beginner, intermediate, advanced)
            count: Number of scenarios to generate
            is_everyday: If True, generate everyday scenarios instead of business

        Returns:
            List of scenario dictionaries with title, description, roles, etc.

        Raises:
            ValueError: If GPT-4o fails to generate scenarios
        """
        target_lang = self.LANG_MAP.get(language, "English")

        if is_everyday:
            prompt = self._create_everyday_prompt(target_lang, difficulty, count)
            system_content = self._create_everyday_system_prompt(target_lang)
        else:
            prompt = self._create_business_prompt(context, target_lang, difficulty, count)
            system_content = self._create_business_system_prompt(target_lang)

        scenario_type = "일상 회화" if is_everyday else "비즈니스"
        logger.info(f"🤖 Calling GPT-4o for {scenario_type} scenario generation (language={language}, difficulty={difficulty}, count={count})")

        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.8
        )

        result = json.loads(response.choices[0].message.content)
        scenarios = result.get("scenarios", [])

        if not scenarios:
            logger.error("❌ GPT-4o returned no scenarios")
            raise ValueError("Failed to generate scenarios from GPT-4o")

        logger.info(f"✅ GPT-4o returned {len(scenarios)} {scenario_type} scenarios")
        return scenarios

    def _create_everyday_system_prompt(self, target_lang: str) -> str:
        """Create system prompt for everyday scenarios."""
        return f"당신은 일상생활 회화 시나리오를 만드는 전문가입니다. {target_lang} 언어 연습에 적합한 실용적이고 현실적인 일상 대화 시나리오를 생성합니다. 제목, 설명, 시나리오 텍스트는 항상 한글로 작성하고, 역할 설명은 간단하게 (1-2 단어) 작성합니다. 'ai' 역할은 '웨이터', '직원', '의사' 같은 현실적인 서비스 제공자 역할을 사용하며, 절대 'AI'나 'Assistant'를 사용하지 않습니다. 비즈니스나 업무 관련 시나리오는 생성하지 않습니다."

    def _create_business_system_prompt(self, target_lang: str) -> str:
        """Create system prompt for business scenarios."""
        return f"당신은 현실적인 비즈니스 회화 시나리오를 만드는 전문가입니다. {target_lang} 언어 연습에 적합한 잘 구조화된 시나리오를 생성합니다. 제목, 설명, 시나리오 텍스트는 항상 한글로 작성하고, 역할 설명은 간단하게 (1-2 단어) 작성합니다. 'ai' 역할은 '고객', '팀장', '동료' 같은 현실적인 상대방 역할을 사용하며, 절대 'AI'나 'Assistant'를 사용하지 않습니다."

    def _create_everyday_prompt(self, target_lang: str, difficulty: str, count: int) -> str:
        """Create prompt for everyday conversation scenarios."""
        return f"""다음은 일상생활에서 자주 접하는 {count}개의 실용적인 회화 시나리오를 생성해주세요.
이 시나리오는 {target_lang} 언어 연습을 위한 것입니다.

시나리오 타입 (다양하게 선택):
- 식당에서의 대화 (주문, 예약, 불만 처리)
- 호텔 체크인/체크아웃
- 쇼핑 (옷, 전자제품, 식료품)
- 병원/약국 방문
- 은행 업무
- 우체국/택배
- 카페에서 주문
- 교통 수단 이용 (택시, 지하철, 버스)
- 헬스장/피트니스 센터
- 미용실/헤어샵
- 부동산 문의
- 렌터카 대여

요구사항:
- 난이도: {difficulty}
- 목표 언어: {target_lang}
- 각 시나리오는 일상생활에서 실제로 겪을 수 있는 상황을 반영
- 해당 상황에서 자주 사용하는 3-5개의 실용적인 표현이나 어휘를 포함
- 다양한 일상 상황을 다루세요
- 제목과 설명은 반드시 한글로 작성
- 역할 설명은 간단하고 간결하게 (1-2 단어)

다음 JSON 형식으로 시나리오를 생성하세요:
{{
  "scenarios": [
    {{
      "title": "한글로 된 시나리오 제목",
      "description": "한글로 된 간단한 설명 (2-3 문장)",
      "scenarioText": "개조식으로 작성된 상세한 시나리오 설명 (한글)",
      "category": "Restaurant|Hotel|Shopping|Hospital|Bank|Post Office|Cafe|Transportation|Fitness|Beauty|Real Estate|Car Rental|Daily Life",
      "roles": {{
        "user": "{target_lang}로 된 간단한 사용자 역할",
        "ai": "{target_lang}로 된 간단한 상대방 역할"
      }},
      "requiredTerminology": ["표현1", "표현2", "표현3"]
    }}
  ]
}}

정확히 {count}개의 일상 회화 시나리오를 "scenarios" 배열에 생성하세요."""

    def _create_business_prompt(self, context: str, target_lang: str, difficulty: str, count: int) -> str:
        """Create prompt for business conversation scenarios."""
        return f"""다음 컨텍스트를 기반으로 {count}개의 현실적인 비즈니스 회화 시나리오를 생성해주세요.
이 시나리오는 {target_lang} 언어 연습을 위한 것입니다.

컨텍스트:
{context[:3000]}

요구사항:
- 난이도: {difficulty}
- 목표 언어: {target_lang}
- 각 시나리오는 현실적인 비즈니스 상황을 반영해야 합니다
- 컨텍스트에서 3-5개의 핵심 전문 용어를 식별하세요
- 다양한 시나리오 타입 사용: Collaboration, Technical Support, Product Explanation, Problem Solving
- 제목과 설명은 반드시 한글로 작성
- 역할 설명은 간단하고 간결하게 (1-2 단어)

다음 JSON 형식으로 시나리오를 생성하세요:
{{
  "scenarios": [
    {{
      "title": "한글로 된 시나리오 제목",
      "description": "한글로 된 간단한 설명 (2-3 문장)",
      "scenarioText": "개조식으로 작성된 상세한 시나리오 설명 (한글)",
      "category": "Collaboration|Technical Support|Product Explanation|Problem Solving",
      "roles": {{
        "user": "{target_lang}로 된 간단한 사용자 역할 (1-2 단어)",
        "ai": "{target_lang}로 된 간단한 상대방 역할 (1-2 단어)"
      }},
      "requiredTerminology": ["용어1", "용어2", "용어3"]
    }}
  ]
}}

정확히 {count}개의 시나리오를 "scenarios" 배열에 생성하세요."""
