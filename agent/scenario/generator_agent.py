"""
Scenario generation agent using GPT-4o.
Generates conversation scenarios from context for language practice.
"""
import json
import logging
from typing import List, Dict, Any, Optional

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
        is_everyday: bool = False,
        user_request: Optional[str] = None,
        glossary_terms: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate conversation scenarios using GPT-4o.

        Args:
            context: Context from projects/schedules/documents
            language: Target language code (en, ko, zh, ja, vi)
            difficulty: Difficulty level (beginner, intermediate, advanced)
            count: Number of scenarios to generate
            is_everyday: If True, generate everyday scenarios instead of business
            user_request: User's specific request for scenario generation (optional)
            glossary_terms: Pre-defined glossary terms from project (optional)

        Returns:
            List of scenario dictionaries with title, description, roles, etc.

        Raises:
            ValueError: If GPT-4o fails to generate scenarios
        """
        target_lang = self.LANG_MAP.get(language, "English")

        if is_everyday:
            prompt = self._create_everyday_prompt(target_lang, difficulty, count, user_request, glossary_terms)
            system_content = self._create_everyday_system_prompt(target_lang)
        else:
            prompt = self._create_business_prompt(context, target_lang, difficulty, count, user_request, glossary_terms)
            system_content = self._create_business_system_prompt(target_lang)

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
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
            raise ValueError("Failed to generate scenarios from GPT-4o")

        return scenarios

    def _create_everyday_system_prompt(self, target_lang: str) -> str:
        """Create system prompt for everyday scenarios."""
        return f"당신은 일상생활 회화 시나리오를 만드는 전문가입니다. {target_lang} 언어 연습에 적합한 실용적이고 현실적인 일상 대화 시나리오를 생성합니다. 제목, 설명, 시나리오 텍스트는 항상 한글로 작성합니다. 역할(roles)은 반드시 한국어로 작성하고 (예: 손님, 웨이터, 직원), 절대 'AI'나 'Assistant'를 사용하지 않습니다. requiredTerminology(핵심 표현)는 반드시 {target_lang}로 작성합니다. 비즈니스나 업무 관련 시나리오는 생성하지 않습니다."

    def _create_business_system_prompt(self, target_lang: str) -> str:
        """Create system prompt for business scenarios."""
        return f"당신은 현실적인 비즈니스 회화 시나리오를 만드는 전문가입니다. {target_lang} 언어 연습에 적합한 잘 구조화된 시나리오를 생성합니다. 제목, 설명, 시나리오 텍스트는 항상 한글로 작성합니다. 역할(roles)은 반드시 한국어로 작성하고 (예: 개발자, 팀장, 고객, PM), PM, CEO, CTO 같은 영어 약어는 그대로 사용 가능합니다. 절대 'AI'나 'Assistant'를 사용하지 않습니다. requiredTerminology(핵심 용어)는 반드시 {target_lang}로 작성합니다."

    def _create_everyday_prompt(self, target_lang: str, difficulty: str, count: int, user_request: Optional[str] = None, glossary_terms: Optional[List[str]] = None) -> str:
        """Create prompt for everyday conversation scenarios."""
        user_request_section = ""
        if user_request:
            user_request_section = f"""
[사용자 요청사항 - 최우선 반영]
{user_request}
위 요청사항을 최대한 반영하여 시나리오를 생성하세요.

"""
        glossary_section = ""
        if glossary_terms and len(glossary_terms) > 0:
            terms_list = ", ".join(glossary_terms[:30])  # Limit to 30 terms
            glossary_section = f"""
[프로젝트 용어집 - 필수 사용]
다음 용어들을 requiredTerminology와 steps의 terminology에 반드시 포함하세요:
{terms_list}

"""
        return f"""다음은 일상생활에서 자주 접하는 {count}개의 실용적인 회화 시나리오를 생성해주세요.
이 시나리오는 {target_lang} 언어 연습을 위한 것입니다.
{user_request_section}{glossary_section}
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
- 다양한 일상 상황을 다루세요
- 제목과 설명은 반드시 한글로 작성
- 역할(roles)은 반드시 한국어로 작성 (단, PM, CEO, CTO 같은 영어 약어는 그대로 사용 가능)
- requiredTerminology는 반드시 {target_lang} 핵심 단어/표현으로 작성 (단어 또는 짧은 구문만, 문장 금지!)
  예: "reservation", "table for two", "check please" (O) / "I'd like to make a reservation." (X)

★★★ 중요: scenarioText 형식 규칙 ★★★
- 반드시 개조식(bullet point)으로만 작성하세요
- 각 항목은 "- " 또는 "• "로 시작해야 합니다
- 절대 번호 목록(1. 2. 3.)을 사용하지 마세요
- 절대 대화체 형식("A: ...", "B: ...")을 사용하지 마세요

★★★ 중요: steps (대화 단계) 필수 포함 ★★★
- 각 시나리오는 3-5개의 대화 단계(steps)를 포함해야 합니다
- 단계 예시: greeting(인사) → main_request(본론) → follow_up(추가 요청) → wrap_up(마무리)
- 각 단계에는 name(영문 식별자), title(한글 제목), guide(한글 가이드), terminology({target_lang} 문장/표현 2-3개)를 포함

★★★ guide 작성 규칙 (매우 중요!) ★★★
- guide는 반드시 2-3문장으로 상세하게 작성하세요
- 해당 시나리오의 구체적인 상황과 맥락을 반영해야 합니다
- 이 단계에서 무엇을 해야 하는지, 어떤 점을 주의해야 하는지 구체적으로 설명하세요
- 예시를 그대로 복사하지 말고, 시나리오 컨텍스트에 맞게 새로 작성하세요
- 좋은 예: "식당에 도착해서 직원에게 예약 여부를 알립니다. 예약이 없다면 몇 명인지, 흡연석인지 비흡연석인지 선호를 말합니다. 대기 시간이 있다면 얼마나 기다려야 하는지도 확인하세요."
- 나쁜 예: "인사를 나눕니다" (너무 짧고 일반적)

★★★ terminology 작성 규칙 ★★★
- 학습자가 실제로 말할 수 있는 완전한 {target_lang} 문장이어야 합니다
- 시나리오 상황에 맞는 구체적인 표현을 사용하세요
- 단어나 카테고리명은 금지 (예: "Ordering" 금지, "I'd like to order the pasta." 사용)

★★★ 난이도별 표현 수준 ★★★
- beginner (초급): 고등학생 수준의 쉬운 표현 사용. 짧고 간단한 문장, 기초 어휘 위주.
  예: "Can I have water?", "How much is this?", "Where is the bathroom?"
- intermediate (중급): 대학생/성인 수준의 일반적인 표현.
  예: "I'd like to order the pasta, please.", "Could you tell me where the nearest subway station is?"
- advanced (고급): 세련되고 자연스러운 원어민 수준의 표현.
  예: "I was wondering if you might have any vegetarian options available.", "Would it be possible to get a table by the window?"

다음 JSON 형식으로 시나리오를 생성하세요:
{{
  "scenarios": [
    {{
      "title": "한글로 된 시나리오 제목",
      "description": "한글로 된 간단한 설명 (2-3 문장)",
      "scenarioText": "- 첫 번째 상황 설명\\n- 두 번째 상황 설명\\n- 세 번째 상황 설명 (반드시 개조식, 번호 금지)",
      "category": "Restaurant|Hotel|Shopping|Hospital|Bank|Post Office|Cafe|Transportation|Fitness|Beauty|Real Estate|Car Rental|Daily Life",
      "roles": {{
        "user": "한국어로 된 사용자 역할",
        "ai": "한국어로 된 상대방 역할"
      }},
      "requiredTerminology": ["키워드1", "키워드2", "핵심표현3 (단어/짧은 구문만, 문장 금지)"],
      "steps": [
        {{
          "name": "단계 영문 식별자",
          "title": "단계 한글 제목",
          "guide": "이 시나리오 상황에 맞는 구체적인 가이드 (새로 작성)",
          "terminology": ["힌트용 완전한 {target_lang} 문장 2-3개 (예: I'd like to order...)"]
        }}
      ]
    }}
  ]
}}

정확히 {count}개의 일상 회화 시나리오를 "scenarios" 배열에 생성하세요."""

    def _create_business_prompt(self, context: str, target_lang: str, difficulty: str, count: int, user_request: Optional[str] = None, glossary_terms: Optional[List[str]] = None) -> str:
        """Create prompt for business conversation scenarios."""
        user_request_section = ""
        if user_request:
            user_request_section = f"""
[사용자 요청사항 - 최우선 반영]
{user_request}
위 요청사항을 최대한 반영하여 시나리오를 생성하세요.

"""
        glossary_section = ""
        if glossary_terms and len(glossary_terms) > 0:
            terms_list = ", ".join(glossary_terms[:30])  # Limit to 30 terms
            glossary_section = f"""
[프로젝트 용어집 - 필수 사용]
다음 용어들을 requiredTerminology와 steps의 terminology에 반드시 포함하세요:
{terms_list}

"""
        return f"""다음 컨텍스트를 기반으로 {count}개의 현실적인 비즈니스 회화 시나리오를 생성해주세요.
이 시나리오는 {target_lang} 언어 연습을 위한 것입니다.
{user_request_section}{glossary_section}
컨텍스트:
{context[:4000]}

★★★ 최우선 규칙: 컨텍스트 상세 내용 직접 반영 ★★★
- 컨텍스트에 나온 프로젝트명, 기술스택, 담당자, 일정, 기능 등을 시나리오에 직접 사용하세요
- 일반적인 "프로젝트 회의", "업무 협의"가 아닌, 컨텍스트의 구체적인 주제로 시나리오를 만드세요
- 예시:
  - 컨텍스트에 "Spring Boot와 Vue.js 기반 개발"이 있으면 → "Vue.js 컴포넌트 구조 논의" 시나리오 생성
  - 컨텍스트에 "UI/UX 디자인 시안 제작"이 있으면 → "UI/UX 디자인 피드백 회의" 시나리오 생성
  - 컨텍스트에 "OpenAI GPT-4 API"가 있으면 → "GPT-4 API 통합 기술 검토" 시나리오 생성
- 시나리오 제목, 설명, scenarioText, roles, terminology 모두 컨텍스트의 구체적 내용을 반영해야 합니다

요구사항:
- 난이도: {difficulty}
- 목표 언어: {target_lang}
- 각 시나리오는 컨텍스트에서 추출한 실제 업무 상황을 반영해야 합니다
- 다양한 시나리오 타입 사용: Collaboration, Technical Support, Product Explanation, Problem Solving
- 제목과 설명은 반드시 한글로 작성
- 역할(roles)은 반드시 한국어로 작성 (단, PM, CEO, CTO, QA 같은 영어 약어는 그대로 사용 가능)
- requiredTerminology는 반드시 {target_lang} 전문 용어/키워드로 작성 (단어 또는 짧은 구문만, 문장 금지!)
  예: "deadline", "RESTful API", "project timeline" (O) / "I'd like to discuss the project." (X)

★★★ 중요: scenarioText 형식 규칙 ★★★
- 반드시 개조식(bullet point)으로만 작성하세요
- 각 항목은 "- " 또는 "• "로 시작해야 합니다
- 절대 번호 목록(1. 2. 3.)을 사용하지 마세요
- 절대 대화체 형식("A: ...", "B: ...")을 사용하지 마세요
- 컨텍스트에 어떤 형식이 있더라도 무시하고 개조식으로만 작성하세요
- scenarioText에도 컨텍스트의 구체적인 내용(기술명, 기능명, 일정 등)을 포함하세요

★★★ 중요: steps (대화 단계) 필수 포함 ★★★
- 각 시나리오는 3-5개의 대화 단계(steps)를 포함해야 합니다
- 비즈니스 대화 단계 예시: ice_breaking(인사) → agenda_setting(아젠다 설정) → discussion(논의) → action_items(액션 아이템) → wrap_up(마무리)
- 각 단계에는 name(영문 식별자), title(한글 제목), guide(한글 가이드), terminology({target_lang} 문장/표현 2-3개)를 포함

★★★ guide 작성 규칙 (매우 중요!) ★★★
- guide는 반드시 2-3문장으로 상세하게 작성하세요
- 해당 시나리오의 구체적인 상황과 맥락을 반영해야 합니다
- 이 단계에서 무엇을 해야 하는지, 어떤 점을 주의해야 하는지 구체적으로 설명하세요
- 예시를 그대로 복사하지 말고, 시나리오 컨텍스트에 맞게 새로 작성하세요

★★★ terminology 작성 규칙 ★★★
- 학습자가 실제로 말할 수 있는 완전한 {target_lang} 문장이어야 합니다
- 시나리오 상황에 맞는 구체적인 표현을 사용하세요
- 컨텍스트에 나온 기술명, 기능명을 terminology 문장에 포함하세요
- 단어나 카테고리명은 금지 (예: "Greetings" 금지, "Nice to meet you." 사용)

★★★ 난이도별 표현 수준 ★★★
- beginner (초급): 고등학생 수준의 쉬운 표현 사용. 짧고 간단한 문장, 기초 어휘 위주.
  예: "I have a question.", "Can you help me?", "I think we should..."
- intermediate (중급): 대학생/직장인 수준의 일반적인 비즈니스 표현.
  예: "I'd like to discuss...", "Could you elaborate on that?", "Let me walk you through..."
- advanced (고급): 전문가 수준의 세련된 표현과 관용구 사용.
  예: "I'd like to touch base on...", "Let's circle back to...", "From a strategic standpoint..."

다음 JSON 형식으로 시나리오를 생성하세요:
{{
  "scenarios": [
    {{
      "title": "한글로 된 시나리오 제목 (컨텍스트의 구체적 주제 반영)",
      "description": "한글로 된 간단한 설명 (2-3 문장, 컨텍스트 내용 포함)",
      "scenarioText": "- 첫 번째 상황 설명 (컨텍스트 상세 내용 포함)\\n- 두 번째 상황 설명\\n- 세 번째 상황 설명 (반드시 개조식, 번호 금지)",
      "category": "Collaboration|Technical Support|Product Explanation|Problem Solving",
      "roles": {{
        "user": "한국어로 된 사용자 역할 (컨텍스트에서 추출)",
        "ai": "한국어로 된 상대방 역할 (컨텍스트에서 추출)"
      }},
      "requiredTerminology": ["키워드1", "키워드2", "전문용어3 (단어/짧은 구문만, 문장 금지)"],
      "steps": [
        {{
          "name": "단계 영문 식별자",
          "title": "단계 한글 제목",
          "guide": "이 시나리오 상황에 맞는 구체적인 가이드 (새로 작성)",
          "terminology": ["힌트용 완전한 {target_lang} 문장 2-3개 (예: I'd like to discuss...)"]
        }}
      ]
    }}
  ]
}}

정확히 {count}개의 시나리오를 "scenarios" 배열에 생성하세요."""
