"""
사용자 메시지 피드백 생성 Agent.

시나리오 회화에서 사용자의 메시지에 대해 문법, 어휘, 발음 등
종합적인 피드백을 생성합니다.
"""
import json
import logging
from typing import Dict, Any, List, Optional

from agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class FeedbackAgent(BaseAgent):
    """
    사용자 메시지 피드백 생성 Agent.

    책임: 사용자 메시지에 대한 문법, 어휘, 발음 피드백 생성.

    Returns:
        Dict containing:
        - grammar_corrections: 문법 교정 리스트
        - terminology_usage: 용어 사용 피드백
        - suggestions: 개선 제안
        - pronunciation_feedback: 발음 피드백 (있는 경우)
        - score: 종합 점수 (1-10)
        - score_breakdown: 세부 점수
    """

    DIFFICULTY_CRITERIA = {
        "beginner": """BEGINNER 기준:
- 문법 (30%): 기본 문장 구조, 현재/과거 시제만 검사, 간단한 관사 사용
- 어휘 (25%): 기본 일상 어휘 사용 여부, 복잡한 표현 요구하지 않음
- 유창성 (25%): 의사소통 가능 여부에 집중, 완벽한 문장 구조 요구하지 않음
- 발음 (20%): 이해 가능한 수준이면 충분
- 평가 기준: 의미 전달 가능하면 7점 이상, 기본 문법만 맞아도 긍정적 평가""",
        "intermediate": """INTERMEDIATE 기준:
- 문법 (30%): 다양한 시제, 관사, 전치사 정확도
- 어휘 (25%): 상황에 맞는 어휘 사용, 자연스러운 표현
- 유창성 (25%): 자연스러운 흐름, 맥락 적절성
- 발음 (20%): 명확하고 자신감 있는 발음
- 평가 기준: 원활한 소통 가능하면 7점 이상""",
        "advanced": """ADVANCED 기준:
- 문법 (30%): 완벽한 문법, 복잡한 구조, 미묘한 뉘앙스
- 어휘 (25%): 전문 용어 정확한 사용, 관용구, 세련된 표현
- 유창성 (25%): 원어민 수준의 자연스러움, 전략적 커뮤니케이션
- 발음 (20%): 원어민에 가까운 억양과 리듬
- 평가 기준: 원어민 수준 요구, 9-10점은 전문가 수준만 가능"""
    }

    # 비즈니스 카테고리 키워드 (이 키워드가 포함되면 비즈니스 피드백 제공)
    BUSINESS_CATEGORIES = ["business", "meeting", "negotiation", "presentation", "interview", "conference", "corporate"]

    # suggestions에서 자주 튀어나오는 '뭉뚱그린 느낌' 표현(금지)
    FORBIDDEN_SUGGESTION_PHRASES = [
        "딱딱", "자연스럽", "처럼 들", "처럼 느껴", "부드럽게", "흐름을 부드럽게",
        "부담", "어색하게 들", "느낌"
    ]

    # 카테고리별 개선 제안 가이드
    SUGGESTION_GUIDE = {
        "business": """3. 개선 제안 (suggestions): 원어민 비즈니스 커뮤니케이션 평가자로서 '뉘앙스/책임 추궁 리스크/협업 요청/다음 조치 합의' 관점으로만 코칭하세요.

출력 강제 규칙(중요):
- suggestions는 반드시 배열 길이 1로 반환
- suggestions[0]는 반드시 아래 '3문장 구조'를 그대로 따르세요(문장 수 고정)

(3문장 구조)
1) "원문은 끝의 “<문제 구절(원문에서 그대로 인용)>”이(가) 원어민 비즈니스 맥락에서 <구체 리스크: 책임 추궁/지식 테스트/압박/명령 등>로 해석될 수 있고, <문제 1개: 앞 문장과의 연결/정보 부족/요청 방식 등>가 있습니다."
2) "따라서 “<대체 전략: 함께 확인/검토/정리/합의 요청>”의 협업 요청 형태로 바꿔 말해보는 것이 좋습니다."
3) "“<개선 영어 문장 1개>”처럼 말해보세요."

제약:
- 영어 예시는 정확히 1문장만 허용
- 문법 교정은 grammar_corrections로만(여기에 문법 교정 나열 금지)
- 아래 표현(느낌만 서술)은 금지: '딱딱하게', '자연스럽게', '~처럼 들려요/느껴져요'""",

        "daily": """3. 개선 제안 (suggestions): 원어민 일상 대화 관점에서 '과도한 격식/직설/불필요한 압박/부자연스런 연결'을 구체적으로 짚고 더 편한 표현으로 유도하세요.

출력 강제 규칙(중요):
- suggestions는 반드시 배열 길이 1로 반환
- suggestions[0]는 반드시 아래 '3문장 구조'(문장 수 고정)

(3문장 구조)
1) "원문은 “<문제 구절(원문에서 그대로 인용)>”이(가) 일상 대화에서 <구체 이유: 과하게 격식/직설/부담 주는 질문 형태 등>로 처리될 수 있고, <문제 1개: 연결/맥락/정보 부족 등>가 있습니다."
2) "따라서 <전략: 완곡하게 묻기/가볍게 제안하기/상대 선택권 주기> 형태로 바꿔 말해보는 것이 좋습니다."
3) "“<개선 영어 문장 1개>”처럼 말해보세요."

제약:
- 영어 예시는 정확히 1문장만 허용
- 아래 표현(느낌만 서술)은 금지: '딱딱하게', '자연스럽게', '~처럼 들려요/느껴져요'"""
    }

    async def process(
        self,
        scenario_context: Dict[str, Any],
        user_message: str,
        detected_terms: List[str],
        missed_terms: List[str],
        current_step: Optional[Dict[str, Any]] = None,
        pronunciation_details: Optional[Dict[str, Any]] = None,
        previously_used_terms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        사용자 메시지에 대한 피드백을 생성합니다.
        """
        self._validate_input(scenario_context, user_message)

        difficulty = scenario_context.get("difficulty", "intermediate")
        system_prompt = self._build_system_prompt(scenario_context, difficulty, current_step)
        user_prompt = self._build_user_prompt(
            user_message, detected_terms, missed_terms,
            current_step, pronunciation_details,
            previously_used_terms or []
        )

        logger.info(f"Generating feedback for scenario: {scenario_context.get('title', 'Unknown')}")

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # ✅ 무난한 완곡화로 흐르는 걸 줄이기 위해 낮춤
                max_tokens=900,
                response_format={"type": "json_object"}
            )

            feedback = json.loads(response.choices[0].message.content)

            # ✅ 발음 상세 정보 추가
            if pronunciation_details:
                feedback['pronunciation_details'] = pronunciation_details

            # ✅ suggestions 포맷 강제 + 금지표현 방지(필요 시 suggestions만 1회 리라이트)
            feedback = await self._normalize_and_fix_suggestions(
                scenario_context=scenario_context,
                user_message=user_message,
                system_prompt=system_prompt,
                feedback=feedback
            )

            return feedback

        except Exception as e:
            logger.error(f"Error generating feedback: {str(e)}")
            raise

    def _validate_input(
        self,
        scenario_context: Dict[str, Any],
        user_message: str
    ) -> None:
        """입력값 검증"""
        required_fields = ["title", "roles", "language", "difficulty"]
        for field in required_fields:
            if field not in scenario_context:
                raise ValueError(f"Missing required field in scenario_context: {field}")

        if not user_message:
            raise ValueError("user_message is required")

    def _is_business_scenario(self, category: str) -> bool:
        """카테고리가 비즈니스 관련인지 확인"""
        if not category:
            return False
        category_lower = category.lower()
        return any(biz_cat in category_lower for biz_cat in self.BUSINESS_CATEGORIES)

    def _suggestion_needs_fix(self, suggestion: str) -> bool:
        """suggestions[0]가 금지표현/형식 위반 가능성이 있는지 검사"""
        if not suggestion:
            return True
        # 금지표현 포함 여부
        if any(p in suggestion for p in self.FORBIDDEN_SUGGESTION_PHRASES):
            return True
        # 3문장 구조 유사 체크: 마침표(또는 문장 종결) 2개 이상 권장
        # (완벽한 문장 분리까지는 강제하지 않고, 최소 안전장치로 둠)
        if suggestion.count("습니다.") < 2 and suggestion.count("있습니다.") < 1:
            return True
        # 영어 예시가 1개인지 대략 체크(따옴표 “ ” 안의 영어가 여러 개면 위험)
        if suggestion.count("“") >= 2 and suggestion.count("”") >= 2:
            # 여러 인용일 수 있으니 위험으로 간주
            return True
        return False

    async def _rewrite_suggestion_once(
        self,
        scenario_context: Dict[str, Any],
        user_message: str,
        system_prompt: str,
        bad_suggestion: str
    ) -> str:
        """suggestions[0]만 규칙에 맞게 1회 리라이트"""
        category = scenario_context.get("category", "")
        is_business = self._is_business_scenario(category)
        scenario_type = "비즈니스" if is_business else "일상"

        rewrite_prompt = f"""
아래는 사용자 메시지에 대한 suggestions 문장입니다. 규칙에 맞게 suggestions[0]만 다시 작성하세요.

[상황]
- 시나리오 타입: {scenario_type}
- 사용자 메시지: "{user_message}"

[반드시 지켜야 할 형식(3문장 구조, 문장 수 고정)]
1) "원문은 ...입니다."
2) "따라서 ... 것이 좋습니다."
3) "“<개선 영어 문장 1개>”처럼 말해보세요."

[제약]
- 영어 예시는 정확히 1문장만
- 금지 표현: {", ".join(self.FORBIDDEN_SUGGESTION_PHRASES)}
- 느낌만 말하지 말고, 문제 구절을 “ ”로 인용하고 '왜 그런 해석이 가능한지'를 구체적으로 설명
- 결과는 JSON으로만: {{ "suggestion": "..." }}

[현재 suggestions(문제 버전)]
{bad_suggestion}
"""
        resp = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": rewrite_prompt}
            ],
            temperature=0.0,
            max_tokens=350,
            response_format={"type": "json_object"}
        )
        obj = json.loads(resp.choices[0].message.content)
        return obj.get("suggestion", bad_suggestion)

    async def _normalize_and_fix_suggestions(
        self,
        scenario_context: Dict[str, Any],
        user_message: str,
        system_prompt: str,
        feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """suggestions를 배열 길이 1로 정규화하고, 필요 시 1회 리라이트"""
        suggestions = feedback.get("suggestions", [])
        if not isinstance(suggestions, list):
            suggestions = [str(suggestions)]

        # 배열 길이 1로 강제
        if len(suggestions) == 0:
            suggestions = [""]  # 리라이트 트리거
        elif len(suggestions) > 1:
            suggestions = [suggestions[0]]

        s0 = suggestions[0] if suggestions else ""
        if self._suggestion_needs_fix(s0):
            fixed = await self._rewrite_suggestion_once(
                scenario_context=scenario_context,
                user_message=user_message,
                system_prompt=system_prompt,
                bad_suggestion=s0
            )
            suggestions = [fixed]

        feedback["suggestions"] = suggestions
        return feedback

    def _build_system_prompt(
        self,
        scenario_context: Dict[str, Any],
        difficulty: str,
        current_step: Optional[Dict[str, Any]]
    ) -> str:
        """시스템 프롬프트 구성"""
        difficulty_criteria = self.DIFFICULTY_CRITERIA.get(
            difficulty, self.DIFFICULTY_CRITERIA["intermediate"]
        )

        terminology = scenario_context.get("required_terminology", [])
        terminology_str = ", ".join(terminology) if terminology else "없음"

        # 카테고리에 따른 피드백 스타일 결정
        category = scenario_context.get("category", "")
        is_business = self._is_business_scenario(category)
        suggestion_guide = self.SUGGESTION_GUIDE["business"] if is_business else self.SUGGESTION_GUIDE["daily"]
        scenario_type = "비즈니스 회화" if is_business else "일상 회화"

        # 현재 단계 정보
        step_info_section = ""
        if current_step:
            step_terminology = current_step.get("terminology", [])
            step_info_section = f"""

현재 대화 단계:
- 단계명: {current_step.get('name', 'N/A')}
- 단계 제목: {current_step.get('title', 'N/A')}
- 가이드: {current_step.get('guide', 'N/A')}
- 이 단계에서 사용해야 할 표현: {', '.join(step_terminology) if step_terminology else '없음'}"""

        return f"""당신은 {scenario_type} 연습에 대한 피드백을 한글로 제공하는 전문 언어 튜터입니다.

시나리오 맥락:
- 제목: {scenario_context.get('title', 'N/A')}
- 설명: {scenario_context.get('description', 'N/A')}
- 상황: {scenario_context.get('scenario_text', 'N/A')}
- 카테고리: {category or 'N/A'} ({'비즈니스' if is_business else '일상'} 시나리오)
- 사용자 역할: {scenario_context.get('roles', {}).get('user', 'User')}
- AI 역할: {scenario_context.get('roles', {}).get('ai', 'AI')}
- 언어: {scenario_context.get('language', 'en')}
- 난이도: {difficulty}
- 필수 전문용어: {terminology_str}{step_info_section}

중요한 피드백 규칙:
1. 모든 피드백은 반드시 한글로 작성해야 합니다
2. 문법 교정: 한글로 문제를 설명한 후, 영어 교정을 제안하세요
   - 예시: "시제가 틀렸어요. 'I go yesterday' 대신 'I went yesterday'라고 해야 해요."
{suggestion_guide}
4. 난이도별 채점 기준 (1-10):

**{difficulty.upper()} 난이도 기준:**

{difficulty_criteria}

점수 가이드:
   - 9-10: 탁월함, 해당 난이도에서 최고 수준
   - 7-8: 좋음, 해당 난이도 목표 달성
   - 5-6: 보통, 개선 필요
   - 3-4: 부족함, 주요 개선 필요
   - 1-2: 매우 부족함, 기본부터 다시

한글 텍스트로 JSON 형식의 피드백을 제공하세요."""

    def _build_user_prompt(
        self,
        user_message: str,
        detected_terms: List[str],
        missed_terms: List[str],
        current_step: Optional[Dict[str, Any]],
        pronunciation_details: Optional[Dict[str, Any]],
        previously_used_terms: List[str] = None
    ) -> str:
        """사용자 프롬프트 구성"""
        previously_used_terms = previously_used_terms or []

        # 이전에 사용한 용어 섹션
        previously_used_section = ""
        if previously_used_terms:
            previously_used_section = f"""

⭐ 이전 대화에서 이미 사용한 용어 (이 세션에서 이미 체크됨):
- {', '.join(previously_used_terms)}
- 이 용어들은 이미 사용 완료된 것으로 간주하세요!
- 현재 메시지에서 다시 안 써도 "missed"가 아닙니다!
"""

        # Step terminology 분석
        step_terminology = []
        step_terminology_section = ""
        if current_step:
            step_terminology = current_step.get("terminology", [])
            if step_terminology:
                step_terminology_section = f"""

현재 단계 표현 분석:
- 이 단계에서 권장하는 표현: {', '.join(step_terminology)}
- 사용자가 이 표현들 중 하나를 사용했거나 의미적으로 비슷한 표현을 썼다면 긍정적으로 평가해주세요
- 완전히 동일한 표현이 아니어도, 의미와 의도가 비슷하면 사용한 것으로 인정합니다
- 예: "I'd like to discuss" 권장 표현에 대해 "Can we talk about" 사용 -> 의미적으로 유사하므로 인정"""

        # 발음 정보
        pronunciation_info = ""
        if pronunciation_details:
            problem_words = [
                f"- '{word['word']}': {word['accuracy_score']:.1f}/100"
                for word in pronunciation_details.get('words', [])
                if word.get('accuracy_score', 100) < 80
            ][:5]

            pronunciation_info = f"""

Azure 발음 평가 결과:
- 전체 발음 점수: {pronunciation_details.get('pronunciation_score', 0):.1f}/100
- 정확도 점수: {pronunciation_details.get('accuracy_score', 0):.1f}/100
- 유창성 점수: {pronunciation_details.get('fluency_score', 0):.1f}/100
- 운율 점수 (억양/강세): {pronunciation_details.get('prosody_score', 0):.1f}/100
- 완성도 점수: {pronunciation_details.get('completeness_score', 0):.1f}/100

발음 문제가 있는 단어들 (정확도 < 80):
{chr(10).join(problem_words) if problem_words else '(모든 단어가 잘 발음되었습니다)'}

이 점수를 기반으로 다음에 대한 구체적인 피드백을 제공하세요:
1. 운율 (Prosody): prosody_score < 80인 경우, 억양(intonation), 강세(stress), 또는 리듬(rhythm) 문제를 설명하세요
2. 문제 단어: 낮은 정확도 점수를 받은 특정 단어를 언급하세요
3. 전반적인 발음 개선 팁"""

        required_terms = detected_terms + missed_terms

        return f"""사용자 메시지: "{user_message}"

전문용어 분석 (의미적 유사성 기반으로 재평가 필요):
- 필수 전문용어 전체: {', '.join(required_terms) if required_terms else '없음'}
- 현재 메시지에서 시스템 감지 (정확히 일치만): 사용={', '.join(detected_terms) if detected_terms else '없음'}
- 아직 미사용 (이전+현재 합쳐서): {', '.join(missed_terms) if missed_terms else '없음'}
{previously_used_section}
- 중요: 사용자가 정확히 같은 표현이 아니더라도 **의미적으로 유사한 표현**을 사용했으면 "사용함"으로 판단하세요

  유사 표현 예시:
  - "I'd like to discuss" ≈ "Can we talk about" ≈ "Let's go over"
  - "Could you please" ≈ "Would you mind" ≈ "Can you"
  - "I apologize for" ≈ "Sorry for" ≈ "I'm sorry about"

  약어/축약형도 동일하게 취급:
  - "technical stack" = "tech stack" (동일!)
  - "application" = "app"
  - "information" = "info"
  - "specification" = "spec"
  - "development" = "dev"
  - "production" = "prod"
  - "repository" = "repo"
  - "documentation" = "docs"

  단어 순서나 관사 차이도 무시:
  - "project timeline" ≈ "the timeline of project" ≈ "timeline for the project"
  - "feature review" ≈ "review of features" ≈ "reviewing features"{step_terminology_section}
{pronunciation_info}

다음의 정확한 JSON 형식으로 피드백을 제공하세요 (모든 텍스트 한글로):
{{
  "grammar_corrections": [
    "<실제 문법 오류가 있으면 여기에 작성. 오류가 없으면 빈 배열 []>"
  ],
  "terminology_usage": {{
    "used": ["<현재 메시지에서 사용한 용어 (의미적 유사성 기반)>"],
    "previously_used": ["<이전 메시지들에서 이미 사용 완료된 용어>"],
    "missed": ["<아직 한 번도 사용하지 않은 용어들>"],
    "similar_expressions": {{
      "<필수용어>": "<사용자가 사용한 유사 표현 (있다면)>"
    }},
    "feedback": "필수 용어 사용에 대한 피드백 (이전에 쓴 건 다시 언급할 필요 없음)",
    "step_expression": {{
      "recommended": {json.dumps(step_terminology, ensure_ascii=False)},
      "used_similar": "<사용자가 권장 표현과 유사한 표현을 사용했는지 여부 (true/false)>",
      "user_expression": "<사용자가 사용한 유사 표현 (있다면)>",
      "feedback": "<권장 표현 사용에 대한 피드백. 유사 표현 썼으면 칭찬, 안 썼으면 다음에 써보라고 권유>"
    }}
  }},
  "suggestions": [
    "원문은 끝의 “<문제 구절>”이(가) <해석 리스크/문제점>가 있습니다. 따라서 <협업 요청/완곡/구체화> 형태로 바꿔 말해보는 것이 좋습니다. “<개선 영어 문장 1개>”처럼 말해보세요."
  ],
  "pronunciation_feedback": [
    "<발음 평가 데이터 기반 실제 피드백>"
  ],
  "score": 7,
  "score_breakdown": {{
    "grammar": 6,
    "vocabulary": 8,
    "fluency": 7,
    "pronunciation": 7
  }}
}}

중요한 규칙:
- grammar_corrections: 사용자 메시지에 **실제로 존재하는** 문법 오류만 지적하세요. 오류가 없으면 빈 배열 []을 반환하세요.
- 예시나 템플릿 문구를 그대로 복사하지 마세요. 오직 사용자 메시지 분석 결과만 작성하세요.
- 문법적으로 완벽한 문장에 대해 거짓 오류를 만들어내지 마세요.
- suggestions: 반드시 배열 길이 1로 반환하고, 3문장 구조(문장 수 고정)를 지키세요.
- suggestions에는 '{", ".join(self.FORBIDDEN_SUGGESTION_PHRASES)}' 같은 뭉뚱그린 느낌 표현을 쓰지 마세요(구체적 이유로 설명).

terminology_usage 규칙 (핵심!):
- used: **현재 메시지**에서 사용한 용어 (의미적 유사성 기반)
- previously_used: **이전 메시지들**에서 이미 사용 완료된 용어 (시스템이 전달한 리스트 그대로 사용)
- missed: 아직 **한 번도** 사용하지 않은 용어 (previously_used에 있으면 missed 아님!)
- 시스템 감지 결과는 참고용일 뿐, 정확히 일치하는 것만 체크한 것입니다
- 사용자가 "I'd like to discuss"를 써야 하는데 "Can we talk about"을 썼다면 → used에 포함!
- similar_expressions: 유사 표현을 사용한 경우, 어떤 표현을 썼는지 기록 (예: {{"I'd like to discuss": "Can we talk about"}})
- feedback: 이전에 이미 쓴 용어는 언급하지 마세요. 현재 메시지에서 새로 사용한 것만 칭찬하세요.

기타 규칙:
- terminology_usage.step_expression: 현재 단계 권장 표현 사용 여부를 평가하세요 (의미적 유사성 기준)
- 발음 평가 데이터가 제공되면, 구체적인 팁과 함께 "pronunciation_feedback" 배열을 반드시 포함해야 합니다
- prosody_score < 80인 경우: 억양(intonation), 강세(stress), 또는 리듬(rhythm)에 대한 피드백을 제공하세요
- 낮은 정확도를 가진 단어가 있다면: 해당 특정 단어와 개선 방법을 언급하세요
- 모든 설명은 한글로 작성하되, 한글 텍스트 안에 영어 단어/교정을 포함하세요"""
