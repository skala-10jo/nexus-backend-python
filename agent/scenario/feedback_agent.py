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
- 어휘 (25%): 비즈니스 용어 적절한 사용, 자연스러운 표현
- 유창성 (25%): 자연스러운 흐름, 맥락 적절성, 비즈니스 예절
- 발음 (20%): 명확하고 자신감 있는 발음
- 평가 기준: 일반적인 비즈니스 소통 가능하면 7점 이상""",
        "advanced": """ADVANCED 기준:
- 문법 (30%): 완벽한 문법, 복잡한 구조, 미묘한 뉘앙스
- 어휘 (25%): 전문 용어 정확한 사용, 관용구, 세련된 표현
- 유창성 (25%): 원어민 수준의 자연스러움, 전략적 커뮤니케이션
- 발음 (20%): 원어민에 가까운 억양과 리듬
- 평가 기준: 원어민 수준 요구, 9-10점은 전문가 수준만 가능"""
    }

    async def process(
        self,
        scenario_context: Dict[str, Any],
        user_message: str,
        detected_terms: List[str],
        missed_terms: List[str],
        current_step: Optional[Dict[str, Any]] = None,
        pronunciation_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        사용자 메시지에 대한 피드백을 생성합니다.

        Args:
            scenario_context: 시나리오 정보
                - title, description, scenario_text
                - roles: {"ai": str, "user": str}
                - language, difficulty
                - required_terminology
            user_message: 사용자 메시지
            detected_terms: 감지된 전문용어 리스트
            missed_terms: 미사용 전문용어 리스트
            current_step: 현재 대화 단계 정보 (선택)
                - name, title, guide, terminology
            pronunciation_details: 발음 평가 결과 (선택)
                - pronunciation_score, accuracy_score, fluency_score
                - prosody_score, completeness_score, words

        Returns:
            피드백 딕셔너리
        """
        self._validate_input(scenario_context, user_message)

        difficulty = scenario_context.get("difficulty", "intermediate")
        system_prompt = self._build_system_prompt(scenario_context, difficulty, current_step)
        user_prompt = self._build_user_prompt(
            user_message, detected_terms, missed_terms,
            current_step, pronunciation_details
        )

        logger.info(f"Generating feedback for scenario: {scenario_context.get('title', 'Unknown')}")

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=800,
                response_format={"type": "json_object"}
            )

            feedback = json.loads(response.choices[0].message.content)

            # 발음 상세 정보 추가
            if pronunciation_details:
                feedback['pronunciation_details'] = pronunciation_details

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

        return f"""당신은 비즈니스 회화 연습에 대한 피드백을 한글로 제공하는 전문 언어 튜터입니다.

시나리오 맥락:
- 제목: {scenario_context.get('title', 'N/A')}
- 설명: {scenario_context.get('description', 'N/A')}
- 상황: {scenario_context.get('scenario_text', 'N/A')}
- 사용자 역할: {scenario_context.get('roles', {}).get('user', 'User')}
- AI 역할: {scenario_context.get('roles', {}).get('ai', 'AI')}
- 언어: {scenario_context.get('language', 'en')}
- 난이도: {difficulty}
- 필수 전문용어: {terminology_str}{step_info_section}

중요한 피드백 규칙:
1. 모든 피드백은 반드시 한글로 작성해야 합니다
2. 문법 교정: 한글로 문제를 설명한 후, 영어 교정을 제안하세요
   - 예시: "시제가 틀렸어요. 'I go yesterday' 대신 'I went yesterday'라고 해야 해요."
3. 제안: 한글 설명과 함께 영어 표현 추천을 제공하세요
   - 예시: "더 자연스러운 표현으로는 'Could you please...' 또는 'Would you mind...'를 사용해보세요."
   - 메시지가 매우 부족하다면, "이런 식으로 해보세요"와 함께 완전한 문장 예시를 제공하세요
   - 제안할 때 사용자의 역할과 상황을 고려하세요 (예: 격식, 어조, 맥락 적절성)
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
        pronunciation_details: Optional[Dict[str, Any]]
    ) -> str:
        """사용자 프롬프트 구성"""
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

전문용어 분석:
- 필수 전문용어: {', '.join(required_terms) if required_terms else '없음'}
- 사용한 용어: {', '.join(detected_terms) if detected_terms else '없음'}
- 미사용 용어: {', '.join(missed_terms) if missed_terms else '없음'}{step_terminology_section}
{pronunciation_info}

다음의 정확한 JSON 형식으로 피드백을 제공하세요 (모든 텍스트 한글로):
{{
  "grammar_corrections": [
    "<실제 문법 오류가 있으면 여기에 작성. 오류가 없으면 빈 배열 []>"
  ],
  "terminology_usage": {{
    "used": {json.dumps(detected_terms or [], ensure_ascii=False)},
    "missed": {json.dumps(missed_terms, ensure_ascii=False)},
    "feedback": "필수 용어 사용에 대한 피드백을 여기에 작성하세요",
    "step_expression": {{
      "recommended": {json.dumps(step_terminology, ensure_ascii=False)},
      "used_similar": "<사용자가 권장 표현과 유사한 표현을 사용했는지 여부 (true/false)>",
      "user_expression": "<사용자가 사용한 유사 표현 (있다면)>",
      "feedback": "<권장 표현 사용에 대한 피드백. 유사 표현 썼으면 칭찬, 안 썼으면 다음에 써보라고 권유>"
    }}
  }},
  "suggestions": [
    "<실제 개선 제안이 있으면 여기에 작성>"
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

중요:
- terminology_usage의 used와 missed 배열은 위에서 제공한 값을 그대로 사용하세요
- terminology_usage.feedback에는 용어 사용에 대한 구체적인 피드백을 한글로 작성하세요
- terminology_usage.step_expression: 현재 단계 권장 표현 사용 여부를 평가하세요 (의미적 유사성 기준)
- 발음 평가 데이터가 제공되면, 구체적인 팁과 함께 "pronunciation_feedback" 배열을 반드시 포함해야 합니다
- prosody_score < 80인 경우: 억양(intonation), 강세(stress), 또는 리듬(rhythm)에 대한 피드백을 제공하세요
- 낮은 정확도를 가진 단어가 있다면: 해당 특정 단어와 개선 방법을 언급하세요
- 모든 설명은 한글로 작성하되, 한글 텍스트 안에 영어 단어/교정을 포함하세요"""
