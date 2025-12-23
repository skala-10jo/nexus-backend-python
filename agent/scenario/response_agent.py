"""
시나리오 응답 생성 Agent.

시나리오 대화에서 AI 역할의 응답을 생성합니다.
- 초기 인사 메시지 생성 (mode="initial")
- 대화 중 응답 생성 (mode="conversation")
"""
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ResponseAgent(BaseAgent):
    """
    시나리오 AI 응답 생성 Agent.

    책임: 시나리오 컨텍스트와 대화 히스토리를 기반으로 AI 응답 생성.

    두 가지 모드 지원:
        - initial: 대화 시작 시 초기 인사 메시지 생성
        - conversation: 사용자 메시지에 대한 응답 생성

    Returns:
        - initial 모드: str (인사 메시지)
        - conversation 모드: Dict[str, Any] {"message": str, "step_completed": bool}
    """

    DIFFICULTY_INSTRUCTIONS = {
        "beginner": """
- 매우 간단하고 기본적인 인사를 사용하세요
- 짧고 쉬운 문장 구조 사용 (5-8 단어)
- 일상적이고 친근한 표현만 사용
- 복잡한 어휘나 관용구 피하기""",
        "intermediate": """
- 자연스러운 비즈니스 인사 사용
- 중간 길이의 문장 (8-12 단어)
- 일반적인 비즈니스 용어 포함 가능
- 약간의 관용적 표현 사용 가능""",
        "advanced": """
- 전문적이고 세련된 비즈니스 인사
- 다양한 문장 구조 사용 가능
- 전문 용어와 관용구 자유롭게 사용
- 뉘앙스와 함축적 표현 활용"""
    }

    CONVERSATION_STYLE = {
        "beginner": """
- 매우 간단한 문장 구조 사용 (주어 + 동사 + 목적어)
- 기본 어휘만 사용 (고등학교 수준)
- 천천히 주제 전환하기
- 한 번에 한 가지 아이디어만 다루기
- 명확하고 직접적인 질문하기""",
        "intermediate": """
- 자연스러운 비즈니스 대화 스타일
- 일반적인 비즈니스 용어 사용
- 복합 문장 가능하지만 간결하게
- 적절한 관용구 사용
- 맥락을 고려한 질문과 응답""",
        "advanced": """
- 전문적이고 세련된 비즈니스 커뮤니케이션
- 전문 용어와 산업 특화 어휘 자유롭게 사용
- 복잡한 문장 구조와 뉘앙스 활용
- 함축적 표현과 고급 관용구 사용
- 전략적이고 다층적인 대화 진행"""
    }

    async def process(
        self,
        scenario_context: Dict[str, Any],
        mode: str = "conversation",
        user_message: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        current_step: Optional[Dict[str, Any]] = None,
        current_step_index: int = 0,
        total_steps: int = 0
    ) -> Any:
        """
        AI 응답을 생성합니다.

        Args:
            scenario_context: 시나리오 정보
                - title: 시나리오 제목
                - description: 시나리오 설명
                - scenario_text: 시나리오 상황 텍스트
                - roles: {"ai": str, "user": str}
                - language: 언어 코드 (en, ko 등)
                - difficulty: beginner/intermediate/advanced
                - required_terminology: 필수 전문용어 리스트
            mode: "initial" (초기 인사) 또는 "conversation" (대화 응답)
            user_message: 사용자 메시지 (conversation 모드에서 필수)
            conversation_history: 대화 히스토리 (conversation 모드에서 필수)
            current_step: 현재 스텝 정보 (선택)
            current_step_index: 현재 스텝 인덱스
            total_steps: 전체 스텝 수

        Returns:
            - initial 모드: str (인사 메시지)
            - conversation 모드: Dict {"message": str, "step_completed": bool}
        """
        self._validate_input(scenario_context, mode, user_message, conversation_history)

        if mode == "initial":
            return await self._generate_initial_message(scenario_context)
        else:
            return await self._generate_conversation_response(
                scenario_context,
                user_message,
                conversation_history,
                current_step,
                current_step_index,
                total_steps
            )

    def _validate_input(
        self,
        scenario_context: Dict[str, Any],
        mode: str,
        user_message: Optional[str],
        conversation_history: Optional[List[Dict[str, str]]]
    ) -> None:
        """입력값 검증"""
        required_fields = ["title", "roles", "language", "difficulty"]
        for field in required_fields:
            if field not in scenario_context:
                raise ValueError(f"Missing required field in scenario_context: {field}")

        if mode not in ("initial", "conversation"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'initial' or 'conversation'")

        if mode == "conversation":
            if user_message is None:
                raise ValueError("user_message is required for conversation mode")
            if conversation_history is None:
                raise ValueError("conversation_history is required for conversation mode")

    async def _generate_initial_message(self, scenario_context: Dict[str, Any]) -> str:
        """
        초기 AI 인사 메시지 생성.

        Returns:
            초기 인사 메시지
        """
        current_date = datetime.now().strftime("%Y-%m-%d")
        difficulty = scenario_context.get("difficulty", "intermediate")

        system_prompt = self._build_initial_system_prompt(scenario_context, current_date, difficulty)
        user_prompt = "친근한 인사로 시작하세요 (1-2문장). 첫 문장: 캐주얼한 인사. 두 번째 문장 (선택): 시나리오 맥락을 은근히 언급. 간결하고 자연스럽게 작성하세요."

        logger.info(f"Generating initial message for scenario: {scenario_context.get('title', 'Unknown')}")

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=80
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating initial message: {str(e)}")
            return "Hello! How are you doing today?"

    async def _generate_conversation_response(
        self,
        scenario_context: Dict[str, Any],
        user_message: str,
        conversation_history: List[Dict[str, str]],
        current_step: Optional[Dict[str, Any]],
        current_step_index: int,
        total_steps: int
    ) -> Dict[str, Any]:
        """
        대화 중 AI 응답 생성.

        Returns:
            {"message": str, "step_completed": bool}
        """
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M")
        difficulty = scenario_context.get("difficulty", "intermediate")

        system_prompt = self._build_conversation_system_prompt(
            scenario_context, current_date, current_time, difficulty,
            current_step, current_step_index, total_steps
        )

        # 대화 히스토리 구성
        messages = [{"role": "system", "content": system_prompt}]

        # 이전 대화 추가 (최근 10개, 빈 메시지 필터링)
        for msg in conversation_history[-10:]:
            content = msg.get("message", "").strip()
            if not content:
                logger.warning("Skipping empty message in conversation history")
                continue
            role = "assistant" if msg["speaker"] == "ai" else "user"
            messages.append({"role": role, "content": content})

        # 현재 사용자 메시지 추가
        messages.append({"role": "user", "content": user_message})

        logger.info(f"Generating conversation response for scenario: {scenario_context.get('title', 'Unknown')}")

        # 재시도 로직 (최대 2회)
        max_retries = 2
        last_error = None

        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.7,  # 안정적인 JSON 응답을 위해 낮춤
                    max_tokens=200,   # 충분한 토큰 확보
                    response_format={"type": "json_object"}
                )

                response_text = response.choices[0].message.content

                # 디버깅용 상세 로그
                logger.info(f"GPT raw response (attempt {attempt + 1}): {response_text}")

                if not response_text or not response_text.strip():
                    logger.warning(f"GPT returned empty content (attempt {attempt + 1}): repr={repr(response_text)}")
                    last_error = ValueError("GPT returned empty response")
                    continue  # 재시도

                try:
                    parsed_response = json.loads(response_text)
                    message = parsed_response.get("message", "").strip()
                    step_completed = parsed_response.get("step_completed", False)

                    logger.info(f"Parsed response - message length: {len(message)}, step_completed: {step_completed}")

                    if not message:
                        logger.warning(f"GPT returned empty message (attempt {attempt + 1}). Full response: {response_text}")
                        last_error = ValueError(f"GPT returned empty message")
                        continue  # 재시도

                    return {
                        "message": message,
                        "step_completed": step_completed
                    }
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON (attempt {attempt + 1}): {response_text}, error: {e}")
                    last_error = ValueError(f"Invalid JSON from GPT: {response_text}")
                    continue  # 재시도

            except Exception as e:
                logger.error(f"Error in GPT call (attempt {attempt + 1}): {str(e)}")
                last_error = e
                continue  # 재시도

        # 모든 재시도 실패
        logger.error(f"All {max_retries} attempts failed. Last error: {last_error}")
        raise last_error if last_error else ValueError("Unknown error in conversation response")

    def _build_initial_system_prompt(
        self,
        scenario_context: Dict[str, Any],
        current_date: str,
        difficulty: str
    ) -> str:
        """초기 인사 메시지용 시스템 프롬프트 구성"""
        difficulty_instruction = self.DIFFICULTY_INSTRUCTIONS.get(
            difficulty, self.DIFFICULTY_INSTRUCTIONS["intermediate"]
        )

        terminology = scenario_context.get("required_terminology", [])
        terminology_str = ", ".join(terminology) if terminology else "없음"

        return f"""당신은 비즈니스 회화 연습 시나리오에 참여하고 있습니다.

오늘 날짜: {current_date}

시나리오: {scenario_context.get('title', 'N/A')}
설명: {scenario_context.get('description', 'N/A')}
상황: {scenario_context.get('scenario_text', 'N/A')}

당신의 역할: {scenario_context.get('roles', {}).get('ai', 'AI')}
사용자 역할: {scenario_context.get('roles', {}).get('user', 'User')}

언어: {scenario_context.get('language', 'en')}
난이도: {difficulty}

나중에 자연스럽게 사용할 필수 전문용어: {terminology_str}

난이도별 지침:
{difficulty_instruction}

기본 지침:
- 짧고 친근한 인사로 시작하세요 (최대 1-2문장)
- 캐주얼한 인사와 함께 시나리오 맥락을 은근히 암시하세요
- 예시: "안녕하세요! 잘 지내셨어요? 프로젝트 미팅 준비되셨나요?" 또는 "Hi! How's it going? Ready for our meeting about the project?"
- 자연스럽고 대화체로 작성하세요
- {scenario_context.get('language', 'en')} 언어로 응답하세요
- 친근하고 환영하는 분위기를 만드세요
- 오늘 날짜를 기반으로 현실적인 맥락을 사용하세요"""

    def _build_conversation_system_prompt(
        self,
        scenario_context: Dict[str, Any],
        current_date: str,
        current_time: str,
        difficulty: str,
        current_step: Optional[Dict[str, Any]],
        current_step_index: int,
        total_steps: int
    ) -> str:
        """대화 응답용 시스템 프롬프트 구성"""
        conversation_style = self.CONVERSATION_STYLE.get(
            difficulty, self.CONVERSATION_STYLE["intermediate"]
        )

        terminology = scenario_context.get("required_terminology", [])
        terminology_str = ", ".join(terminology) if terminology else "없음"

        # 스텝 정보 구성
        step_context = ""
        step_judgment_instruction = ""
        if current_step and total_steps > 0:
            step_terminology = current_step.get("terminology", [])
            step_terminology_str = ", ".join(step_terminology) if step_terminology else "없음"

            step_context = f"""
현재 진행 단계: {current_step_index + 1}/{total_steps}
현재 스텝: {current_step.get('name', 'Unknown')}
스텝 가이드: {current_step.get('guide', '')}
이 스텝에서 사용할 용어: {step_terminology_str}
"""
            step_judgment_instruction = f"""
⚠️ 스텝 완료 판단 (매우 중요!):
현재 스텝 "{current_step.get('name', 'unknown')}"의 완료 여부를 적극적으로 판단하세요.

스텝별 완료 기준:
- ice_breaking: 인사와 안부를 주고받았으면 완료 (2-3회 교환이면 충분)
- agenda_setting: 미팅 목적/주제가 언급되었으면 완료
- discussion: 주요 논의 사항을 다뤘으면 완료 (3-4회 의미 있는 교환)
- action_items: 할 일이나 다음 단계가 언급되었으면 완료
- wrap_up: 마무리 인사나 감사 표현이 있으면 완료

판단 원칙:
1. 완벽할 필요 없음 - 스텝의 핵심 목적이 어느 정도 달성되면 true
2. 자연스러운 흐름 우선 - 대화가 다음 주제로 넘어가려는 징후가 보이면 true
3. 적극적으로 판단 - 애매하면 true로 설정하여 대화 흐름을 유지
4. 절대로 "다음 단계로 넘어갑시다" 같은 명시적 언급 금지

예시:
- 사용자: "I'm doing well, thanks!" → ice_breaking 완료 (step_completed: true)
- 사용자: "Let's talk about the project timeline" → agenda_setting 완료 (step_completed: true)
- 사용자: "I think we covered the main points" → discussion 완료 (step_completed: true)
"""

        return f"""당신은 비즈니스 회화 연습 시나리오에 참여하고 있습니다.

오늘 날짜: {current_date}
현재 시간: {current_time}

시나리오: {scenario_context.get('title', 'N/A')}
설명: {scenario_context.get('description', 'N/A')}
상황: {scenario_context.get('scenario_text', 'N/A')}

당신의 역할: {scenario_context.get('roles', {}).get('ai', 'AI')}
사용자 역할: {scenario_context.get('roles', {}).get('user', 'User')}

언어: {scenario_context.get('language', 'en')}
난이도: {difficulty}

자연스럽게 사용할 필수 전문용어: {terminology_str}
{step_context}
난이도별 대화 스타일:
{conversation_style}

기본 지침:
- 중요: 응답을 매우 간결하게 유지하세요 - 읽는 시간이 7초 이내여야 합니다 (최대 15-20 단어)
- 1-2개의 짧은 문장만 사용하세요 (절대 2문장 이상 사용하지 마세요)
- 실제 대화처럼 캐주얼한 스몰토크와 비즈니스 주제를 자연스럽게 섞으세요
- 가끔은 비즈니스 대화 전후에 개인적인 것들(주말, 점심, 날씨 등)을 물어보세요
- 비즈니스를 논의할 때 필수 전문용어를 자연스럽게 사용하세요
- {scenario_context.get('language', 'en')} 언어로 응답하세요
- 언어 연습에 대해 격려하고 지원적으로 대하세요
- 사용자가 문법 오류를 범하면, 응답에서 부드럽게 교정을 포함하세요
- 오늘 맥락을 기반으로 현실적인 날짜와 시간을 사용하세요
- 알림: "빠른 채팅 메시지"로 생각하세요, "이메일"이 아닙니다 - 대화체이고 간결하게
{step_judgment_instruction}
응답은 반드시 다음 JSON 형식으로 제공하세요:
{{
    "message": "AI 응답 메시지 (간결하게, 15-20 단어 이내)",
    "step_completed": true 또는 false (위의 스텝 완료 기준에 따라 적극적으로 판단!)
}}

⚠️ step_completed 주의사항:
- 기본값을 false로 두지 마세요! 매번 적극적으로 판단하세요
- 스텝 완료 기준에 부합하면 반드시 true로 설정하세요
- 대화가 자연스럽게 다음 주제로 넘어가면 true로 설정하세요"""
