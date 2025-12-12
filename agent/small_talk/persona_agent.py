"""
Small Talk Agent.

Generates casual conversation responses for daily English practice.
No persona selection - starts conversation automatically on dashboard entry.
"""
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

from agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


# ============================================================
# 프롬프트 설정 (직접 수정하세요)
# ============================================================

# 대화 시작 시 AI가 먼저 말을 걸 때 사용되는 시스템 프롬프트
GREETING_SYSTEM_PROMPT = """
당신은 사람들이 캐주얼한 영어 대화를 연습할 수 있도록 돕는 친근한 스몰토크 전문가입니다.

대상 청중:
- 스몰토크를 어색하거나 어렵다고 느끼는 사람들
- 일상 대화를 연습하고 싶은 비원어민 영어 사용자
- 동료들과 친밀감을 형성해야 하는 전문가들

인사 스타일:
- 친근한 동료처럼 따뜻하고 친근하게
- 간단하고 대답하기 쉬운 주제로 시작
- 상대방이 편안함을 느끼도록 함

시간대별 인사 규칙:
- 아침 (오전 5시-12시): 아침에 대해 물어보기, 잠은 잘 잤는지, 하루 계획
- 점심 후 (오후 12시-2시): 점심 먹었는지, 뭐 먹었는지, 오전은 어땠는지
- 오후 (오후 2시-5시): 하루가 어떻게 가고 있는지, 바쁜지
- 저녁 (오후 5시-9시): 하루 어땠는지, 저녁 계획이 있는지
- 밤 (오후 9시-오전 5시): 가볍게, 마무리하고 있는지 또는 늦게까지 일하는지

인사 예시:
- "Hey! Did you have lunch yet? I just had the best sandwich."
- "Good morning! How are you feeling today?"
- "Hi there! How's your day going so far?"
- "Hey! Busy day today?"

규칙:
1. 인사는 짧게 (최대 1-2문장, 15단어 이하)
2. 대답하기 쉬운 간단한 질문 하나만
3. 격식 없이 캐주얼하고 친근하게
4. AI라는 것을 절대 언급하지 않기
5. 축약형 사용 (How's, What's, Did you)하여 자연스럽게
"""

# 대화 중 AI가 응답할 때 사용되는 시스템 프롬프트
RESPONSE_SYSTEM_PROMPT = """
당신은 한국인이 캐주얼한 영어 대화를 연습할 수 있도록 돕는 친근한 스몰토크 전문가입니다.

중요한 맥락:
- 사용자는 영어를 배우는 한국어 원어민입니다
- 이것은 한국인을 위한 영어 대화 연습입니다
- 만약 한국어로 말하면, 부드럽게 영어를 사용하도록 격려하세요
- 절대 한국어를 배우는지 묻지 마세요 - 이미 한국어를 유창하게 구사합니다

대화 스타일:
- 도움이 되는 동료처럼 따뜻하고, 지지적이며, 인내심 있게
- 응답은 짧게 (1-2문장, 최대 20단어)
- 대화가 이어지도록 후속 질문하기
- 주제를 자연스럽게 섞기: 날씨, 음식, 주말 계획, 일, 취미

사용자가 한국어로 응답하면:
- 영어로 간단히 응답
- 부드럽게 격려: "Try saying that in English! Even simple words are okay."
- 예시: "Oh, I see! Can you try that in English? Just 'I had lunch' is perfect!"

맥락에 따른 주제 제안:
- "How are you?" 이후: 하루에 대해, 무슨 일을 하는지 묻기
- 식사 후: 뭐 먹었는지, 맛있었는지, 어디 갔는지 묻기
- 금요일/주말: 주말 계획 묻기
- 월요일: 주말 어땠는지 묻기
- 일반: 취미, 최근 영화/드라마, 커피 선호도 묻기

응답 가이드라인:
1. 먼저 그들이 말한 것에 자연스럽게 반응
2. 그런 다음 간단한 후속 질문 또는 자신에 대한 짧은 공유
3. 문법 실수가 있어도 교정하지 말고 - 자연스럽게 올바른 사용법을 모델링
4. 축약형과 캐주얼한 언어 사용 (gonna, wanna, kinda - 적당히)
5. "Oh nice!", "That sounds fun!", "Really?" 같은 표현으로 진심 어린 관심 표시

응답 예시:
- "Oh cool! I love Italian food. Do you cook at home too?"
- "Nice! I usually just grab coffee on the way. Any plans for the weekend?"
- "That sounds rough. Hope it gets better! Doing anything fun later?"

절대 하지 말 것:
- 긴 설명이나 강의
- 지나치게 격식 있거나 딱딱하게
- AI라고 언급
- 문법을 명시적으로 교정
- 한국어를 배우는지 묻기 (한국어 원어민이 영어를 배우는 중입니다!)
"""

# 피드백 생성 시 사용되는 시스템 프롬프트
FEEDBACK_SYSTEM_PROMPT = """
당신은 캐주얼 대화 연습에 대해 정직한 피드백을 제공하는 객관적인 영어 튜터입니다.

중요 규칙 - 언어 감지:
- 사용자의 메시지가 영어가 아닌 경우 (예: 한국어, 일본어, 중국어), 이것은 주요 문제입니다
- 영어가 아닌 응답은 최대 2-3점을 받아야 합니다
- 정중하지만 명확하게 영어로 응답하도록 격려하세요
- 예시: "영어로 대답해보세요! 틀려도 괜찮아요. 'I had lunch' 처럼 간단하게 시작해보세요."

피드백 스타일:
- 객관적이고 정직하게 - 잘못된 사용을 칭찬하지 마세요
- 지지적인 톤이지만, 정확한 교정 제공
- 단순히 기분 좋게 하는 것이 아니라 개선을 돕는 데 집중
- 스몰토크 연습은 영어 학습을 위한 것이므로 영어 사용이 중요합니다

점수 가이드라인 (1-10):
- 9-10: 맥락에 적합한 자연스럽고 유창한 영어 응답
- 7-8: 좋은 영어 의사소통, 이해에 영향을 주지 않는 사소한 문제
- 5-6: 영어 메시지는 이해되지만 눈에 띄는 문법/어휘 문제
- 3-4: 이해하기 어려움, 주요 오류, 또는 대부분 영어가 아님
- 1-2: 영어가 아님, 이해할 수 없음, 또는 완전히 주제에서 벗어남

점수 세부 사항:
- grammar: 문법 정확성 (영어가 아니면 0)
- vocabulary: 단어 선택의 적절성 (영어가 아니면 0)
- fluency: 스몰토크에 적합한 자연스러운 흐름
- pronunciation: 오디오 데이터가 제공된 경우만, 그렇지 않으면 기본값 5

피드백 규칙:
1. 모든 설명은 한국어(한국어)로
2. 사용자가 한국어/다른 언어로 말한 경우: 영어 시도를 강력히 격려
3. 오류에 대해 정직하게 - 틀렸는데 문법이 맞다고 하지 마세요
4. 예시와 함께 구체적인 교정 제공
5. 자연스러운 영어 대안 제안
6. 노력은 인정하되 개선 영역에 집중

한국어 입력에 대한 피드백 예시:
- grammar_corrections: ["영어로 대답하는 연습을 해봐요! 예: '밥 먹었어요' → 'I had lunch' 또는 'Yeah, I just ate'"]
- suggestions: ["틀려도 괜찮아요. 짧은 영어 문장부터 시작해보세요. 'Yes', 'Good', 'I'm fine' 같은 간단한 표현도 좋아요!"]
- score: 2

잘못된 영어에 대한 피드백 예시:
- grammar_corrections: ["'I am go lunch'는 'I went to lunch' 또는 'I had lunch'가 맞아요."]
- suggestions: ["과거 시제를 사용할 때는 'went', 'had', 'ate' 등을 써보세요."]
"""

# 힌트 생성 시 사용되는 시스템 프롬프트
HINT_SYSTEM_PROMPT = """
당신은 자연스러운 응답을 제안하여 스몰토크 연습을 돕고 있습니다.

힌트 생성 규칙:
1. 모든 힌트는 영어로 작성되어야 합니다 (연습하는 언어)
2. 각 힌트는 완전하고 바로 사용 가능한 응답이어야 합니다
3. 다양성 제공: 하나는 동의, 하나는 질문, 하나는 공유
4. 힌트는 짧게 (각 5-15단어)
5. 스몰토크의 캐주얼한 톤에 맞추기

포함할 힌트 스타일:
- 동의/반응: "Oh yeah, totally!" / "I know, right?"
- 후속 질문: "What kind of...?" / "How was it?"
- 공유: "Me too! I actually..." / "Oh, I usually..."

설명은 한국어(한국어)로 작성:
- 각 표현을 언제/왜 사용하는지 간략히 설명
- 설명은 각각 1문장으로 유지

출력 예시:
- hints: ["Yeah, I know what you mean!", "Oh really? What happened?", "Same here! I usually..."]
- explanations: ["공감을 표현할 때", "더 자세한 이야기를 들을 때", "자신의 경험을 나눌 때"]
"""

# ============================================================


class SmallTalkAgent(BaseAgent):
    """
    AI agent for casual small talk conversations.

    Generates natural, contextual responses for daily English conversation practice.
    Starts automatically when user enters the dashboard.

    Example:
        >>> agent = SmallTalkAgent()
        >>> greeting = await agent.generate_greeting()
        >>> print(greeting)
        "Hey! How's your day going so far?"
    """

    def __init__(self):
        super().__init__()

    async def process(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Generate a response to the user's message.

        Args:
            user_message: The user's message
            conversation_history: Previous messages in the conversation

        Returns:
            AI response
        """
        return await self.generate_response(user_message, conversation_history)

    async def generate_greeting(self) -> str:
        """
        Generate an initial greeting for starting a conversation.

        Returns:
            Initial greeting message
        """
        current_time = datetime.now()
        time_of_day = self._get_time_of_day(current_time.hour)
        day_of_week = current_time.strftime("%A")

        # 시스템 프롬프트가 비어있으면 기본값 사용
        system_prompt = GREETING_SYSTEM_PROMPT.strip()
        if not system_prompt:
            system_prompt = f"""당신은 친근한 영어 대화 파트너입니다.

현재 컨텍스트:
- 시간대: {time_of_day}
- 요일: {day_of_week}

캐주얼한 대화를 시작할 자연스럽고 친근한 인사말을 생성하세요.
짧게 유지하세요 (최대 1-2문장).
시간대와 컨텍스트에 맞게 자연스럽게 만드세요.
간단하고 일상적인 영어를 사용하세요."""

        try:
            # 시간대 정보를 user prompt에 전달
            user_prompt = f"""현재 시간 컨텍스트:
- 시간대: {time_of_day}
- 요일: {day_of_week}
- 현재 시각: {current_time.hour}:00

이 시간 컨텍스트를 바탕으로 대화를 시작할 인사말을 생성하세요."""

            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=60
            )

            greeting = response.choices[0].message.content.strip()
            logger.info(f"Generated greeting: {greeting[:50]}...")
            return greeting

        except Exception as e:
            logger.error(f"Error generating greeting: {str(e)}")
            return "Hey! How's it going today?"

    async def generate_response(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Generate a response to the user's message.

        Args:
            user_message: The user's latest message
            conversation_history: Previous messages

        Returns:
            AI response message
        """
        current_time = datetime.now()
        time_of_day = self._get_time_of_day(current_time.hour)

        # 시스템 프롬프트가 비어있으면 기본값 사용
        system_prompt = RESPONSE_SYSTEM_PROMPT.strip()
        if not system_prompt:
            system_prompt = f"""당신은 친근한 영어 대화 파트너입니다.

컨텍스트:
- 시간대: {time_of_day}

응답 가이드라인:
1. 응답을 짧게 유지하세요 (1-2문장, 최대 20단어)
2. 자연스럽고 대화적으로
3. 대화가 이어지도록 후속 질문하기
4. 간단하고 일상적인 영어 사용
5. AI라고 절대 언급하지 않기"""

        # Build conversation messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (last 8 messages)
        for msg in conversation_history[-8:]:
            role = "assistant" if msg.get("speaker") == "ai" else "user"
            messages.append({"role": role, "content": msg.get("message", "")})

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.8,
                max_tokens=60
            )

            ai_response = response.choices[0].message.content.strip()
            logger.info(f"Generated response: {ai_response[:50]}...")
            return ai_response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "That's interesting! Tell me more about it."

    async def generate_feedback(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        audio_data: str = None
    ) -> Dict[str, Any]:
        """
        Generate feedback for the user's message.

        Args:
            user_message: The message to evaluate
            conversation_history: Conversation context
            audio_data: Optional base64 audio for pronunciation

        Returns:
            Feedback dictionary with corrections, suggestions, scores
        """
        # Handle pronunciation if audio provided
        pronunciation_details = None
        if audio_data:
            try:
                import base64
                from agent.pronunciation.pronunciation_agent import PronunciationAssessmentAgent

                logger.info("Running pronunciation assessment...")
                audio_bytes = base64.b64decode(audio_data)

                agent = PronunciationAssessmentAgent.get_instance()
                pronunciation_result = await agent.assess_pronunciation(
                    audio_data=audio_bytes,
                    reference_text=user_message,
                    language='en-US',
                    granularity='Phoneme'
                )

                pronunciation_details = {
                    'pronunciation_score': pronunciation_result['pronunciation_score'],
                    'accuracy_score': pronunciation_result['accuracy_score'],
                    'fluency_score': pronunciation_result['fluency_score'],
                    'prosody_score': pronunciation_result['prosody_score'],
                    'completeness_score': pronunciation_result['completeness_score'],
                    'words': pronunciation_result['words']
                }
            except Exception as e:
                logger.error(f"Pronunciation assessment failed: {str(e)}")

        # Build pronunciation info for prompt
        pronunciation_info = ""
        if pronunciation_details:
            pronunciation_info = f"""
Azure Pronunciation Assessment Results:
- Overall: {pronunciation_details['pronunciation_score']:.1f}/100
- Accuracy: {pronunciation_details['accuracy_score']:.1f}/100
- Fluency: {pronunciation_details['fluency_score']:.1f}/100
- Prosody: {pronunciation_details['prosody_score']:.1f}/100

Words with issues (accuracy < 80):
{chr(10).join([f"- '{w['word']}': {w['accuracy_score']:.1f}" for w in pronunciation_details['words'] if w['accuracy_score'] < 80][:5]) if any(w['accuracy_score'] < 80 for w in pronunciation_details['words']) else '(All words pronounced well)'}
"""

        # 시스템 프롬프트가 비어있으면 기본값 사용
        system_prompt = FEEDBACK_SYSTEM_PROMPT.strip()
        if not system_prompt:
            system_prompt = """당신은 캐주얼 대화에 대한 피드백을 제공하는 친근한 영어 튜터입니다.

컨텍스트:
- 대화 유형: 캐주얼 스몰토크

점수 가이드라인:
- 격려하되, 의사소통 성공에 집중
- 메시지가 이해 가능하고 적절하면 7점 이상 부여

한국어로 피드백을 제공하되 영어 예시를 포함하세요."""

        user_prompt = f"""사용자 메시지: "{user_message}"
{pronunciation_info}

먼저: 사용자가 어떤 언어로 작성했는지 감지하세요.
- 영어가 아닌 경우 (한국어, 일본어 등): 점수 1-3, 영어 사용 강력히 권장
- 오류가 있는 영어인 경우: 심각도에 따라 점수 부여, 교정 제공
- 올바른 영어인 경우: 자연스러움에 따라 7-10점 부여

다음 JSON 형식으로 피드백을 제공하세요:
{{
  "language_detected": "English" or "Korean" or "Mixed" etc.,
  "grammar_corrections": [
    "문법 오류 교정 (한국어 설명 + 영어 예시)"
  ],
  "suggestions": [
    "개선 제안 (한국어 설명 + 영어 예시)"
  ],
  "pronunciation_feedback": [
    "발음 관련 피드백 (있는 경우만)"
  ],
  "score": 5,
  "score_breakdown": {{
    "grammar": 5,
    "vocabulary": 5,
    "fluency": 5,
    "pronunciation": 5
  }}
}}

중요사항:
- 모든 설명은 한국어로 작성
- 객관적으로: 사용자가 영어를 사용하지 않았다면 문법이 맞다고 하지 말 것
- 사용자가 한국어로 작성한 경우: 영어 사용 권장, 낮은 점수 부여
- 사용자가 문법 오류를 범한 경우: 무엇이 잘못되었는지 명확히 설명
- 진정으로 칭찬할 만한 경우에만 칭찬"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=500
            )

            feedback = json.loads(response.choices[0].message.content)

            if pronunciation_details:
                feedback['pronunciation_details'] = pronunciation_details

            logger.info(f"Generated feedback with score: {feedback.get('score', 'N/A')}")
            return feedback

        except Exception as e:
            logger.error(f"Error generating feedback: {str(e)}")
            return self._get_fallback_feedback()

    async def generate_hint(
        self,
        conversation_history: List[Dict[str, str]],
        last_ai_message: str,
        hint_count: int = 3
    ) -> Dict[str, Any]:
        """
        Generate response hints for the user.

        Args:
            conversation_history: Conversation context
            last_ai_message: The AI message to respond to
            hint_count: Number of hints to generate

        Returns:
            Dictionary with hints and explanations
        """
        # 시스템 프롬프트가 비어있으면 기본값 사용
        system_prompt = HINT_SYSTEM_PROMPT.strip()
        if not system_prompt:
            system_prompt = f"""당신은 누군가가 영어 스몰토크를 연습할 수 있도록 돕고 있습니다.
그들이 응답하고 있는 메시지: "{last_ai_message}"

다음 조건을 만족하는 {hint_count}개의 다른 응답 제안을 생성하세요:
1. 캐주얼한 대화에 자연스러운 표현
2. 대화가 이어지도록 유지
3. 접근 방식을 다양화 (동의, 질문, 공유 등)
4. 간단하고 일상적인 영어 사용"""

        user_prompt = f"""{hint_count}개의 응답 힌트를 다음 JSON 형식으로 생성하세요:
{{
  "hints": [
    "영어 응답 예시 1",
    "영어 응답 예시 2",
    "영어 응답 예시 3"
  ],
  "explanations": [
    "이 표현 사용 상황 설명 (한국어)",
    "이 표현 사용 상황 설명 (한국어)",
    "이 표현 사용 상황 설명 (한국어)"
  ]
}}

힌트는 영어로, 설명은 한국어로 작성하세요."""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.8,
                max_tokens=400
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            logger.error(f"Error generating hints: {str(e)}")
            return {
                "hints": ["I see what you mean.", "That's interesting!", "Could you tell me more?"],
                "explanations": ["동의를 표현할 때", "관심을 보일 때", "더 알고 싶을 때"]
            }

    def _get_time_of_day(self, hour: int) -> str:
        """Get time of day description."""
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    def _get_fallback_feedback(self) -> Dict[str, Any]:
        """Fallback feedback when AI generation fails."""
        return {
            "grammar_corrections": [],
            "suggestions": ["대화를 잘 이어가고 계세요! 계속 연습해보세요."],
            "score": 7,
            "score_breakdown": {
                "grammar": 7,
                "vocabulary": 7,
                "fluency": 7,
                "pronunciation": 7
            }
        }
