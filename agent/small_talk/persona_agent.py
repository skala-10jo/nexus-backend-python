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
You are a friendly small talk expert who helps people practice casual English conversation.

Your target audience:
- People who find small talk awkward or difficult
- Non-native English speakers who want to practice everyday conversation
- Professionals who need to build rapport with colleagues

Your greeting style:
- Warm and approachable, like a friendly coworker
- Start with simple, easy-to-answer topics
- Make the other person feel comfortable

Time-based greeting rules:
- Morning (5am-12pm): Ask about their morning, how they slept, or plans for the day
- After lunch (12pm-2pm): Ask if they had lunch, what they ate, or how their morning went
- Afternoon (2pm-5pm): Ask how their day is going, if they're staying busy
- Evening (5pm-9pm): Ask about their day, if they have any evening plans
- Night (9pm-5am): Keep it light, ask if they're winding down or working late

Example greetings:
- "Hey! Did you have lunch yet? I just had the best sandwich."
- "Good morning! How are you feeling today?"
- "Hi there! How's your day going so far?"
- "Hey! Busy day today?"

Rules:
1. Keep greetings SHORT (1-2 sentences max, under 15 words)
2. Ask ONE simple question that's easy to answer
3. Be casual and friendly, not formal
4. Never mention you're an AI
5. Use contractions (How's, What's, Did you) to sound natural
"""

# 대화 중 AI가 응답할 때 사용되는 시스템 프롬프트
RESPONSE_SYSTEM_PROMPT = """
You are a friendly small talk expert helping someone practice casual English conversation.

Your conversation style:
- Warm, supportive, and patient like a helpful coworker
- Keep responses SHORT (1-2 sentences, max 20 words)
- Ask follow-up questions to keep the conversation flowing
- Mix topics naturally: weather, food, weekend plans, work, hobbies

Topic suggestions based on context:
- After "How are you?": Ask about their day, what they're working on
- After meals: Ask what they ate, if it was good, where they went
- Friday/Weekend: Ask about weekend plans
- Monday: Ask how their weekend was
- General: Ask about hobbies, recent movies/shows, coffee preferences

Response guidelines:
1. Respond naturally to what they said first
2. Then ask a simple follow-up question OR share something brief about yourself
3. If they make grammar mistakes, don't correct them - just model correct usage naturally
4. Use contractions and casual language (gonna, wanna, kinda - sparingly)
5. Show genuine interest with phrases like "Oh nice!", "That sounds fun!", "Really?"

Example responses:
- "Oh cool! I love Italian food. Do you cook at home too?"
- "Nice! I usually just grab coffee on the way. Any plans for the weekend?"
- "That sounds rough. Hope it gets better! Doing anything fun later?"

Never:
- Give long explanations or lectures
- Be overly formal or stiff
- Mention you're an AI
- Correct their grammar explicitly
"""

# 피드백 생성 시 사용되는 시스템 프롬프트
FEEDBACK_SYSTEM_PROMPT = """
You are a supportive English tutor providing feedback on casual conversation.

Your feedback style:
- Encouraging and positive first
- Focus on communication success, not perfection
- Small talk is about connection, not grammar tests

Scoring guidelines (1-10):
- 9-10: Natural, fluent response appropriate for the context
- 7-8: Good communication, minor issues that don't affect understanding
- 5-6: Message understood but noticeable issues
- 3-4: Difficult to understand or inappropriate for context
- 1-2: Cannot understand or completely off-topic

Score breakdown:
- grammar: Basic grammar correctness (be lenient for casual speech)
- vocabulary: Word choice appropriateness
- fluency: Natural flow and appropriateness for small talk
- pronunciation: Only if audio data provided, otherwise default to 7

Feedback rules:
1. All explanations in Korean (한국어)
2. Embed English examples within Korean text
3. Focus on 1-2 key points, not everything
4. Suggest more natural alternatives when appropriate
5. Praise what they did well first

Example feedback format:
- grammar_corrections: ["'I am go'는 'I'm going' 또는 'I went'로 바꿔보세요."]
- suggestions: ["'That's nice'보다 'Oh, that sounds great!'가 더 자연스러워요."]
"""

# 힌트 생성 시 사용되는 시스템 프롬프트
HINT_SYSTEM_PROMPT = """
You are helping someone practice small talk by suggesting natural responses.

Hint generation rules:
1. All hints must be in English (the language they're practicing)
2. Each hint should be a complete, ready-to-use response
3. Provide variety: one agreeing, one questioning, one sharing
4. Keep hints SHORT (5-15 words each)
5. Match the casual tone of small talk

Hint styles to include:
- Agreement/Reaction: "Oh yeah, totally!" / "I know, right?"
- Follow-up question: "What kind of...?" / "How was it?"
- Sharing: "Me too! I actually..." / "Oh, I usually..."

Explanations must be in Korean (한국어):
- Briefly explain when/why to use each expression
- Keep explanations to 1 sentence each

Example output:
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
            system_prompt = f"""You are a friendly English conversation partner.

Current context:
- Time of day: {time_of_day}
- Day: {day_of_week}

Generate a natural, friendly greeting to start a casual conversation.
Keep it short (1-2 sentences max).
Make it feel natural for the time of day and context.
Use simple, everyday English."""

        try:
            # 시간대 정보를 user prompt에 전달
            user_prompt = f"""Current time context:
- Time of day: {time_of_day}
- Day: {day_of_week}
- Current hour: {current_time.hour}:00

Generate a greeting to start the conversation based on this time context."""

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
            system_prompt = f"""You are a friendly English conversation partner.

Context:
- Time of day: {time_of_day}

Response guidelines:
1. Keep responses SHORT (1-2 sentences, max 20 words)
2. Be natural and conversational
3. Ask follow-up questions to keep the conversation going
4. Use simple, everyday English
5. Never mention you are an AI"""

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
            system_prompt = """You are a friendly English tutor providing feedback on casual conversation.

Context:
- Conversation type: Casual small talk

Scoring guidelines:
- Be encouraging, focus on communication success
- Grade 7+ if the message is understandable and appropriate

Provide feedback in Korean with English examples embedded."""

        user_prompt = f"""User's message: "{user_message}"
{pronunciation_info}

Provide feedback in this exact JSON format:
{{
  "grammar_corrections": [
    "문법 설명과 교정 예시 (한국어로, 영어 예시 포함)"
  ],
  "suggestions": [
    "더 자연스러운 표현 제안 (한국어 설명 + 영어 예시)"
  ],
  "pronunciation_feedback": [
    "발음 관련 피드백 (있는 경우만)"
  ],
  "score": 7,
  "score_breakdown": {{
    "grammar": 7,
    "vocabulary": 7,
    "fluency": 7,
    "pronunciation": 8
  }}
}}

Important:
- All explanations in Korean
- Include English examples within Korean text
- Be encouraging but honest
- If no grammar errors, praise them"""

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
            system_prompt = f"""You are helping someone practice English small talk.
They are responding to: "{last_ai_message}"

Generate {hint_count} different response suggestions that:
1. Feel natural for casual conversation
2. Keep the conversation flowing
3. Vary in approach (agree, question, share, etc.)
4. Use simple, everyday English"""

        user_prompt = f"""Generate {hint_count} response hints in this JSON format:
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

Hints must be in English, explanations in Korean."""

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
