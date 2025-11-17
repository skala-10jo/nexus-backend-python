"""
회화 연습 서비스
GPT-4o를 이용한 시나리오 기반 대화 생성
"""
import logging
from typing import List, Dict, Any
from uuid import UUID
from openai import AsyncOpenAI

from app.config import settings
from app.database import get_db

logger = logging.getLogger(__name__)


class ConversationService:
    """회화 연습 대화 생성 서비스"""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def start_conversation(
        self,
        scenario_id: str,
        user_id: UUID
    ) -> Dict[str, Any]:
        """
        대화 시작

        Args:
            scenario_id: 시나리오 ID
            user_id: 사용자 ID

        Returns:
            시나리오 정보 및 초기 AI 메시지
        """
        try:
            # DB에서 시나리오 조회
            db = next(get_db())
            from app.models.scenario import Scenario

            scenario = db.query(Scenario).filter(
                Scenario.id == UUID(scenario_id),
                Scenario.user_id == user_id
            ).first()

            if not scenario:
                raise ValueError(f"Scenario not found: {scenario_id}")

            # 초기 AI 메시지 생성
            initial_message = await self._generate_initial_message(scenario)

            return {
                "scenario": {
                    "id": str(scenario.id),
                    "title": scenario.title,
                    "description": scenario.description,
                    "difficulty": scenario.difficulty,
                    "category": scenario.category,
                    "language": scenario.language,
                    "roles": scenario.roles,
                    "required_terminology": scenario.required_terminology,
                    "scenario_text": scenario.scenario_text
                },
                "initialMessage": initial_message
            }

        except Exception as e:
            logger.error(f"Error starting conversation: {str(e)}")
            raise

    async def send_message(
        self,
        scenario_id: str,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        user_id: UUID
    ) -> Dict[str, Any]:
        """
        사용자 메시지 전송 및 AI 응답 생성

        Args:
            scenario_id: 시나리오 ID
            user_message: 사용자 메시지
            conversation_history: 대화 히스토리
            user_id: 사용자 ID

        Returns:
            AI 응답 및 감지된 전문용어
        """
        try:
            # DB에서 시나리오 조회
            db = next(get_db())
            from app.models.scenario import Scenario

            scenario = db.query(Scenario).filter(
                Scenario.id == UUID(scenario_id),
                Scenario.user_id == user_id
            ).first()

            if not scenario:
                raise ValueError(f"Scenario not found: {scenario_id}")

            # AI 응답 생성
            ai_message = await self._generate_ai_response(
                scenario=scenario,
                user_message=user_message,
                conversation_history=conversation_history
            )

            # 전문용어 감지
            detected_terms = self._detect_terminology(
                message=user_message,
                required_terminology=scenario.required_terminology
            )

            return {
                "aiMessage": ai_message,
                "detectedTerms": detected_terms
            }

        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            raise

    async def _generate_initial_message(self, scenario) -> str:
        """
        초기 AI 메시지 생성 (스몰토크로 시작)

        Args:
            scenario: 시나리오 객체

        Returns:
            초기 AI 메시지
        """
        try:
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%d")

            # 시스템 프롬프트
            system_prompt = f"""You are participating in a business conversation practice scenario.

Today's date: {current_date}

Scenario: {scenario.title}
Description: {scenario.description}
Context: {scenario.scenario_text}

Your Role: {scenario.roles.get('ai', 'AI')}
User's Role: {scenario.roles.get('user', 'User')}

Language: {scenario.language}
Difficulty: {scenario.difficulty}

Required Terminology to use naturally later: {', '.join(scenario.required_terminology)}

Instructions:
- Start with friendly small talk (greetings, how are you, weather, weekend, etc.)
- Keep it casual and natural (2-3 sentences)
- Gradually transition into the business scenario topic
- Respond in {scenario.language} language
- Be encouraging and supportive for language practice
- Use realistic dates and contexts based on today's date"""

            # GPT-4o 호출
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Start the conversation with casual small talk like greeting, asking how they are, or mentioning the weather."}
                ],
                temperature=0.8,
                max_tokens=150
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating initial message: {str(e)}")
            return "Hello! How are you doing today?"

    async def _generate_ai_response(
        self,
        scenario,
        user_message: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        AI 응답 생성 (스몰토크 포함)

        Args:
            scenario: 시나리오 객체
            user_message: 사용자 메시지
            conversation_history: 대화 히스토리

        Returns:
            AI 응답 메시지
        """
        try:
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_time = datetime.now().strftime("%H:%M")

            # 시스템 프롬프트
            system_prompt = f"""You are participating in a business conversation practice scenario.

Today's date: {current_date}
Current time: {current_time}

Scenario: {scenario.title}
Description: {scenario.description}
Context: {scenario.scenario_text}

Your Role: {scenario.roles.get('ai', 'AI')}
User's Role: {scenario.roles.get('user', 'User')}

Language: {scenario.language}
Difficulty: {scenario.difficulty}

Required Terminology to use naturally: {', '.join(scenario.required_terminology)}

Instructions:
- Mix casual small talk naturally with business topics (like real conversations)
- Keep responses conversational and natural (2-3 sentences)
- Occasionally ask about personal things (weekend, lunch, weather) before/after business talk
- Use the required terminology naturally when discussing business
- Respond in {scenario.language} language
- Be encouraging and supportive for language practice
- If user makes grammatical errors, gently incorporate corrections in your response
- Use realistic dates and times based on today's context"""

            # 대화 히스토리 구성
            messages = [{"role": "system", "content": system_prompt}]

            # 이전 대화 추가
            for msg in conversation_history[-10:]:  # 최근 10개만
                role = "assistant" if msg["speaker"] == "ai" else "user"
                messages.append({"role": role, "content": msg["message"]})

            # 현재 사용자 메시지 추가
            messages.append({"role": "user", "content": user_message})

            # GPT-4o 호출
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.8,
                max_tokens=200
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            raise

    def _detect_terminology(
        self,
        message: str,
        required_terminology: List[str]
    ) -> List[str]:
        """
        메시지에서 전문용어 감지

        Args:
            message: 사용자 메시지
            required_terminology: 필수 전문용어 목록

        Returns:
            감지된 전문용어 목록
        """
        detected = []
        message_lower = message.lower()

        for term in required_terminology:
            if term.lower() in message_lower:
                detected.append(term)

        return detected
