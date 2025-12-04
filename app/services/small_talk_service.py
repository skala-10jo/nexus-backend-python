"""
Small Talk Service

스몰토크 대화 서비스.
DB 없이 메모리 기반으로 동작하며, SmallTalkAgent를 조율합니다.
"""
import logging
from typing import List, Dict, Any

from agent.small_talk.persona_agent import SmallTalkAgent

logger = logging.getLogger(__name__)


class SmallTalkService:
    """
    스몰토크 대화 비즈니스 로직

    SmallTalkAgent를 사용하여 대화를 생성합니다.
    DB 저장 없이 stateless하게 동작합니다.
    """

    def __init__(self):
        self.agent = SmallTalkAgent()

    async def start_conversation(self) -> Dict[str, Any]:
        """
        대화 시작 - 초기 인사 생성

        Returns:
            초기 메시지
        """
        try:
            logger.info("Starting small talk conversation")

            # 초기 인사 생성
            initial_message = await self.agent.generate_greeting()

            return {
                "initial_message": initial_message
            }

        except Exception as e:
            logger.error(f"Error starting conversation: {str(e)}")
            raise

    async def send_message(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        메시지 전송 및 AI 응답 생성

        Args:
            user_message: 사용자 메시지
            conversation_history: 대화 히스토리 (프론트엔드에서 관리)

        Returns:
            AI 응답 메시지
        """
        try:
            logger.info(f"Processing message: {user_message[:50]}...")

            # AI 응답 생성
            ai_message = await self.agent.generate_response(
                user_message=user_message,
                conversation_history=conversation_history
            )

            return {
                "ai_message": ai_message
            }

        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            raise

    async def generate_feedback(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        audio_data: str = None
    ) -> Dict[str, Any]:
        """
        사용자 메시지에 대한 피드백 생성

        Args:
            user_message: 피드백 받을 메시지
            conversation_history: 대화 히스토리 (맥락 파악용)
            audio_data: Base64 인코딩된 오디오 (발음 평가용, 선택)

        Returns:
            피드백 딕셔너리 (문법 교정, 제안, 점수 등)
        """
        try:
            logger.info(f"Generating feedback for message: {user_message[:50]}...")

            feedback = await self.agent.generate_feedback(
                user_message=user_message,
                conversation_history=conversation_history,
                audio_data=audio_data
            )

            return feedback

        except Exception as e:
            logger.error(f"Error generating feedback: {str(e)}")
            raise

    async def generate_hint(
        self,
        conversation_history: List[Dict[str, str]],
        last_ai_message: str,
        hint_count: int = 3
    ) -> Dict[str, Any]:
        """
        응답 힌트 생성

        Args:
            conversation_history: 대화 히스토리
            last_ai_message: 마지막 AI 메시지
            hint_count: 생성할 힌트 개수

        Returns:
            힌트 목록과 설명
        """
        try:
            logger.info(f"Generating {hint_count} hints")

            result = await self.agent.generate_hint(
                conversation_history=conversation_history,
                last_ai_message=last_ai_message,
                hint_count=hint_count
            )

            return result

        except Exception as e:
            logger.error(f"Error generating hint: {str(e)}")
            raise
