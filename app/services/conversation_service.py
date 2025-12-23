"""
회화 연습 서비스
GPT-4o를 이용한 시나리오 기반 대화 생성
"""
import base64
import json
import logging
from typing import List, Dict, Any
from uuid import UUID

from app.database import get_db
from app.models.scenario import Scenario
from app.models.conversation import ConversationSession, ConversationMessage
from agent.scenario.response_agent import ResponseAgent
from agent.scenario.feedback_agent import FeedbackAgent
from agent.scenario.hint_agent import HintAgent
from agent.translate import ContextEnhancedTranslationAgent
from agent.pronunciation.pronunciation_agent import PronunciationAssessmentAgent

logger = logging.getLogger(__name__)


class ConversationService:
    """회화 연습 대화 생성 서비스"""

    def __init__(self):
        self.response_agent = ResponseAgent()
        self.feedback_agent = FeedbackAgent()
        self.translation_agent = ContextEnhancedTranslationAgent()

    def _build_scenario_dict(self, scenario) -> Dict[str, Any]:
        """시나리오 응답 객체 생성"""
        return {
            "id": str(scenario.id),
            "title": scenario.title,
            "description": scenario.description,
            "difficulty": scenario.difficulty,
            "category": scenario.category,
            "language": scenario.language,
            "roles": scenario.roles,
            "requiredTerms": scenario.required_terminology,
            "scenario_text": scenario.scenario_text,
            "steps": scenario.steps or []
        }

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

            scenario = db.query(Scenario).filter(
                Scenario.id == UUID(scenario_id),
                Scenario.user_id == user_id
            ).first()

            if not scenario:
                raise ValueError(f"Scenario not found: {scenario_id}")

            # 기존 active 세션이 있는지 먼저 확인
            existing_session = db.query(ConversationSession).filter(
                ConversationSession.scenario_id == UUID(scenario_id),
                ConversationSession.user_id == user_id,
                ConversationSession.status == 'active'
            ).first()

            if existing_session and existing_session.total_messages > 0:
                # 기존 세션이 있고 메시지가 있으면 시나리오 정보만 반환 (초기 메시지 없음)
                logger.info(f"Existing session found with {existing_session.total_messages} messages, skipping initial message")
                return {
                    "scenario": self._build_scenario_dict(scenario),
                    "initialMessage": None,
                    "sessionId": str(existing_session.id)
                }

            # 새 세션 생성
            session = await self._get_or_create_session(db, scenario_id, user_id)

            # 초기 AI 메시지 생성
            initial_message = await self._generate_initial_message(scenario)

            # AI 초기 메시지를 DB에 저장
            await self._save_message(
                db=db,
                session_id=session.id,
                sender="ai",
                message_text=initial_message,
                sequence_number=1
            )

            return {
                "scenario": self._build_scenario_dict(scenario),
                "initialMessage": initial_message,
                "sessionId": str(session.id)
            }

        except Exception as e:
            logger.error(f"Error starting conversation: {str(e)}")
            raise

    async def send_message(
        self,
        scenario_id: str,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        user_id: UUID,
        current_step_index: int = 0
    ) -> Dict[str, Any]:
        """
        사용자 메시지 전송 및 AI 응답 생성

        Args:
            scenario_id: 시나리오 ID
            user_message: 사용자 메시지
            conversation_history: 대화 히스토리 (프론트에서 전달, 백업용)
            user_id: 사용자 ID
            current_step_index: 현재 스텝 인덱스 (0-based)

        Returns:
            AI 응답, 감지된 전문용어, 스텝 완료 여부
        """
        try:
            # DB에서 시나리오 조회
            db = next(get_db())

            scenario = db.query(Scenario).filter(
                Scenario.id == UUID(scenario_id),
                Scenario.user_id == user_id
            ).first()

            if not scenario:
                raise ValueError(f"Scenario not found: {scenario_id}")

            # 세션 가져오기 또는 생성
            session = await self._get_or_create_session(db, scenario_id, user_id)

            # 현재 메시지 번호 계산
            next_seq = session.total_messages + 1

            # 사용자 메시지 DB에 저장
            await self._save_message(
                db=db,
                session_id=session.id,
                sender="user",
                message_text=user_message,
                sequence_number=next_seq
            )

            # 스텝 정보 가져오기 (steps 컬럼이 있는 경우)
            steps = getattr(scenario, 'steps', None) or []
            current_step = steps[current_step_index] if steps and current_step_index < len(steps) else None

            # 디버깅용 스텝 정보 로그
            logger.info(f"Step info - total_steps: {len(steps)}, current_step_index: {current_step_index}, has_current_step: {current_step is not None}")
            if current_step:
                logger.info(f"Current step: {current_step.get('name', 'Unknown')} - {current_step.get('title', 'No title')}")

            # AI 응답 생성 (스텝 판단 포함)
            ai_response = await self._generate_ai_response(
                scenario=scenario,
                user_message=user_message,
                conversation_history=conversation_history,
                current_step=current_step,
                current_step_index=current_step_index,
                total_steps=len(steps) if steps else 0
            )

            # 전문용어 감지
            detected_terms = self._detect_terminology(
                message=user_message,
                required_terminology=scenario.required_terminology
            )

            # AI 메시지 DB에 저장
            await self._save_message(
                db=db,
                session_id=session.id,
                sender="ai",
                message_text=ai_response["message"],
                detected_terms=detected_terms,
                sequence_number=next_seq + 1
            )

            return {
                "aiMessage": ai_response["message"],
                "detectedTerms": detected_terms,
                "stepCompleted": ai_response.get("step_completed", False)
            }

        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            raise

    async def _generate_initial_message(self, scenario) -> str:
        """
        초기 AI 메시지 생성 (ResponseAgent 위임)

        Args:
            scenario: 시나리오 객체

        Returns:
            초기 AI 메시지
        """
        scenario_context = {
            "title": scenario.title,
            "description": scenario.description,
            "scenario_text": scenario.scenario_text,
            "roles": scenario.roles,
            "language": scenario.language,
            "difficulty": scenario.difficulty,
            "required_terminology": scenario.required_terminology
        }

        return await self.response_agent.process(
            scenario_context=scenario_context,
            mode="initial"
        )

    async def _generate_ai_response(
        self,
        scenario,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        current_step: Dict[str, Any] = None,
        current_step_index: int = 0,
        total_steps: int = 0
    ) -> Dict[str, Any]:
        """
        AI 응답 생성 (ResponseAgent 위임)

        Args:
            scenario: 시나리오 객체
            user_message: 사용자 메시지
            conversation_history: 대화 히스토리
            current_step: 현재 스텝 정보 (있는 경우)
            current_step_index: 현재 스텝 인덱스
            total_steps: 전체 스텝 수

        Returns:
            AI 응답 메시지와 스텝 완료 여부를 포함한 딕셔너리
        """
        scenario_context = {
            "title": scenario.title,
            "description": scenario.description,
            "scenario_text": scenario.scenario_text,
            "roles": scenario.roles,
            "language": scenario.language,
            "difficulty": scenario.difficulty,
            "required_terminology": scenario.required_terminology
        }

        return await self.response_agent.process(
            scenario_context=scenario_context,
            mode="conversation",
            user_message=user_message,
            conversation_history=conversation_history,
            current_step=current_step,
            current_step_index=current_step_index,
            total_steps=total_steps
        )

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

    async def generate_message_feedback(
        self,
        scenario_id: str,
        user_message: str,
        detected_terms: List[str],
        user_id: UUID,
        audio_data: str = None,
        current_step: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        사용자 메시지에 대한 피드백 생성 (FeedbackAgent 위임)

        Args:
            scenario_id: 시나리오 ID
            user_message: 사용자 메시지
            detected_terms: 감지된 전문용어
            user_id: 사용자 ID
            audio_data: Base64 인코딩된 오디오 데이터 (선택)
            current_step: 현재 대화 단계 정보 (선택)

        Returns:
            피드백 (문법 교정, 용어 사용, 제안, 점수, 단계별 표현 피드백)
        """
        try:
            # DB에서 시나리오 조회
            db = next(get_db())

            scenario = db.query(Scenario).filter(
                Scenario.id == UUID(scenario_id),
                Scenario.user_id == user_id
            ).first()

            if not scenario:
                raise ValueError(f"Scenario not found: {scenario_id}")

            # Azure Pronunciation Assessment 수행 (오디오 데이터가 있는 경우)
            pronunciation_details = None
            if audio_data:
                try:
                    logger.info("Running Azure Pronunciation Assessment...")
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

                    logger.info(f"Pronunciation assessment completed: {pronunciation_result['pronunciation_score']:.1f}")

                except Exception as e:
                    logger.error(f"Pronunciation assessment failed: {str(e)}", exc_info=True)

            # 세션에서 이전에 사용한 용어들 조회
            previously_used_terms = await self._get_session_used_terms(db, scenario_id, user_id)
            logger.info(f"Previously used terms in session: {previously_used_terms}")

            # 필수 용어 목록
            required_terms = scenario.required_terminology or []

            # 현재 메시지에서 감지된 용어 (정확히 일치)
            current_detected = [term for term in required_terms if term.lower() in user_message.lower()]

            # 아직 사용되지 않은 용어 = 필수 용어 - (이전 사용 + 현재 사용)
            all_used_terms = set(previously_used_terms) | set(current_detected)
            missed_terms = [term for term in required_terms if term not in all_used_terms]

            # 시나리오 컨텍스트 구성
            scenario_context = {
                "title": scenario.title,
                "description": scenario.description,
                "scenario_text": scenario.scenario_text,
                "roles": scenario.roles,
                "language": scenario.language,
                "difficulty": scenario.difficulty,
                "category": scenario.category,
                "required_terminology": required_terms
            }

            # FeedbackAgent로 피드백 생성 위임
            # 이전에 사용한 용어 정보도 함께 전달
            feedback = await self.feedback_agent.process(
                scenario_context=scenario_context,
                user_message=user_message,
                detected_terms=detected_terms,
                missed_terms=missed_terms,
                current_step=current_step,
                pronunciation_details=pronunciation_details,
                previously_used_terms=list(previously_used_terms)
            )

            # FeedbackAgent가 의미적 유사성 기반으로 판단한 용어 사용 정보를 최상위에 추가
            # 프론트엔드에서 쉽게 접근할 수 있도록
            terminology_usage = feedback.get("terminology_usage", {})
            feedback["detectedTerms"] = terminology_usage.get("used", [])  # 현재 메시지에서 사용
            feedback["previouslyUsedTerms"] = terminology_usage.get("previously_used", [])  # 이전에 이미 사용
            feedback["missedTerms"] = terminology_usage.get("missed", [])  # 아직 미사용
            feedback["similarExpressions"] = terminology_usage.get("similar_expressions", {})

            # 피드백을 마지막 사용자 메시지에 저장
            await self._save_feedback_to_message(
                db=db,
                scenario_id=scenario_id,
                user_id=user_id,
                user_message=user_message,
                feedback=json.dumps(feedback, ensure_ascii=False)
            )

            return feedback

        except Exception as e:
            logger.error(f"Error generating feedback: {str(e)}")
            raise

    async def translate_message(
        self,
        scenario_id: str,
        message: str,
        source_language: str,
        target_language: str,
        user_id: UUID
    ) -> str:
        """
        메시지 번역 (컨텍스트 기반 ContextEnhancedTranslationAgent 사용)

        시나리오의 맥락과 전문용어를 활용하여 일관성 있는 번역을 제공합니다.

        Args:
            scenario_id: 시나리오 ID
            message: 번역할 메시지
            source_language: 원본 언어 코드 (en, ko 등)
            target_language: 목표 언어 코드 (ko, en 등)
            user_id: 사용자 ID

        Returns:
            번역된 텍스트
        """
        try:
            # DB에서 시나리오 조회
            db = next(get_db())

            scenario = db.query(Scenario).filter(
                Scenario.id == UUID(scenario_id),
                Scenario.user_id == user_id
            ).first()

            if not scenario:
                raise ValueError(f"Scenario not found: {scenario_id}")

            # 시나리오 컨텍스트 구성
            context = f"""시나리오: {scenario.title}
설명: {scenario.description}
상황: {scenario.scenario_text}
카테고리: {scenario.category}
난이도: {scenario.difficulty}"""

            # required_terminology를 용어집 포맷으로 변환
            glossary_terms = [
                {"english_term": term}
                for term in (scenario.required_terminology or [])
            ]

            # ContextEnhancedTranslationAgent로 번역 수행
            translated_text = await self.translation_agent.process(
                text=message,
                source_lang=source_language,
                target_lang=target_language,
                context=context,
                glossary_terms=glossary_terms,
                detected_terms=[]  # 간단한 회화 번역에서는 탐지된 용어 없이 진행
            )

            logger.info(f"Translated message for scenario {scenario_id}: {source_language} -> {target_language}")
            return translated_text

        except Exception as e:
            logger.error(f"Error translating message: {str(e)}")
            raise

    async def reset_conversation(
        self,
        scenario_id: str,
        user_id: UUID
    ) -> None:
        """
        대화 초기화 - 해당 시나리오의 모든 세션 및 메시지 삭제

        Args:
            scenario_id: 시나리오 ID
            user_id: 사용자 ID
        """
        try:
            db = next(get_db())

            # 해당 시나리오의 모든 세션 삭제 (CASCADE로 메시지도 자동 삭제)
            deleted_count = db.query(ConversationSession).filter(
                ConversationSession.scenario_id == UUID(scenario_id),
                ConversationSession.user_id == user_id
            ).delete()

            db.commit()
            logger.info(f"Reset conversation for scenario {scenario_id}: deleted {deleted_count} sessions")

        except Exception as e:
            logger.error(f"Error resetting conversation: {str(e)}")
            if db:
                db.rollback()
            raise

    async def get_conversation_history(
        self,
        scenario_id: str,
        user_id: UUID
    ) -> Dict[str, Any]:
        """
        저장된 대화 히스토리 조회

        Args:
            scenario_id: 시나리오 ID
            user_id: 사용자 ID

        Returns:
            세션 정보 및 메시지 목록
        """
        try:
            db = next(get_db())

            # 가장 최근 active 세션 조회
            session = db.query(ConversationSession).filter(
                ConversationSession.scenario_id == UUID(scenario_id),
                ConversationSession.user_id == user_id,
                ConversationSession.status == 'active'
            ).order_by(ConversationSession.started_at.desc()).first()

            if not session:
                return {
                    "sessionId": None,
                    "messages": []
                }

            # 세션의 메시지들 조회
            messages = db.query(ConversationMessage).filter(
                ConversationMessage.session_id == session.id
            ).order_by(ConversationMessage.sequence_number.asc()).all()

            message_list = [
                {
                    "id": str(msg.id),
                    "sender": msg.sender,
                    "message": msg.message_text,
                    "translatedText": msg.translated_text,
                    "detectedTerms": msg.detected_terms or [],
                    "feedback": msg.feedback,
                    "sequenceNumber": msg.sequence_number,
                    "createdAt": msg.created_at.isoformat()
                }
                for msg in messages
            ]

            return {
                "sessionId": str(session.id),
                "status": session.status,
                "startedAt": session.started_at.isoformat(),
                "totalMessages": session.total_messages,
                "messages": message_list
            }

        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            raise

    async def _get_or_create_session(
        self,
        db,
        scenario_id: str,
        user_id: UUID
    ):
        """
        세션 가져오기 또는 생성

        Args:
            db: 데이터베이스 세션
            scenario_id: 시나리오 ID
            user_id: 사용자 ID

        Returns:
            ConversationSession 객체
        """
        # 기존 active 세션 찾기
        session = db.query(ConversationSession).filter(
            ConversationSession.scenario_id == UUID(scenario_id),
            ConversationSession.user_id == user_id,
            ConversationSession.status == 'active'
        ).order_by(ConversationSession.started_at.desc()).first()

        if session:
            return session

        # 세션이 없으면 새로 생성
        scenario = db.query(Scenario).filter(Scenario.id == UUID(scenario_id)).first()

        new_session = ConversationSession(
            user_id=user_id,
            scenario_id=UUID(scenario_id),
            status='active',
            user_role=scenario.roles.get('user', 'User') if scenario else None,
            ai_role=scenario.roles.get('ai', 'AI') if scenario else None,
            total_messages=0
        )
        db.add(new_session)
        db.commit()
        db.refresh(new_session)

        logger.info(f"Created new conversation session: {new_session.id}")
        return new_session

    async def _save_message(
        self,
        db,
        session_id: UUID,
        sender: str,
        message_text: str,
        sequence_number: int,
        translated_text: str = None,
        detected_terms: List[str] = None,
        feedback: str = None
    ) -> None:
        """
        메시지를 DB에 저장

        Args:
            db: 데이터베이스 세션
            session_id: 세션 ID
            sender: 발신자 ('user' or 'ai')
            message_text: 메시지 내용
            sequence_number: 메시지 순서
            translated_text: 번역된 텍스트 (선택)
            detected_terms: 감지된 전문용어 (선택)
            feedback: 피드백 (선택)
        """
        new_message = ConversationMessage(
            session_id=session_id,
            sender=sender,
            message_text=message_text,
            translated_text=translated_text,
            detected_terms=detected_terms,
            feedback=feedback,
            sequence_number=sequence_number
        )
        db.add(new_message)

        # 세션의 total_messages 증가
        session = db.query(ConversationSession).filter(
            ConversationSession.id == session_id
        ).first()
        if session:
            session.total_messages += 1

        db.commit()
        logger.info(f"Saved message: session={session_id}, sender={sender}, seq={sequence_number}")

    async def _save_feedback_to_message(
        self,
        db,
        scenario_id: str,
        user_id: UUID,
        user_message: str,
        feedback: str
    ) -> None:
        """
        사용자 메시지에 피드백 저장

        Args:
            db: 데이터베이스 세션
            scenario_id: 시나리오 ID
            user_id: 사용자 ID
            user_message: 사용자 메시지 (매칭용)
            feedback: 피드백 JSON 문자열
        """
        # 해당 시나리오의 active 세션 찾기
        session = db.query(ConversationSession).filter(
            ConversationSession.scenario_id == UUID(scenario_id),
            ConversationSession.user_id == user_id,
            ConversationSession.status == 'active'
        ).order_by(ConversationSession.started_at.desc()).first()

        if not session:
            logger.warning(f"No active session found for scenario {scenario_id}")
            return

        # 해당 세션에서 메시지 텍스트가 일치하는 가장 최근 사용자 메시지 찾기
        message = db.query(ConversationMessage).filter(
            ConversationMessage.session_id == session.id,
            ConversationMessage.sender == 'user',
            ConversationMessage.message_text == user_message
        ).order_by(ConversationMessage.sequence_number.desc()).first()

        if message:
            message.feedback = feedback
            db.commit()
            logger.info(f"Saved feedback for message: session={session.id}, seq={message.sequence_number}")
        else:
            logger.warning(f"User message not found for feedback: {user_message[:50]}...")

    async def generate_hint(
        self,
        scenario_id: str,
        conversation_history: List[Dict[str, str]],
        last_ai_message: str,
        user_id: UUID,
        current_step: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        단계별 대화 힌트 생성

        시나리오 맥락과 현재 step의 terminology를 기반으로
        단어 → 구문 → 문장 순서의 단계별 힌트를 생성합니다.

        Args:
            scenario_id: 시나리오 ID
            conversation_history: 대화 히스토리
            last_ai_message: 마지막 AI 메시지
            user_id: 사용자 ID
            current_step: 현재 대화 단계 정보 (선택)
                - name: 단계 영문 식별자
                - title: 단계 한글 제목
                - guide: 단계 가이드
                - terminology: 이 단계에서 사용할 표현 리스트

        Returns:
            2단계 힌트:
                - targetExpression: 목표 표현
                - wordHints: 핵심 단어 리스트 (Level 0)
                - fullSentence: 완전한 문장 (Level 1)
                - explanation: 한국어 설명
                - stepInfo: 현재 단계 정보
        """
        try:
            # DB에서 시나리오 조회
            db = next(get_db())

            scenario = db.query(Scenario).filter(
                Scenario.id == UUID(scenario_id),
                Scenario.user_id == user_id
            ).first()

            if not scenario:
                raise ValueError(f"Scenario not found: {scenario_id}")

            # 시나리오 컨텍스트 구성
            scenario_context = {
                "title": scenario.title,
                "description": scenario.description,
                "scenario_text": scenario.scenario_text,
                "roles": scenario.roles,
                "required_terminology": scenario.required_terminology or [],
                "difficulty": scenario.difficulty,
                "language": scenario.language,
                "category": scenario.category
            }

            # HintAgent 호출
            hint_agent = HintAgent()
            result = await hint_agent.process(
                scenario_context=scenario_context,
                conversation_history=conversation_history,
                last_ai_message=last_ai_message,
                current_step=current_step
            )

            step_name = current_step.get('name', 'N/A') if current_step else 'N/A'
            logger.info(f"Generated stepped hint for scenario {scenario_id}, step: {step_name}")

            return result

        except Exception as e:
            logger.error(f"Error generating hint: {str(e)}")
            raise

    async def _get_session_used_terms(
        self,
        db,
        scenario_id: str,
        user_id: UUID
    ) -> set:
        """
        세션에서 이전에 사용된 모든 용어 조회

        Args:
            db: 데이터베이스 세션
            scenario_id: 시나리오 ID
            user_id: 사용자 ID

        Returns:
            이전에 사용된 용어들의 집합
        """
        # 해당 시나리오의 active 세션 찾기
        session = db.query(ConversationSession).filter(
            ConversationSession.scenario_id == UUID(scenario_id),
            ConversationSession.user_id == user_id,
            ConversationSession.status == 'active'
        ).order_by(ConversationSession.started_at.desc()).first()

        if not session:
            return set()

        # 해당 세션의 모든 메시지에서 detected_terms 수집
        messages = db.query(ConversationMessage).filter(
            ConversationMessage.session_id == session.id,
            ConversationMessage.sender == 'user',
            ConversationMessage.detected_terms.isnot(None)
        ).all()

        all_used_terms = set()
        for msg in messages:
            if msg.detected_terms:
                all_used_terms.update(msg.detected_terms)

        return all_used_terms
