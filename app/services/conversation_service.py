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
            from app.models.conversation import ConversationSession

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
                    "initialMessage": None,  # 기존 대화가 있으므로 초기 메시지 없음
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
        user_id: UUID
    ) -> Dict[str, Any]:
        """
        사용자 메시지 전송 및 AI 응답 생성

        Args:
            scenario_id: 시나리오 ID
            user_message: 사용자 메시지
            conversation_history: 대화 히스토리 (프론트에서 전달, 백업용)
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

            # AI 메시지 DB에 저장
            await self._save_message(
                db=db,
                session_id=session.id,
                sender="ai",
                message_text=ai_message,
                detected_terms=detected_terms,
                sequence_number=next_seq + 1
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
- Start with a brief, friendly greeting (1-2 sentences maximum)
- Include a casual greeting AND subtly hint at the scenario context
- Example: "Hi! How's it going? Ready for our meeting about the project?" or "Hey! How are you? Excited to discuss the proposal today?"
- Keep it natural and conversational
- Respond in {scenario.language} language
- Be friendly and welcoming
- Use realistic contexts based on today's date"""

            # GPT-4o 호출
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Start with a friendly greeting (1-2 sentences). First sentence: casual greeting. Second sentence (optional): subtly reference the scenario context. Keep it brief and natural."}
                ],
                temperature=0.7,
                max_tokens=80
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
- CRITICAL: Keep responses extremely brief - reading time must be under 7 seconds (15-20 words maximum)
- Use ONLY 1-2 short sentences (never more than 2 sentences)
- Mix casual small talk naturally with business topics (like real conversations)
- Occasionally ask about personal things (weekend, lunch, weather) before/after business talk
- Use the required terminology naturally when discussing business
- Respond in {scenario.language} language
- Be encouraging and supportive for language practice
- If user makes grammatical errors, gently incorporate corrections in your response
- Use realistic dates and times based on today's context
- REMINDER: Think "quick chat message" not "email" - be conversational and concise"""

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
                max_tokens=60  # 7초 이내 읽기 시간을 위해 60 토큰으로 제한
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

    async def generate_message_feedback(
        self,
        scenario_id: str,
        user_message: str,
        detected_terms: List[str],
        user_id: UUID,
        audio_data: str = None
    ) -> Dict[str, Any]:
        """
        사용자 메시지에 대한 피드백 생성

        Args:
            scenario_id: 시나리오 ID
            user_message: 사용자 메시지
            detected_terms: 감지된 전문용어
            user_id: 사용자 ID
            audio_data: Base64 인코딩된 오디오 데이터 (선택)

        Returns:
            피드백 (문법 교정, 용어 사용, 제안, 점수, 상세 발음 정보)
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

            # Azure Pronunciation Assessment 수행 (오디오 데이터가 있는 경우)
            pronunciation_details = None
            if audio_data:
                try:
                    import base64
                    from agent.pronunciation.pronunciation_agent import PronunciationAssessmentAgent

                    logger.info("🎤 Running Azure Pronunciation Assessment...")
                    audio_bytes = base64.b64decode(audio_data)

                    # 발음 평가 수행
                    agent = PronunciationAssessmentAgent.get_instance()
                    pronunciation_result = await agent.assess_pronunciation(
                        audio_data=audio_bytes,
                        reference_text=user_message,
                        language='en-US',
                        granularity='Phoneme'
                    )

                    # 상세 발음 정보 저장
                    pronunciation_details = {
                        'pronunciation_score': pronunciation_result['pronunciation_score'],
                        'accuracy_score': pronunciation_result['accuracy_score'],
                        'fluency_score': pronunciation_result['fluency_score'],
                        'prosody_score': pronunciation_result['prosody_score'],
                        'completeness_score': pronunciation_result['completeness_score'],
                        'words': pronunciation_result['words']
                    }

                    logger.info(f"✅ Pronunciation assessment completed: {pronunciation_result['pronunciation_score']:.1f}")

                except Exception as e:
                    logger.error(f"Pronunciation assessment failed: {str(e)}", exc_info=True)
                    # 발음 평가 실패해도 피드백은 계속 생성

            # GPT-4o로 피드백 생성
            system_prompt = f"""You are an expert language tutor providing feedback on business conversation practice in KOREAN language.

Scenario Context:
- Title: {scenario.title}
- Description: {scenario.description}
- Situation: {scenario.scenario_text}
- User's Role: {scenario.roles.get('user', 'User')}
- AI's Role: {scenario.roles.get('ai', 'AI')}
- Language: {scenario.language}
- Difficulty: {scenario.difficulty}
- Required Terminology: {', '.join(scenario.required_terminology)}

IMPORTANT FEEDBACK RULES:
1. ALL feedback must be written in KOREAN (한글)
2. Grammar corrections: Explain the issue in Korean, then suggest the English correction
   - Example: "시제가 틀렸어요. 'I go yesterday' 대신 'I went yesterday'라고 해야 해요."
3. Suggestions: Provide Korean explanation with English phrase recommendations
   - Example: "더 자연스러운 표현으로는 'Could you please...' 또는 'Would you mind...'를 사용해보세요."
   - If the message is very poor, provide a complete sentence example with "이런 식으로 해보세요"
   - Consider the user's role and situation when making suggestions (e.g., formality, tone, context appropriateness)
4. Scoring System (1-10):
   - Grammar (30%): Correct sentence structure, tense, articles
   - Vocabulary (25%): Word choice, natural expressions, terminology usage
   - Fluency (25%): Natural flow, politeness, business context appropriateness
   - Pronunciation (20%): Estimate clarity and correctness of pronunciation based on word choice and sentence complexity
   - 9-10: Excellent, native-level
   - 7-8: Good, minor improvements needed
   - 5-6: Fair, several issues to fix
   - 3-4: Poor, major improvements needed
   - 1-2: Very poor, needs complete revision

Provide feedback in JSON format with Korean text."""

            # pronunciation_details가 있으면 추가 정보 제공
            pronunciation_info = ""
            if pronunciation_details:
                pronunciation_info = f"""

Azure Pronunciation Assessment Results:
- Overall Pronunciation Score: {pronunciation_details['pronunciation_score']:.1f}/100
- Accuracy Score: {pronunciation_details['accuracy_score']:.1f}/100
- Fluency Score: {pronunciation_details['fluency_score']:.1f}/100
- Prosody Score (억양/강세): {pronunciation_details['prosody_score']:.1f}/100
- Completeness Score: {pronunciation_details['completeness_score']:.1f}/100

Words with pronunciation issues (accuracy < 80):
{chr(10).join([f"- '{word['word']}': {word['accuracy_score']:.1f}/100" for word in pronunciation_details['words'] if word['accuracy_score'] < 80][:5]) if any(w['accuracy_score'] < 80 for w in pronunciation_details['words']) else '(모든 단어가 잘 발음되었습니다)'}

Based on these scores, provide specific feedback on:
1. Prosody (운율): If prosody_score < 80, explain issues with intonation (억양), stress (강세), or rhythm (리듬)
2. Problematic words: Mention specific words with low accuracy scores
3. Overall pronunciation improvement tips"""

            user_prompt = f"""User's message: "{user_message}"

Detected terminology used: {', '.join(detected_terms) if detected_terms else 'None'}{pronunciation_info}

Provide feedback in this exact JSON format (ALL TEXT IN KOREAN):
{{
  "grammar_corrections": [
    "시제 문제: 'I was go'는 틀렸어요. 'I went' 또는 'I was going'이라고 해야 해요."
  ],
  "terminology_usage": {{
    "used": ["term1", "term2"],
    "missed": ["term3"]
  }},
  "suggestions": [
    "더 공손한 표현으로는 'Could you please...' 또는 'Would you mind...'를 사용해보세요."
  ],
  "pronunciation_feedback": [
    "억양 (Prosody): 문장 끝에서 억양이 올라가야 하는데 평평하게 발음했어요. 질문할 때는 끝을 올려서 말해보세요.",
    "강세 (Stress): 'important'는 두 번째 음절 '-por-'에 강세를 주어야 해요. 'im-POR-tant'처럼 발음해보세요.",
    "'needed' 단어의 /d/ 소리가 정확하지 않아요. 혀끝을 윗니 뒤에 대고 'd' 소리를 내보세요."
  ],
  "score": 7,
  "score_breakdown": {{
    "grammar": 6,
    "vocabulary": 8,
    "fluency": 7,
    "pronunciation": 7
  }}
}}

IMPORTANT:
- If pronunciation assessment data is provided, MUST include "pronunciation_feedback" array with specific tips
- If prosody_score < 80: provide feedback on intonation (억양), stress (강세), or rhythm (리듬)
- If any words have low accuracy: mention those specific words and how to improve
- All explanations in KOREAN (한글), but include English words/corrections within the Korean text"""

            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500,
                response_format={"type": "json_object"}
            )

            # JSON 파싱
            import json
            feedback = json.loads(response.choices[0].message.content)

            # 발음 상세 정보 추가 (있는 경우)
            if pronunciation_details:
                feedback['pronunciation_details'] = pronunciation_details

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
        message: str,
        target_language: str = "ko"
    ) -> str:
        """
        메시지 번역 (GPT-4o 사용)

        Args:
            message: 번역할 메시지
            target_language: 목표 언어 (기본값: "ko" 한국어)

        Returns:
            번역된 텍스트
        """
        try:
            language_names = {
                "ko": "Korean (한국어)",
                "en": "English",
                "ja": "Japanese (日本語)",
                "zh": "Chinese (中文)",
                "vi": "Vietnamese (Tiếng Việt)"
            }

            target_lang_name = language_names.get(target_language, "Korean (한국어)")

            system_prompt = f"""You are a professional translator.
Translate the given text to {target_lang_name}.
Provide ONLY the translated text without any explanations or additional comments."""

            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.3,
                max_tokens=500
            )

            translated_text = response.choices[0].message.content.strip()
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
            from app.models.conversation import ConversationSession

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
            from app.models.conversation import ConversationSession, ConversationMessage

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
        from app.models.conversation import ConversationSession
        from app.models.scenario import Scenario

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
        from app.models.conversation import ConversationMessage, ConversationSession

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
        from app.models.conversation import ConversationSession, ConversationMessage

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
