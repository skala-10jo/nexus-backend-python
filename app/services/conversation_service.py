"""
회화 연습 서비스
GPT-4o를 이용한 시나리오 기반 대화 생성
"""
import json
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
                        "requiredTerms": scenario.required_terminology,
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
                    "requiredTerms": scenario.required_terminology,
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

            # 난이도별 지침
            difficulty_instructions = {
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

            # 시스템 프롬프트
            system_prompt = f"""당신은 비즈니스 회화 연습 시나리오에 참여하고 있습니다.

오늘 날짜: {current_date}

시나리오: {scenario.title}
설명: {scenario.description}
상황: {scenario.scenario_text}

당신의 역할: {scenario.roles.get('ai', 'AI')}
사용자 역할: {scenario.roles.get('user', 'User')}

언어: {scenario.language}
난이도: {scenario.difficulty}

나중에 자연스럽게 사용할 필수 전문용어: {', '.join(scenario.required_terminology)}

난이도별 지침:
{difficulty_instructions.get(scenario.difficulty, difficulty_instructions['intermediate'])}

기본 지침:
- 짧고 친근한 인사로 시작하세요 (최대 1-2문장)
- 캐주얼한 인사와 함께 시나리오 맥락을 은근히 암시하세요
- 예시: "안녕하세요! 잘 지내셨어요? 프로젝트 미팅 준비되셨나요?" 또는 "Hi! How's it going? Ready for our meeting about the project?"
- 자연스럽고 대화체로 작성하세요
- {scenario.language} 언어로 응답하세요
- 친근하고 환영하는 분위기를 만드세요
- 오늘 날짜를 기반으로 현실적인 맥락을 사용하세요"""

            # GPT-4o 호출
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "친근한 인사로 시작하세요 (1-2문장). 첫 문장: 캐주얼한 인사. 두 번째 문장 (선택): 시나리오 맥락을 은근히 언급. 간결하고 자연스럽게 작성하세요."}
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

            # 난이도별 대화 스타일 지침
            conversation_style = {
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

            # 시스템 프롬프트
            system_prompt = f"""당신은 비즈니스 회화 연습 시나리오에 참여하고 있습니다.

오늘 날짜: {current_date}
현재 시간: {current_time}

시나리오: {scenario.title}
설명: {scenario.description}
상황: {scenario.scenario_text}

당신의 역할: {scenario.roles.get('ai', 'AI')}
사용자 역할: {scenario.roles.get('user', 'User')}

언어: {scenario.language}
난이도: {scenario.difficulty}

자연스럽게 사용할 필수 전문용어: {', '.join(scenario.required_terminology)}

난이도별 대화 스타일:
{conversation_style.get(scenario.difficulty, conversation_style['intermediate'])}

기본 지침:
- 중요: 응답을 매우 간결하게 유지하세요 - 읽는 시간이 7초 이내여야 합니다 (최대 15-20 단어)
- 1-2개의 짧은 문장만 사용하세요 (절대 2문장 이상 사용하지 마세요)
- 실제 대화처럼 캐주얼한 스몰토크와 비즈니스 주제를 자연스럽게 섞으세요
- 가끔은 비즈니스 대화 전후에 개인적인 것들(주말, 점심, 날씨 등)을 물어보세요
- 비즈니스를 논의할 때 필수 전문용어를 자연스럽게 사용하세요
- {scenario.language} 언어로 응답하세요
- 언어 연습에 대해 격려하고 지원적으로 대하세요
- 사용자가 문법 오류를 범하면, 응답에서 부드럽게 교정을 포함하세요
- 오늘 맥락을 기반으로 현실적인 날짜와 시간을 사용하세요
- 알림: "빠른 채팅 메시지"로 생각하세요, "이메일"이 아닙니다 - 대화체이고 간결하게"""

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
            system_prompt = f"""당신은 비즈니스 회화 연습에 대한 피드백을 한글로 제공하는 전문 언어 튜터입니다.

시나리오 맥락:
- 제목: {scenario.title}
- 설명: {scenario.description}
- 상황: {scenario.scenario_text}
- 사용자 역할: {scenario.roles.get('user', 'User')}
- AI 역할: {scenario.roles.get('ai', 'AI')}
- 언어: {scenario.language}
- 난이도: {scenario.difficulty}
- 필수 전문용어: {', '.join(scenario.required_terminology)}

중요한 피드백 규칙:
1. 모든 피드백은 반드시 한글로 작성해야 합니다
2. 문법 교정: 한글로 문제를 설명한 후, 영어 교정을 제안하세요
   - 예시: "시제가 틀렸어요. 'I go yesterday' 대신 'I went yesterday'라고 해야 해요."
3. 제안: 한글 설명과 함께 영어 표현 추천을 제공하세요
   - 예시: "더 자연스러운 표현으로는 'Could you please...' 또는 'Would you mind...'를 사용해보세요."
   - 메시지가 매우 부족하다면, "이런 식으로 해보세요"와 함께 완전한 문장 예시를 제공하세요
   - 제안할 때 사용자의 역할과 상황을 고려하세요 (예: 격식, 어조, 맥락 적절성)
4. 난이도별 채점 기준 (1-10):

**{scenario.difficulty.upper()} 난이도 기준:**

{'BEGINNER 기준:' if scenario.difficulty == 'beginner' else ''}
{'- 문법 (30%): 기본 문장 구조, 현재/과거 시제만 검사, 간단한 관사 사용' if scenario.difficulty == 'beginner' else ''}
{'- 어휘 (25%): 기본 일상 어휘 사용 여부, 복잡한 표현 요구하지 않음' if scenario.difficulty == 'beginner' else ''}
{'- 유창성 (25%): 의사소통 가능 여부에 집중, 완벽한 문장 구조 요구하지 않음' if scenario.difficulty == 'beginner' else ''}
{'- 발음 (20%): 이해 가능한 수준이면 충분' if scenario.difficulty == 'beginner' else ''}
{'- 평가 기준: 의미 전달 가능하면 7점 이상, 기본 문법만 맞아도 긍정적 평가' if scenario.difficulty == 'beginner' else ''}

{'INTERMEDIATE 기준:' if scenario.difficulty == 'intermediate' else ''}
{'- 문법 (30%): 다양한 시제, 관사, 전치사 정확도' if scenario.difficulty == 'intermediate' else ''}
{'- 어휘 (25%): 비즈니스 용어 적절한 사용, 자연스러운 표현' if scenario.difficulty == 'intermediate' else ''}
{'- 유창성 (25%): 자연스러운 흐름, 맥락 적절성, 비즈니스 예절' if scenario.difficulty == 'intermediate' else ''}
{'- 발음 (20%): 명확하고 자신감 있는 발음' if scenario.difficulty == 'intermediate' else ''}
{'- 평가 기준: 일반적인 비즈니스 소통 가능하면 7점 이상' if scenario.difficulty == 'intermediate' else ''}

{'ADVANCED 기준:' if scenario.difficulty == 'advanced' else ''}
{'- 문법 (30%): 완벽한 문법, 복잡한 구조, 미묘한 뉘앙스' if scenario.difficulty == 'advanced' else ''}
{'- 어휘 (25%): 전문 용어 정확한 사용, 관용구, 세련된 표현' if scenario.difficulty == 'advanced' else ''}
{'- 유창성 (25%): 원어민 수준의 자연스러움, 전략적 커뮤니케이션' if scenario.difficulty == 'advanced' else ''}
{'- 발음 (20%): 원어민에 가까운 억양과 리듬' if scenario.difficulty == 'advanced' else ''}
{'- 평가 기준: 원어민 수준 요구, 9-10점은 전문가 수준만 가능' if scenario.difficulty == 'advanced' else ''}

점수 가이드:
   - 9-10: 탁월함, 해당 난이도에서 최고 수준
   - 7-8: 좋음, 해당 난이도 목표 달성
   - 5-6: 보통, 개선 필요
   - 3-4: 부족함, 주요 개선 필요
   - 1-2: 매우 부족함, 기본부터 다시

한글 텍스트로 JSON 형식의 피드백을 제공하세요."""

            # pronunciation_details가 있으면 추가 정보 제공
            pronunciation_info = ""
            if pronunciation_details:
                pronunciation_info = f"""

Azure 발음 평가 결과:
- 전체 발음 점수: {pronunciation_details['pronunciation_score']:.1f}/100
- 정확도 점수: {pronunciation_details['accuracy_score']:.1f}/100
- 유창성 점수: {pronunciation_details['fluency_score']:.1f}/100
- 운율 점수 (억양/강세): {pronunciation_details['prosody_score']:.1f}/100
- 완성도 점수: {pronunciation_details['completeness_score']:.1f}/100

발음 문제가 있는 단어들 (정확도 < 80):
{chr(10).join([f"- '{word['word']}': {word['accuracy_score']:.1f}/100" for word in pronunciation_details['words'] if word['accuracy_score'] < 80][:5]) if any(w['accuracy_score'] < 80 for w in pronunciation_details['words']) else '(모든 단어가 잘 발음되었습니다)'}

이 점수를 기반으로 다음에 대한 구체적인 피드백을 제공하세요:
1. 운율 (Prosody): prosody_score < 80인 경우, 억양(intonation), 강세(stress), 또는 리듬(rhythm) 문제를 설명하세요
2. 문제 단어: 낮은 정확도 점수를 받은 특정 단어를 언급하세요
3. 전반적인 발음 개선 팁"""

            # 미사용 용어 계산
            required_terms = scenario.required_terminology or []
            missed_terms = [term for term in required_terms if term.lower() not in user_message.lower()]

            user_prompt = f"""사용자 메시지: "{user_message}"

전문용어 분석:
- 필수 전문용어: {', '.join(required_terms) if required_terms else '없음'}
- 사용한 용어: {', '.join(detected_terms) if detected_terms else '없음'}
- 미사용 용어: {', '.join(missed_terms) if missed_terms else '없음'}
{pronunciation_info}

다음의 정확한 JSON 형식으로 피드백을 제공하세요 (모든 텍스트 한글로):
{{
  "grammar_corrections": [
    "시제 문제: 'I was go'는 틀렸어요. 'I went' 또는 'I was going'이라고 해야 해요."
  ],
  "terminology_usage": {{
    "used": {json.dumps(detected_terms or [], ensure_ascii=False)},
    "missed": {json.dumps(missed_terms, ensure_ascii=False)},
    "feedback": "필수 용어 사용에 대한 피드백을 여기에 작성하세요"
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

중요:
- terminology_usage의 used와 missed 배열은 위에서 제공한 값을 그대로 사용하세요
- terminology_usage.feedback에는 용어 사용에 대한 구체적인 피드백을 한글로 작성하세요
- 발음 평가 데이터가 제공되면, 구체적인 팁과 함께 "pronunciation_feedback" 배열을 반드시 포함해야 합니다
- prosody_score < 80인 경우: 억양(intonation), 강세(stress), 또는 리듬(rhythm)에 대한 피드백을 제공하세요
- 낮은 정확도를 가진 단어가 있다면: 해당 특정 단어와 개선 방법을 언급하세요
- 모든 설명은 한글로 작성하되, 한글 텍스트 안에 영어 단어/교정을 포함하세요"""

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

            system_prompt = f"""당신은 전문 번역가입니다.
주어진 텍스트를 {target_lang_name}로 번역하세요.
설명이나 추가 코멘트 없이 번역된 텍스트만 제공하세요."""

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
