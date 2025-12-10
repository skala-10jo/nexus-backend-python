"""
실시간 음성 번역 WebSocket API (Azure Speech + Azure Translator)

엔드포인트:
- WS /api/ai/voice/realtime: 실시간 음성 번역 WebSocket 연결

최적화:
- Azure Speech SDK 자동 언어 감지
- Azure Translator API 멀티 타겟 번역
- 프로젝트별 전문용어사전 후처리 (선택적)
- WebSocket 압축 (permessage-deflate)
- 비동기 처리 (asyncio)
- 에러 처리 강화
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Optional, List
from uuid import UUID
import json
import uuid
import logging
import time
import asyncio

from app.services.voice_translation_service import VoiceTranslationService
from app.database import SessionLocal
import azure.cognitiveservices.speech as speechsdk

# 로거 설정
logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter()

# 활성 WebSocket 연결 관리
active_connections: Dict[str, WebSocket] = {}

# 세션별 Session 인스턴스 관리
session_instances: Dict[str, 'VoiceTranslationSession'] = {}


# ============================================================
# 언어 코드 변환 헬퍼
# ============================================================
def bcp47_to_iso639(bcp47_code: str) -> str:
    """BCP-47 → ISO 639-1 변환 (ko-KR → ko)"""
    return bcp47_code.split('-')[0]


def iso639_to_bcp47(iso_code: str) -> str:
    """ISO 639-1 → BCP-47 변환 (ko → ko-KR)"""
    mapping = {
        "ko": "ko-KR",
        "en": "en-US",
        "ja": "ja-JP",
        "vi": "vi-VN",
        "zh": "zh-CN"
    }
    return mapping.get(iso_code, f"{iso_code}-XX")


# ============================================================
# 표준 WebSocket 메시지 프로토콜 래퍼
# ============================================================
async def send_standard_message(websocket: WebSocket, message_type: str, **kwargs):
    """
    표준 WebSocket 메시지 전송 래퍼 함수

    오직 5가지 메시지 타입만 허용:
    - recognizing: 중간 인식 결과
    - recognized: 최종 인식 결과 + 번역
    - error: 에러 메시지
    - end: 연결 종료
    - pong: Heartbeat 응답

    Args:
        websocket: WebSocket 연결
        message_type: 메시지 타입 (recognizing, recognized, error, end, pong만 허용)
        **kwargs: 메시지 데이터
    """
    ALLOWED_TYPES = {"recognizing", "recognized", "error", "end", "pong"}

    if message_type not in ALLOWED_TYPES:
        logger.warning(f"⚠️ 비표준 메시지 차단: type={message_type}")
        return

    message = {"type": message_type, **kwargs}
    await websocket.send_json(message)
    logger.debug(f"📤 표준 메시지 전송: type={message_type}, keys={list(kwargs.keys())}")


class VoiceTranslationSession:
    """실시간 음성 번역 세션 관리 (Azure Speech + Azure Translator + 용어집 후처리)"""

    def __init__(self, session_id: str, websocket: WebSocket, loop: asyncio.AbstractEventLoop):
        self.session_id = session_id
        self.websocket = websocket
        self.loop = loop  # 메인 이벤트 루프 저장 (스레드 간 비동기 호출용)

        # Service 초기화 (AI Agent 아키텍처 가이드 준수: API → Service → Agent)
        self.service = VoiceTranslationService()

        # Azure Speech 리소스
        self.recognizer: Optional[speechsdk.SpeechRecognizer] = None
        self.push_stream: Optional[speechsdk.audio.PushAudioInputStream] = None

        # 세션 설정
        self.selected_languages: List[str] = []  # BCP-47 코드 (ko-KR, en-US, ja-JP)

        # 프로젝트 및 용어집 설정 (전문용어사전 후처리용)
        self.project_id: Optional[UUID] = None
        self.db_session = None  # DB 세션 (프로젝트 선택 시 용어집 조회용)

        # 통계
        self.processed_chunks = 0
        self.total_translations = 0
        self.start_time = time.time()

        # WebSocket 연결 상태 플래그 (닫힌 후 메시지 전송 방지)
        self.is_closed = False

        logger.info(f"✅ VoiceTranslationSession 생성: session_id={session_id}")

    async def initialize(self, selected_languages: List[str], project_id: Optional[str] = None):
        """
        세션 초기화 및 Azure Speech 자동 언어 감지 설정

        Args:
            selected_languages: 선택된 언어 목록 (BCP-47 코드)
                예: ["ko-KR", "en-US", "ja-JP"]
            project_id: 프로젝트 ID (None이면 용어집 미적용)
        """
        if not selected_languages or len(selected_languages) < 2:
            await send_standard_message(
                self.websocket, "error",
                error="최소 2개 이상의 언어를 선택해야 합니다"
            )
            return

        self.selected_languages = selected_languages

        # 프로젝트 설정 (용어집 후처리용)
        if project_id:
            try:
                self.project_id = UUID(project_id)
                self.db_session = SessionLocal()
                logger.info(f"📚 프로젝트 연결: project_id={project_id} (용어집 후처리 활성화)")
            except ValueError as e:
                logger.warning(f"⚠️ 잘못된 project_id 형식: {project_id} - 용어집 미적용")
                self.project_id = None

        try:
            # Azure Speech 자동 언어 감지 스트림 생성 (Service를 통해 Agent 호출)
            logger.info(f"🔧 Azure Speech 자동 언어 감지 설정: {selected_languages}")
            self.recognizer, self.push_stream = await self.service.setup_stream_with_auto_detect(
                candidate_languages=selected_languages
            )

            # 이벤트 핸들러 등록 (필수 핸들러만)
            self.recognizer.recognizing.connect(self._on_recognizing)
            self.recognizer.recognized.connect(self._on_recognized)
            self.recognizer.canceled.connect(self._on_canceled)

            # 연속 인식 시작 (비동기)
            logger.info(f"🚀 Starting continuous recognition for session: {self.session_id}")
            self.recognizer.start_continuous_recognition_async()
            logger.info(f"✅ Continuous recognition started for session: {self.session_id}")

        except Exception as e:
            logger.error(f"❌ 세션 초기화 실패: {str(e)}", exc_info=True)
            await send_standard_message(
                self.websocket, "error",
                error=f"세션 초기화 실패: {str(e)}"
            )

    def _on_recognizing(self, evt: speechsdk.SpeechRecognitionEventArgs):
        """중간 인식 결과 핸들러 (recognizing)"""
        if self.is_closed:
            return
        logger.info(f"🎤 [Recognizing] reason={evt.result.reason}, text='{evt.result.text}'")
        if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
            text = evt.result.text
            if text and text.strip():

                # 별도 스레드에서 메인 이벤트 루프로 코루틴 스케줄링
                asyncio.run_coroutine_threadsafe(
                    send_standard_message(
                        self.websocket,
                        "recognizing",
                        text=text
                    ),
                    self.loop
                )

    def _on_recognized(self, evt: speechsdk.SpeechRecognitionEventArgs):
        """최종 인식 결과 핸들러 (recognized)"""
        if self.is_closed:
            return
        # NoMatch는 무시 (음성이 감지되지 않은 경우)
        if evt.result.reason == speechsdk.ResultReason.NoMatch:
            logger.debug(f"⚪ [NoMatch] 음성 감지 안됨")
            return
        logger.info(f"✅ [Recognized] reason={evt.result.reason}, text='{evt.result.text}'")
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = evt.result.text

            if not text or not text.strip():
                return

            # 자동 감지된 언어 추출
            detected_lang_bcp47 = evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
            ) or "ko-KR"

            # 별도 스레드에서 메인 이벤트 루프로 코루틴 스케줄링
            asyncio.run_coroutine_threadsafe(
                self._translate_and_send(text, detected_lang_bcp47, evt),
                self.loop
            )

    async def _translate_and_send(
        self,
        text: str,
        detected_lang_bcp47: str,
        evt: speechsdk.SpeechRecognitionEventArgs
    ):
        """
        번역 수행 및 recognized 메시지 전송

        Args:
            text: 인식된 텍스트
            detected_lang_bcp47: 자동 감지된 언어 (BCP-47)
            evt: 인식 이벤트
        """
        if self.is_closed:
            return
        try:
            # 감지된 언어를 제외한 타겟 언어 목록 생성
            target_langs_bcp47 = [
                lang for lang in self.selected_languages
                if lang != detected_lang_bcp47
            ]

            if not target_langs_bcp47:
                await send_standard_message(
                    self.websocket, "recognized",
                    text=text, detected_language=detected_lang_bcp47,
                    translations=[], confidence=0.9
                )
                return

            # BCP-47 → ISO 639-1 변환
            detected_lang_iso = bcp47_to_iso639(detected_lang_bcp47)
            target_langs_iso = [bcp47_to_iso639(lang) for lang in target_langs_bcp47]

            # Azure Translator 멀티 타겟 번역 (프로젝트가 있으면 용어집 후처리 및 용어 탐지 적용)
            detected_terms = []

            if self.project_id and self.db_session:
                # 용어집 후처리 포함 번역 + 용어 탐지
                translations, detected_terms = await self.service.translate_to_multiple_languages_with_glossary(
                    text=text,
                    source_lang=detected_lang_iso,
                    target_langs=target_langs_iso,
                    project_id=self.project_id,
                    db=self.db_session
                )
            else:
                # 기본 번역 (용어집 미적용)
                translations = await self.service.translate_to_multiple_languages(
                    text=text,
                    source_lang=detected_lang_iso,
                    target_langs=target_langs_iso
                )

            # ISO 639-1 → BCP-47 변환
            translations_bcp47 = [
                {"lang": iso639_to_bcp47(t["lang"]), "text": t["text"]}
                for t in translations
            ]

            # recognized 메시지 전송 (용어 탐지 결과 포함)
            await send_standard_message(
                self.websocket, "recognized",
                text=text, detected_language=detected_lang_bcp47,
                translations=translations_bcp47, confidence=0.9,
                detected_terms=detected_terms  # 탐지된 전문용어 추가
            )

        except Exception as e:
            logger.error(f"❌ 번역 실패: {str(e)}", exc_info=True)
            await send_standard_message(
                self.websocket, "error",
                error=f"번역 중 오류 발생: {str(e)}"
            )

    def _on_canceled(self, evt: speechsdk.SpeechRecognitionCanceledEventArgs):
        """인식 취소/에러 핸들러"""
        try:
            cancellation = getattr(evt, 'cancellation_details', None)
            if cancellation:
                cancel_reason = getattr(cancellation, 'reason', 'Unknown')
                if cancel_reason == speechsdk.CancellationReason.EndOfStream:
                    return  # 정상 종료

                error_details = getattr(cancellation, 'error_details', '')
                if error_details:
                    asyncio.run_coroutine_threadsafe(
                        send_standard_message(
                            self.websocket, "error",
                            error=f"음성 인식 오류: {error_details}"
                        ),
                        self.loop
                    )
        except Exception as e:
            logger.error(f"_on_canceled error: {str(e)}")

    async def process_audio_chunk(self, audio_bytes: bytes):
        """
        오디오 청크를 Azure Speech PushStream에 전송

        Args:
            audio_bytes: PCM 16kHz 16bit Mono 오디오 데이터
        """
        if not self.push_stream:
            logger.warning("⚠️ PushStream이 초기화되지 않았습니다")
            return

        if not audio_bytes or len(audio_bytes) == 0:
            logger.debug("⚪ 빈 오디오 청크 무시")
            return

        try:
            # Azure Speech PushStream에 PCM 바이너리 쓰기
            self.push_stream.write(audio_bytes)
            self.processed_chunks += 1

            # 첫 5개 청크만 로그 출력 (디버깅용)
            if self.processed_chunks <= 5:
                logger.info(f"📤 오디오 청크 #{self.processed_chunks}: {len(audio_bytes)} bytes")

        except Exception as e:
            logger.error(f"❌ 오디오 청크 처리 실패: {str(e)}", exc_info=True)
            await send_standard_message(
                self.websocket, "error",
                error=f"오디오 처리 중 오류 발생: {str(e)}"
            )

    async def cleanup(self):
        """세션 정리 (Azure Speech 리소스 및 DB 세션 해제)"""
        self.is_closed = True  # 먼저 플래그 설정하여 콜백 차단
        try:
            if self.recognizer:
                self.recognizer.stop_continuous_recognition()
                logger.info(f"🛑 Azure Speech 연속 인식 중지: session_id={self.session_id}")

            if self.push_stream:
                self.push_stream.close()
                logger.info(f"🔒 PushStream 닫힘: session_id={self.session_id}")

            # DB 세션 정리 (프로젝트 선택 시 생성된 경우)
            if self.db_session:
                self.db_session.close()
                logger.info(f"🔒 DB 세션 닫힘: session_id={self.session_id}")

        except Exception as e:
            logger.error(f"❌ 세션 정리 실패: {str(e)}", exc_info=True)

    def get_stats(self) -> Dict:
        """세션 통계 반환"""
        elapsed_time = time.time() - self.start_time
        return {
            "session_id": self.session_id,
            "elapsed_time": round(elapsed_time, 2),
            "processed_chunks": self.processed_chunks,
            "total_translations": self.total_translations,
            "selected_languages": self.selected_languages,
            "project_id": str(self.project_id) if self.project_id else None,
            "glossary_enabled": self.project_id is not None
        }


@router.websocket("/api/ai/voice/realtime")
async def voice_translation_websocket(websocket: WebSocket):
    """
    실시간 음성 번역 WebSocket 엔드포인트

    클라이언트와 WebSocket 연결을 맺고 실시간으로 음성 번역을 수행합니다.

    프로토콜:
    1. 클라이언트 → 서버 (JSON): {"selected_languages": ["ko-KR", "en-US", "ja-JP"], "project_id": "uuid" (선택)}
    2. 클라이언트 → 서버 (Binary): 오디오 청크 (WebM/Opus, 16kHz, mono)
    3. 서버 → 클라이언트 (JSON):
       - {"type": "recognizing", "text": "..."}
       - {"type": "recognized", "text": "...", "detected_language": "ko-KR", "translations": [...]}
       - {"type": "error", "error": "..."}
       - {"type": "end"}

    전문용어사전 적용:
    - project_id를 전달하면 해당 프로젝트에 연결된 문서의 용어집을 후처리로 적용합니다.
    - project_id가 없으면 기본 Azure Translator 번역만 수행합니다.
    """

    logger.info("🌐 [WS-Backend] WebSocket 연결 요청 받음")

    # WebSocket 연결 수락
    await websocket.accept()
    logger.info("✅ [WS-Backend] WebSocket 연결 수락 완료")

    # 세션 ID 생성
    session_id = str(uuid.uuid4())
    session: Optional[VoiceTranslationSession] = None

    # 활성 연결 등록
    active_connections[session_id] = websocket

    logger.info(f"✅ [WS-Backend] WebSocket 연결됨: session_id={session_id}")

    try:
        # 메시지 수신 루프
        while True:
            # 클라이언트로부터 메시지 수신
            message = await websocket.receive()

            # JSON 메시지 처리 (세션 초기화)
            if "text" in message:
                logger.info(f"📥 [WS-Backend] JSON 메시지 수신: {message['text']}")
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError as e:
                    logger.error(f"❌ [WS-Backend] JSON 파싱 에러: {e}")
                    await send_standard_message(websocket, "error", error="잘못된 JSON 형식입니다")
                    continue

                # selected_languages로 세션 초기화
                if "selected_languages" in data:
                    selected_languages = data["selected_languages"]
                    project_id = data.get("project_id")  # 프로젝트 ID (선택)
                    logger.info(f"📝 [WS-Backend] 세션 초기화 요청: {selected_languages}, project={project_id}")

                    # 세션 생성 및 초기화 (현재 이벤트 루프 전달 - SDK 콜백의 스레드 안전성 확보)
                    loop = asyncio.get_event_loop()
                    session = VoiceTranslationSession(session_id, websocket, loop)
                    await session.initialize(selected_languages, project_id=project_id)

                    # 세션 저장
                    session_instances[session_id] = session
                    glossary_status = "✅ 용어집 활성화" if project_id else "⚪ 용어집 미사용"
                    logger.info(f"✅ [WS-Backend] 세션 초기화 완료: session_id={session_id}, {glossary_status}")

                # Heartbeat ping 처리 - pong 응답
                elif data.get("type") == "ping":
                    logger.debug(f"💓 [WS-Backend] Heartbeat ping 수신: session_id={session_id}")
                    await send_standard_message(websocket, "pong")

                # 종료 메시지
                elif data.get("type") == "end":
                    logger.info(f"🔚 클라이언트 종료 요청: session_id={session_id}")
                    await send_standard_message(websocket, "end")
                    break

                else:
                    # 알 수 없는 메시지는 경고만 로그하고 무시 (연결 유지)
                    logger.warning(f"⚠️ [WS-Backend] 알 수 없는 메시지 타입: {data}")

            # Binary 메시지 처리 (오디오 청크)
            elif "bytes" in message:
                audio_bytes = message["bytes"]

                if not session:
                    await send_standard_message(
                        websocket, "error",
                        error="세션이 초기화되지 않았습니다. selected_languages를 먼저 보내세요"
                    )
                    continue

                if audio_bytes and len(audio_bytes) > 0:
                    # 원본 바이너리를 Azure Speech PushStream에 전송
                    await session.process_audio_chunk(audio_bytes)

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket 연결 종료: session_id={session_id}")

    except Exception as e:
        logger.error(f"❌ WebSocket 에러: session_id={session_id}, error={str(e)}", exc_info=True)
        try:
            await send_standard_message(websocket, "error", error=f"서버 오류: {str(e)}")
        except:
            pass

    finally:
        # 연결 정리
        if session_id in active_connections:
            del active_connections[session_id]

        if session_id in session_instances:
            session = session_instances[session_id]
            await session.cleanup()
            del session_instances[session_id]

        logger.info(f"🧹 세션 정리 완료: session_id={session_id}")


@router.get("/api/ai/voice/sessions")
async def get_active_sessions():
    """
    현재 활성 세션 목록 조회 (관리/모니터링용)

    Returns:
        {
            "success": true,
            "data": {
                "active_sessions": 3,
                "sessions": [...]
            }
        }
    """
    sessions_info = []

    for session_id, session in session_instances.items():
        sessions_info.append(session.get_stats())

    return {
        "success": True,
        "data": {
            "active_sessions": len(active_connections),
            "sessions": sessions_info
        }
    }
