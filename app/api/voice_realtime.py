"""
실시간 음성 인식 WebSocket API (Azure Speech)

엔드포인트:
- WS /api/ai/voice/realtime: 실시간 음성 인식 WebSocket 연결

최적화:
- 단일 언어 STT (언어 감지 없음, 빠른 응답)
- WebSocket 압축 (permessage-deflate)
- 비동기 처리 (asyncio)
- 에러 처리 강화
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Optional
import json
import uuid
import logging
import time
import asyncio

from app.services.voice_translation_service import VoiceTranslationService
import azure.cognitiveservices.speech as speechsdk

# 로거 설정
logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter()

# 활성 WebSocket 연결 관리
active_connections: Dict[str, WebSocket] = {}

# 세션별 Session 인스턴스 관리
session_instances: Dict[str, 'VoiceRealtimeSession'] = {}


# ============================================================
# 표준 WebSocket 메시지 프로토콜 래퍼
# ============================================================
async def send_standard_message(websocket: WebSocket, message_type: str, **kwargs):
    """
    표준 WebSocket 메시지 전송 래퍼 함수

    오직 4가지 메시지 타입만 허용:
    - recognizing: 중간 인식 결과
    - recognized: 최종 인식 결과 + 번역
    - error: 에러 메시지
    - end: 연결 종료

    Args:
        websocket: WebSocket 연결
        message_type: 메시지 타입 (recognizing, recognized, error, end만 허용)
        **kwargs: 메시지 데이터
    """
    ALLOWED_TYPES = {"recognizing", "recognized", "error", "end"}

    if message_type not in ALLOWED_TYPES:
        logger.warning(f"⚠️ 비표준 메시지 차단: type={message_type}")
        return

    message = {"type": message_type, **kwargs}
    await websocket.send_json(message)
    logger.debug(f"📤 표준 메시지 전송: type={message_type}, keys={list(kwargs.keys())}")


class VoiceRealtimeSession:
    """실시간 음성 인식 세션 관리 (단일 언어 STT)"""

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
        self.language: str = "en-US"  # 인식 언어 (BCP-47)

        # 통계
        self.processed_chunks = 0
        self.start_time = time.time()

        logger.info(f"✅ VoiceRealtimeSession 생성: session_id={session_id}")

    async def initialize(self, language: str):
        """
        세션 초기화 및 Azure Speech 단일 언어 설정

        Args:
            language: 인식 언어 (BCP-47 코드, 예: "en-US")
        """
        if not language:
            await send_standard_message(
                self.websocket, "error",
                error="언어가 지정되지 않았습니다"
            )
            return

        self.language = language

        try:
            # Azure Speech 단일 언어 스트림 생성 (Service를 통해 Agent 호출)
            logger.info(f"🔧 Azure Speech 단일 언어 설정: {language}")
            self.recognizer, self.push_stream = await self.service.setup_stream_single_language(
                language=language
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
        # NoMatch는 무시 (음성이 감지되지 않은 경우)
        if evt.result.reason == speechsdk.ResultReason.NoMatch:
            logger.debug(f"⚪ [NoMatch] 음성 감지 안됨")
            return
        logger.info(f"✅ [Recognized] reason={evt.result.reason}, text='{evt.result.text}'")
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = evt.result.text

            if not text or not text.strip():
                return

            # 별도 스레드에서 메인 이벤트 루프로 코루틴 스케줄링
            # 번역 없이 바로 recognized 메시지 전송
            asyncio.run_coroutine_threadsafe(
                send_standard_message(
                    self.websocket, "recognized",
                    text=text
                ),
                self.loop
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
        """세션 정리 (Azure Speech 리소스 해제)"""
        try:
            if self.recognizer:
                self.recognizer.stop_continuous_recognition()
                logger.info(f"🛑 Azure Speech 연속 인식 중지: session_id={self.session_id}")

            if self.push_stream:
                self.push_stream.close()
                logger.info(f"🔒 PushStream 닫힘: session_id={self.session_id}")

        except Exception as e:
            logger.error(f"❌ 세션 정리 실패: {str(e)}", exc_info=True)

    def get_stats(self) -> Dict:
        """세션 통계 반환"""
        elapsed_time = time.time() - self.start_time
        return {
            "session_id": self.session_id,
            "elapsed_time": round(elapsed_time, 2),
            "processed_chunks": self.processed_chunks,
            "language": self.language
        }


@router.websocket("/api/ai/voice/realtime")
async def voice_realtime_websocket(websocket: WebSocket):
    """
    실시간 음성 인식 WebSocket 엔드포인트 (단일 언어)

    클라이언트와 WebSocket 연결을 맺고 실시간으로 음성 인식을 수행합니다.

    프로토콜:
    1. 클라이언트 → 서버 (JSON): {"language": "en-US"}
    2. 클라이언트 → 서버 (Binary): 오디오 청크 (PCM, 16kHz, mono)
    3. 서버 → 클라이언트 (JSON):
       - {"type": "recognizing", "text": "..."}
       - {"type": "recognized", "text": "..."}
       - {"type": "error", "error": "..."}
       - {"type": "end"}
    """

    logger.info("🌐 [WS-Backend] WebSocket 연결 요청 받음")

    # WebSocket 연결 수락
    await websocket.accept()
    logger.info("✅ [WS-Backend] WebSocket 연결 수락 완료")

    # 세션 ID 생성
    session_id = str(uuid.uuid4())
    session: Optional[VoiceRealtimeSession] = None

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

                # language로 세션 초기화 (단일 언어)
                if "language" in data:
                    language = data["language"]
                    logger.info(f"📝 [WS-Backend] 세션 초기화 요청: {language}")

                    # 세션 생성 및 초기화 (현재 이벤트 루프 전달 - SDK 콜백의 스레드 안전성 확보)
                    loop = asyncio.get_event_loop()
                    session = VoiceRealtimeSession(session_id, websocket, loop)
                    await session.initialize(language)

                    # 세션 저장
                    session_instances[session_id] = session
                    logger.info(f"✅ [WS-Backend] 세션 초기화 완료: session_id={session_id}")

                # 종료 메시지
                elif data.get("type") == "end":
                    logger.info(f"🔚 클라이언트 종료 요청: session_id={session_id}")
                    await send_standard_message(websocket, "end")
                    break

                else:
                    await send_standard_message(
                        websocket, "error",
                        error=f"알 수 없는 메시지: {data}"
                    )

            # Binary 메시지 처리 (오디오 청크)
            elif "bytes" in message:
                audio_bytes = message["bytes"]

                if not session:
                    await send_standard_message(
                        websocket, "error",
                        error="세션이 초기화되지 않았습니다. language를 먼저 보내세요"
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
