"""
단일 언어 STT 전용 WebSocket API (번역 없음, 고성능)

엔드포인트:
- WS /api/ai/voice/stt-stream: 단일 언어 실시간 음성 인식

용도:
- 실무 시나리오 회화연습 페이지
- Speaking Tutor Learning Mode
- 번역이 필요 없는 순수 STT 작업

차이점 (/api/ai/voice/realtime 대비):
- 자동 언어 감지 없음 (단일 언어만 인식)
- 번역 없음 (STT 결과만 반환)
- 더 빠른 응답 속도
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Optional
import json
import uuid
import logging
import time
import asyncio

from agent.stt_translation.stt_agent import STTAgent
import azure.cognitiveservices.speech as speechsdk

# 로거 설정
logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter()

# 활성 WebSocket 연결 관리
active_connections: Dict[str, WebSocket] = {}

# 세션별 Session 인스턴스 관리
session_instances: Dict[str, 'STTOnlySession'] = {}


# ============================================================
# 표준 WebSocket 메시지 프로토콜 래퍼
# ============================================================
async def send_message(websocket: WebSocket, message_type: str, **kwargs):
    """
    WebSocket 메시지 전송 래퍼 함수

    메시지 타입:
    - recognizing: 중간 인식 결과
    - recognized: 최종 인식 결과
    - error: 에러 메시지
    - end: 연결 종료

    Args:
        websocket: WebSocket 연결
        message_type: 메시지 타입
        **kwargs: 메시지 데이터
    """
    ALLOWED_TYPES = {"recognizing", "recognized", "error", "end", "pong"}

    if message_type not in ALLOWED_TYPES:
        logger.warning(f"Unknown message type: {message_type}")
        return

    message = {"type": message_type, **kwargs}
    await websocket.send_json(message)
    logger.debug(f"Sent message: type={message_type}")


class STTOnlySession:
    """
    단일 언어 STT 전용 세션 (번역 없음)

    회화연습, Learning Mode 등 번역이 필요 없는 경우 사용
    자동 언어 감지 없이 지정된 언어로만 인식하여 성능 최적화
    """

    def __init__(self, session_id: str, websocket: WebSocket, loop: asyncio.AbstractEventLoop):
        self.session_id = session_id
        self.websocket = websocket
        self.loop = loop  # 메인 이벤트 루프 (스레드 간 비동기 호출용)

        # STT Agent (싱글톤)
        self.agent = STTAgent.get_instance()

        # Azure Speech 리소스
        self.recognizer: Optional[speechsdk.SpeechRecognizer] = None
        self.push_stream: Optional[speechsdk.audio.PushAudioInputStream] = None

        # 세션 설정
        self.language: str = "en-US"
        self.auto_segment: bool = False  # 자동 분절 모드 (침묵 감지)

        # 통계
        self.processed_chunks = 0
        self.start_time = time.time()

        logger.info(f"STTOnlySession created: session_id={session_id}")

    async def initialize(self, language: str, auto_segment: bool = False):
        """
        세션 초기화 및 Azure Speech 단일 언어 설정

        Args:
            language: 인식 언어 (BCP-47 코드, 예: "en-US", "ko-KR")
            auto_segment: 자동 분절 모드 (True: 침묵 감지로 자동 문장 분리)
        """
        # 언어 유효성 검사
        if not language or not isinstance(language, str):
            await send_message(
                self.websocket, "error",
                error="language must be a string (e.g., 'en-US')"
            )
            return False

        self.language = language
        self.auto_segment = auto_segment

        try:
            logger.info(f"Setting up STT stream: language={language}, auto_segment={auto_segment}")

            # STT Agent를 통해 단일 언어 스트림 설정
            self.recognizer, self.push_stream = await self.agent.process_stream(
                language=language,
                auto_segment=auto_segment
            )

            # 이벤트 핸들러 등록
            self.recognizer.recognizing.connect(self._on_recognizing)
            self.recognizer.recognized.connect(self._on_recognized)
            self.recognizer.canceled.connect(self._on_canceled)

            # 연속 인식 시작
            self.recognizer.start_continuous_recognition_async()
            logger.info(f"STT stream started: session_id={self.session_id}, language={language}")

            return True

        except Exception as e:
            logger.error(f"Session initialization failed: {str(e)}", exc_info=True)
            await send_message(
                self.websocket, "error",
                error=f"STT initialization failed: {str(e)}"
            )
            return False

    def _on_recognizing(self, evt: speechsdk.SpeechRecognitionEventArgs):
        """중간 인식 결과 핸들러"""
        if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
            text = evt.result.text
            if text and text.strip():
                asyncio.run_coroutine_threadsafe(
                    send_message(
                        self.websocket,
                        "recognizing",
                        text=text
                    ),
                    self.loop
                )

    def _on_recognized(self, evt: speechsdk.SpeechRecognitionEventArgs):
        """최종 인식 결과 핸들러"""
        # NoMatch는 무시
        if evt.result.reason == speechsdk.ResultReason.NoMatch:
            logger.debug("No speech detected")
            return

        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = evt.result.text
            if text and text.strip():
                logger.info(f"Recognized: '{text}'")
                asyncio.run_coroutine_threadsafe(
                    send_message(
                        self.websocket,
                        "recognized",
                        text=text,
                        language=self.language
                    ),
                    self.loop
                )

    def _on_canceled(self, evt: speechsdk.SpeechRecognitionCanceledEventArgs):
        """인식 취소/에러 핸들러"""
        try:
            cancellation = getattr(evt, 'cancellation_details', None)
            if cancellation:
                cancel_reason = getattr(cancellation, 'reason', None)
                if cancel_reason == speechsdk.CancellationReason.EndOfStream:
                    return  # 정상 종료

                error_details = getattr(cancellation, 'error_details', '')
                if error_details:
                    asyncio.run_coroutine_threadsafe(
                        send_message(
                            self.websocket, "error",
                            error=f"Recognition error: {error_details}"
                        ),
                        self.loop
                    )
        except Exception as e:
            logger.error(f"Error in canceled handler: {str(e)}")

    async def process_audio_chunk(self, audio_bytes: bytes):
        """
        오디오 청크를 Azure Speech PushStream에 전송

        Args:
            audio_bytes: PCM 16kHz 16bit Mono 오디오 데이터
        """
        if not self.push_stream:
            logger.warning("PushStream not initialized")
            return

        if not audio_bytes or len(audio_bytes) == 0:
            return

        try:
            self.push_stream.write(audio_bytes)
            self.processed_chunks += 1

            # 처음 5개 청크만 로그
            if self.processed_chunks <= 5:
                logger.info(f"Audio chunk #{self.processed_chunks}: {len(audio_bytes)} bytes")

        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            await send_message(
                self.websocket, "error",
                error=f"Audio processing error: {str(e)}"
            )

    async def cleanup(self):
        """세션 정리 (Azure Speech 리소스 해제)"""
        try:
            if self.recognizer:
                self.recognizer.stop_continuous_recognition()
                logger.info(f"Recognizer stopped: session_id={self.session_id}")

            if self.push_stream:
                self.push_stream.close()
                logger.info(f"PushStream closed: session_id={self.session_id}")

        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

    def get_stats(self) -> Dict:
        """세션 통계 반환"""
        elapsed_time = time.time() - self.start_time
        return {
            "session_id": self.session_id,
            "elapsed_time": round(elapsed_time, 2),
            "processed_chunks": self.processed_chunks,
            "language": self.language
        }


@router.websocket("/api/ai/voice/stt-stream")
async def voice_stt_stream_websocket(websocket: WebSocket):
    """
    단일 언어 STT WebSocket 엔드포인트 (번역 없음, 고성능)

    회화연습, Learning Mode 등 번역이 필요 없는 경우 사용합니다.
    자동 언어 감지 없이 지정된 언어로만 인식하여 더 빠른 응답을 제공합니다.

    프로토콜:
    Client → Server:
    1. {"language": "en-US", "auto_segment": false}  # 단일 언어 (BCP-47, string)
       - auto_segment: true면 침묵 감지로 자동 문장 분리 (기본: false)
    2. [Binary: PCM 16kHz 16bit Mono]
    3. {"type": "end"}

    Server → Client:
    1. {"type": "recognizing", "text": "..."}
    2. {"type": "recognized", "text": "...", "language": "en-US"}
    3. {"type": "error", "error": "..."}
    4. {"type": "end"}
    """
    logger.info("STT-only WebSocket connection requested")

    # WebSocket 연결 수락
    await websocket.accept()
    logger.info("STT-only WebSocket connection accepted")

    # 세션 ID 생성
    session_id = str(uuid.uuid4())
    session: Optional[STTOnlySession] = None

    # 활성 연결 등록
    active_connections[session_id] = websocket

    logger.info(f"STT-only session created: session_id={session_id}")

    try:
        # 메시지 수신 루프
        while True:
            message = await websocket.receive()

            # JSON 메시지 처리 (세션 초기화 또는 종료)
            if "text" in message:
                logger.info(f"JSON message received: {message['text']}")
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {e}")
                    await send_message(websocket, "error", error="Invalid JSON format")
                    continue

                # language로 세션 초기화
                if "language" in data:
                    language = data["language"]
                    auto_segment = data.get("auto_segment", False)  # 자동 분절 모드 (기본: False)

                    # 배열이 아닌 문자열인지 확인
                    if isinstance(language, list):
                        # 배열이면 첫 번째 요소 사용 (하위 호환)
                        language = language[0] if language else "en-US"
                        logger.warning(f"Array received, using first element: {language}")

                    logger.info(f"Initializing STT session: language={language}, auto_segment={auto_segment}")

                    # 세션 생성 및 초기화
                    loop = asyncio.get_event_loop()
                    session = STTOnlySession(session_id, websocket, loop)
                    success = await session.initialize(language, auto_segment=auto_segment)

                    if success:
                        session_instances[session_id] = session
                        logger.info(f"STT session initialized: session_id={session_id}")
                    else:
                        logger.error(f"STT session initialization failed")

                # 종료 메시지
                elif data.get("type") == "end":
                    logger.info(f"End signal received: session_id={session_id}")
                    await send_message(websocket, "end")
                    break

                # Heartbeat ping 처리 - pong 응답
                elif data.get("type") == "ping":
                    logger.debug(f"💓 [STT-Stream] Heartbeat ping received: session_id={session_id}")
                    await send_message(websocket, "pong")

                else:
                    # 알 수 없는 메시지는 경고만 남기고 연결 유지
                    logger.warning(f"Unknown message type (ignored): {data}")

            # Binary 메시지 처리 (오디오 청크)
            elif "bytes" in message:
                audio_bytes = message["bytes"]

                if not session:
                    await send_message(
                        websocket, "error",
                        error="Session not initialized. Send language first."
                    )
                    continue

                if audio_bytes and len(audio_bytes) > 0:
                    await session.process_audio_chunk(audio_bytes)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session_id={session_id}")

    except Exception as e:
        logger.error(f"WebSocket error: session_id={session_id}, error={str(e)}", exc_info=True)
        try:
            await send_message(websocket, "error", error=f"Server error: {str(e)}")
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

        logger.info(f"Session cleaned up: session_id={session_id}")


@router.get("/api/ai/voice/stt-stream/sessions")
async def get_active_stt_sessions():
    """
    현재 활성 STT 세션 목록 조회 (모니터링용)

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
