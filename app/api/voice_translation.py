"""
실시간 음성 번역 WebSocket API (Azure Speech + Azure Translator)

엔드포인트:
- WS /api/ai/voice/realtime: 실시간 음성 번역 WebSocket 연결

최적화:
- Azure Speech SDK 자동 언어 감지
- Azure Translator API 멀티 타겟 번역
- WebSocket 압축 (permessage-deflate)
- 비동기 처리 (asyncio)
- 에러 처리 강화
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Optional, List
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


class VoiceTranslationSession:
    """실시간 음성 번역 세션 관리 (Azure Speech + Azure Translator)"""

    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket

        # Service 초기화 (AI Agent 아키텍처 가이드 준수: API → Service → Agent)
        self.service = VoiceTranslationService()

        # Azure Speech 리소스
        self.recognizer: Optional[speechsdk.SpeechRecognizer] = None
        self.push_stream: Optional[speechsdk.audio.PushAudioInputStream] = None

        # 세션 설정
        self.selected_languages: List[str] = []  # BCP-47 코드 (ko-KR, en-US, ja-JP)

        # 통계
        self.processed_chunks = 0
        self.total_translations = 0
        self.start_time = time.time()

        logger.info(f"✅ VoiceTranslationSession 생성: session_id={session_id}")

    async def initialize(self, selected_languages: List[str]):
        """
        세션 초기화 및 Azure Speech 자동 언어 감지 설정

        Args:
            selected_languages: 선택된 언어 목록 (BCP-47 코드)
                예: ["ko-KR", "en-US", "ja-JP"]
        """
        if not selected_languages or len(selected_languages) < 2:
            await send_standard_message(
                self.websocket, "error",
                error="최소 2개 이상의 언어를 선택해야 합니다"
            )
            return

        self.selected_languages = selected_languages

        try:
            # Azure Speech 자동 언어 감지 스트림 생성 (Service를 통해 Agent 호출)
            logger.info(f"🔧 Azure Speech 자동 언어 감지 설정: {selected_languages}")
            self.recognizer, self.push_stream = await self.service.setup_stream_with_auto_detect(
                candidate_languages=selected_languages
            )

            # 이벤트 핸들러 등록
            self.recognizer.recognizing.connect(self._on_recognizing)
            self.recognizer.recognized.connect(self._on_recognized)
            self.recognizer.canceled.connect(self._on_canceled)

            # 연속 인식 시작
            self.recognizer.start_continuous_recognition()
            logger.info(f"🎤 Azure Speech 연속 인식 시작: session_id={self.session_id}")

        except Exception as e:
            logger.error(f"❌ 세션 초기화 실패: {str(e)}", exc_info=True)
            await send_standard_message(
                self.websocket, "error",
                error=f"세션 초기화 실패: {str(e)}"
            )

    def _on_recognizing(self, evt: speechsdk.SpeechRecognitionEventArgs):
        """
        중간 인식 결과 핸들러 (recognizing)

        Azure Speech SDK가 실시간으로 부분 인식 결과를 반환할 때 호출됩니다.
        번역은 수행하지 않고 원본 텍스트만 전송합니다.
        """
        if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
            text = evt.result.text
            if text and text.strip():
                logger.debug(f"🔍 Recognizing: '{text}'")

                # 비동기로 메시지 전송 (이벤트 핸들러는 동기 함수)
                asyncio.create_task(
                    send_standard_message(
                        self.websocket,
                        "recognizing",
                        text=text
                    )
                )

    def _on_recognized(self, evt: speechsdk.SpeechRecognitionEventArgs):
        """
        최종 인식 결과 핸들러 (recognized)

        Azure Speech SDK가 최종 인식 결과를 반환할 때 호출됩니다.
        1. 자동 감지된 언어 확인
        2. 감지된 언어를 제외한 타겟 언어로 번역
        3. recognized 메시지 전송
        """
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = evt.result.text

            # 텍스트가 비어있으면 무시
            if not text or not text.strip():
                logger.debug("⚪ 무음 구간 감지 (recognized)")
                return

            # 자동 감지된 언어 추출
            detected_lang_bcp47 = evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
            )

            if not detected_lang_bcp47:
                logger.warning("⚠️ 언어 자동 감지 실패, 기본값 사용: ko-KR")
                detected_lang_bcp47 = "ko-KR"

            logger.info(f"🎤 Recognized: '{text}' (언어: {detected_lang_bcp47})")

            # 비동기 번역 작업 실행
            asyncio.create_task(
                self._translate_and_send(text, detected_lang_bcp47, evt)
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
        try:
            # 감지된 언어를 제외한 타겟 언어 목록 생성
            target_langs_bcp47 = [
                lang for lang in self.selected_languages
                if lang != detected_lang_bcp47
            ]

            if not target_langs_bcp47:
                logger.warning(f"⚠️ 번역 타겟 언어 없음 (감지 언어: {detected_lang_bcp47})")
                # 번역 없이 원문만 전송
                await send_standard_message(
                    self.websocket,
                    "recognized",
                    text=text,
                    detected_language=detected_lang_bcp47,
                    translations=[],
                    confidence=0.9
                )
                return

            # BCP-47 → ISO 639-1 변환
            detected_lang_iso = bcp47_to_iso639(detected_lang_bcp47)
            target_langs_iso = [bcp47_to_iso639(lang) for lang in target_langs_bcp47]

            logger.info(
                f"🌐 번역 시작: {detected_lang_iso} → {target_langs_iso}, "
                f"text='{text[:50]}...'"
            )

            # Azure Translator 멀티 타겟 번역 (Service를 통해 Agent 호출)
            translations = await self.service.translate_to_multiple_languages(
                text=text,
                source_lang=detected_lang_iso,
                target_langs=target_langs_iso
            )

            # ISO 639-1 → BCP-47 변환 (프론트엔드 호환)
            translations_bcp47 = [
                {"lang": iso639_to_bcp47(t["lang"]), "text": t["text"]}
                for t in translations
            ]

            logger.info(f"✅ 번역 완료: {len(translations_bcp47)}개 언어")

            # 통계 업데이트
            self.processed_chunks += 1
            self.total_translations += len(translations_bcp47)

            # recognized 메시지 전송
            await send_standard_message(
                self.websocket,
                "recognized",
                text=text,
                detected_language=detected_lang_bcp47,
                translations=translations_bcp47,
                confidence=0.9
            )

        except Exception as e:
            logger.error(f"❌ 번역 실패: {str(e)}", exc_info=True)
            await send_standard_message(
                self.websocket, "error",
                error=f"번역 중 오류 발생: {str(e)}"
            )

    def _on_canceled(self, evt: speechsdk.SpeechRecognitionCanceledEventArgs):
        """
        인식 취소/에러 핸들러

        Azure Speech SDK가 에러를 반환할 때 호출됩니다.
        """
        logger.error(
            f"❌ Speech 인식 취소: reason={evt.reason}, "
            f"cancellation_details={evt.cancellation_details}"
        )

        # 에러 메시지 전송
        asyncio.create_task(
            send_standard_message(
                self.websocket, "error",
                error=f"음성 인식 오류: {evt.cancellation_details.error_details}"
            )
        )

    async def process_audio_chunk(self, audio_bytes: bytes):
        """
        오디오 청크를 Azure Speech PushStream에 전송

        Args:
            audio_bytes: 원본 바이너리 오디오 데이터 (WebM/Opus, 16kHz, mono)
        """
        if not self.push_stream:
            logger.warning("⚠️ PushStream이 초기화되지 않았습니다")
            return

        if not audio_bytes or len(audio_bytes) == 0:
            logger.debug("⚪ 빈 오디오 청크 무시")
            return

        try:
            # Azure Speech PushStream에 원본 바이너리 쓰기 (base64 디코딩 없음!)
            self.push_stream.write(audio_bytes)
            logger.debug(f"📤 오디오 청크 전송: {len(audio_bytes)} bytes")

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
            "total_translations": self.total_translations,
            "selected_languages": self.selected_languages
        }


@router.websocket("/api/ai/voice/realtime")
async def voice_translation_websocket(websocket: WebSocket):
    """
    실시간 음성 번역 WebSocket 엔드포인트

    클라이언트와 WebSocket 연결을 맺고 실시간으로 음성 번역을 수행합니다.

    프로토콜:
    1. 클라이언트 → 서버 (JSON): {"selected_languages": ["ko-KR", "en-US", "ja-JP"]}
    2. 클라이언트 → 서버 (Binary): 오디오 청크 (WebM/Opus, 16kHz, mono)
    3. 서버 → 클라이언트 (JSON):
       - {"type": "recognizing", "text": "..."}
       - {"type": "recognized", "text": "...", "detected_language": "ko-KR", "translations": [...]}
       - {"type": "error", "error": "..."}
       - {"type": "end"}
    """

    # WebSocket 연결 수락
    await websocket.accept()

    # 세션 ID 생성
    session_id = str(uuid.uuid4())
    session: Optional[VoiceTranslationSession] = None

    # 활성 연결 등록
    active_connections[session_id] = websocket

    logger.info(f"✅ WebSocket 연결됨: session_id={session_id}")

    try:
        # 메시지 수신 루프
        while True:
            # 클라이언트로부터 메시지 수신
            message = await websocket.receive()

            # JSON 메시지 처리 (세션 초기화)
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await send_standard_message(websocket, "error", error="잘못된 JSON 형식입니다")
                    continue

                # selected_languages로 세션 초기화
                if "selected_languages" in data:
                    selected_languages = data["selected_languages"]
                    logger.info(f"📝 세션 초기화 요청: {selected_languages}")

                    # 세션 생성 및 초기화
                    session = VoiceTranslationSession(session_id, websocket)
                    await session.initialize(selected_languages)

                    # 세션 저장
                    session_instances[session_id] = session

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
