"""
실시간 음성 번역 WebSocket API

엔드포인트:
- WS /api/ai/voice/realtime: 실시간 음성 번역 WebSocket 연결

최적화:
- WebSocket 압축 (permessage-deflate)
- 비동기 처리 (asyncio)
- 에러 처리 강화
- 세션 관리
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, WebSocketException
from typing import Dict, Optional
import json
import uuid
import logging
import time

from agent.voice.realtime_stt_agent import RealtimeSTTAgent
from agent.voice.multi_translation_agent import MultiLanguageTranslationAgent
from agent.voice.speaker_diarization_agent import SpeakerDiarizationAgent

# 로거 설정
logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter()

# 활성 WebSocket 연결 관리
active_connections: Dict[str, WebSocket] = {}

# 세션별 Agent 인스턴스 관리
session_agents: Dict[str, Dict] = {}


class VoiceTranslationSession:
    """실시간 음성 번역 세션 관리"""

    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket

        # Agent 초기화
        self.stt_agent = RealtimeSTTAgent()
        self.translation_agent = MultiLanguageTranslationAgent()
        self.speaker_agent = SpeakerDiarizationAgent()

        # 세션 설정
        self.input_language: Optional[str] = None
        self.output_languages: list = []

        # 통계
        self.processed_chunks = 0
        self.total_translations = 0
        self.start_time = time.time()

    def get_stats(self) -> Dict:
        """세션 통계 반환"""
        elapsed_time = time.time() - self.start_time
        return {
            "session_id": self.session_id,
            "elapsed_time": round(elapsed_time, 2),
            "processed_chunks": self.processed_chunks,
            "total_translations": self.total_translations,
            "speaker_count": self.speaker_agent.get_speaker_count(),
            "speaker_stats": self.speaker_agent.get_speaker_stats()
        }


@router.websocket("/api/ai/voice/realtime")
async def voice_translation_websocket(websocket: WebSocket):
    """
    실시간 음성 번역 WebSocket 엔드포인트

    클라이언트와 WebSocket 연결을 맺고 실시간으로 음성 번역을 수행합니다.

    메시지 타입:
    - init: 세션 초기화
    - audio_chunk: 오디오 청크 전송
    - ping: 연결 유지
    - get_stats: 세션 통계 요청
    """

    # WebSocket 연결 수락
    await websocket.accept()

    # 세션 ID 생성
    session_id = str(uuid.uuid4())

    # 세션 생성
    session = VoiceTranslationSession(session_id, websocket)

    # 활성 연결 등록
    active_connections[session_id] = websocket
    session_agents[session_id] = session

    logger.info(f"✅ WebSocket 연결됨: session_id={session_id}")

    try:
        # 연결 확인 메시지 전송
        await websocket.send_json({
            "type": "connected",
            "sessionId": session_id,
            "message": "실시간 음성 번역 서버에 연결되었습니다",
            "timestamp": time.time()
        })

        # 메시지 수신 루프
        while True:
            # 클라이언트로부터 메시지 수신
            try:
                data = await websocket.receive_json()
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "잘못된 JSON 형식입니다",
                    "errorCode": "INVALID_JSON"
                })
                continue

            message_type = data.get("type")

            # 메시지 타입별 처리
            if message_type == "init":
                await handle_init(session, data)

            elif message_type == "audio_chunk":
                await handle_audio_chunk(session, data)

            elif message_type == "ping":
                await handle_ping(session)

            elif message_type == "get_stats":
                await handle_get_stats(session)

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"알 수 없는 메시지 타입: {message_type}",
                    "errorCode": "UNKNOWN_MESSAGE_TYPE"
                })

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket 연결 종료: session_id={session_id}")

    except Exception as e:
        logger.error(f"❌ WebSocket 에러: session_id={session_id}, error={str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"서버 오류: {str(e)}",
                "errorCode": "SERVER_ERROR"
            })
        except:
            pass

    finally:
        # 연결 정리
        if session_id in active_connections:
            del active_connections[session_id]
        if session_id in session_agents:
            del session_agents[session_id]
        logger.info(f"🧹 세션 정리 완료: session_id={session_id}")


async def handle_init(session: VoiceTranslationSession, data: Dict):
    """세션 초기화 처리"""

    input_language = data.get("inputLanguage")
    output_languages = data.get("outputLanguages", [])

    # 입력 검증
    if not input_language:
        await session.websocket.send_json({
            "type": "error",
            "message": "입력 언어가 지정되지 않았습니다",
            "errorCode": "MISSING_INPUT_LANGUAGE"
        })
        return

    if not output_languages:
        await session.websocket.send_json({
            "type": "error",
            "message": "출력 언어가 지정되지 않았습니다",
            "errorCode": "MISSING_OUTPUT_LANGUAGES"
        })
        return

    # 세션 설정 업데이트
    session.input_language = input_language
    session.output_languages = output_languages

    logger.info(
        f"📝 세션 초기화: session_id={session.session_id}, "
        f"input={input_language}, output={output_languages}"
    )

    # 성공 응답
    await session.websocket.send_json({
        "type": "init_success",
        "sessionId": session.session_id,
        "inputLanguage": input_language,
        "outputLanguages": output_languages,
        "message": "세션이 초기화되었습니다",
        "timestamp": time.time()
    })


async def handle_audio_chunk(session: VoiceTranslationSession, data: Dict):
    """오디오 청크 처리"""

    # 세션 초기화 확인
    if not session.input_language or not session.output_languages:
        await session.websocket.send_json({
            "type": "error",
            "message": "세션이 초기화되지 않았습니다. init 메시지를 먼저 보내세요",
            "errorCode": "SESSION_NOT_INITIALIZED"
        })
        return

    audio_data = data.get("audioData")
    timestamp = data.get("timestamp", time.time())
    audio_energy = data.get("audioEnergy", 0.5)  # 클라이언트에서 전송 (선택사항)
    audio_format = data.get("audioFormat", {})  # 오디오 형식 정보 (선택사항)

    if not audio_data:
        await session.websocket.send_json({
            "type": "error",
            "message": "오디오 데이터가 없습니다",
            "errorCode": "MISSING_AUDIO_DATA"
        })
        return

    try:
        # 1. STT 수행
        stt_result = await session.stt_agent.process(
            audio_data=audio_data,
            input_language=session.input_language,
            audio_format=audio_format  # 오디오 형식 정보 전달
        )

        original_text = stt_result["text"]

        # 텍스트가 비어있으면 무시 (무음 구간)
        if not original_text.strip():
            logger.debug(f"⚪ 무음 구간 감지: session_id={session.session_id}")
            return

        logger.info(f"🎤 STT 결과: '{original_text}' (session_id={session.session_id})")

        # 2. 화자 구분
        speaker_result = await session.speaker_agent.process(
            audio_energy=audio_energy,
            timestamp=timestamp,
            duration=stt_result.get("duration", 1.0)
        )

        speaker_id = speaker_result["speaker_id"]
        logger.info(f"👤 화자 구분: speaker_id={speaker_id} (session_id={session.session_id})")

        # 3. 다국어 번역
        translations = await session.translation_agent.process(
            text=original_text,
            source_lang=session.input_language,
            target_langs=session.output_languages,
            use_context=True  # 컨텍스트 사용
        )

        logger.info(f"🌐 번역 완료: {len(translations)}개 언어 (session_id={session.session_id})")

        # 4. 통계 업데이트
        session.processed_chunks += 1
        session.total_translations += len(translations)

        # 5. 결과 전송
        await session.websocket.send_json({
            "type": "translation_result",
            "sessionId": session.session_id,
            "speakerId": speaker_id,
            "speakerConfidence": speaker_result["confidence"],
            "isNewSpeaker": speaker_result["is_new_speaker"],
            "originalText": original_text,
            "inputLanguage": session.input_language,
            "translations": translations,
            "timestamp": timestamp,
            "sttConfidence": stt_result["confidence"]
        })

    except Exception as e:
        logger.error(
            f"❌ 오디오 처리 실패: session_id={session.session_id}, error={str(e)}"
        )

        await session.websocket.send_json({
            "type": "error",
            "message": f"오디오 처리 중 오류 발생: {str(e)}",
            "errorCode": "AUDIO_PROCESSING_ERROR"
        })


async def handle_ping(session: VoiceTranslationSession):
    """Ping-Pong (연결 유지)"""

    await session.websocket.send_json({
        "type": "pong",
        "sessionId": session.session_id,
        "timestamp": time.time()
    })


async def handle_get_stats(session: VoiceTranslationSession):
    """세션 통계 전송"""

    stats = session.get_stats()

    await session.websocket.send_json({
        "type": "stats",
        "data": stats,
        "timestamp": time.time()
    })


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

    for session_id, session in session_agents.items():
        sessions_info.append(session.get_stats())

    return {
        "success": True,
        "data": {
            "active_sessions": len(active_connections),
            "sessions": sessions_info
        }
    }
