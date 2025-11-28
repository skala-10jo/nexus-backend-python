"""
Voice STT WebSocket API 엔드포인트

실시간 음성 인식을 위한 WebSocket 연결 제공
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging
import base64
import asyncio
import jwt
from app.services.voice_stt_service import VoiceSTTService
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["Voice STT WebSocket"])

# 사용자별 동시 연결 수 제한
active_connections = {}  # {user_id: connection_count}
MAX_CONNECTIONS_PER_USER = 3

# 허용된 Origin (CORS 대응)
ALLOWED_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]


def verify_token_from_string(token: str) -> dict:
    """
    JWT 토큰 검증 및 사용자 정보 추출

    Args:
        token: JWT 토큰 문자열

    Returns:
        dict: {"user_id": UUID, "username": str}

    Raises:
        jwt.JWTError: 토큰 검증 실패 시
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=settings.JWT_ALGORITHMS
        )
        user_id_str = payload.get("userId")
        username = payload.get("username")

        if user_id_str is None:
            raise jwt.JWTError("Invalid token: missing user ID")

        return {
            "user_id": user_id_str,
            "username": username
        }
    except jwt.JWTError as e:
        logger.error(f"Token verification failed: {str(e)}")
        raise


@router.websocket("/ws/stt")
async def websocket_stt(websocket: WebSocket):
    """
    실시간 음성 인식 WebSocket 엔드포인트

    메시지 프로토콜:

    Client → Server:
    1. {"type": "auth", "token": "JWT_TOKEN"}
    2. {"type": "config", "language": "en-US", "enable_diarization": false}
    3. {"type": "audio", "data": "base64_encoded_audio_chunk"}
    4. {"type": "stop"}

    Server → Client:
    1. {"type": "auth_success", "user_id": "...", "username": "..."}
    2. {"type": "auth_error", "message": "..."}
    3. {"type": "interim", "text": "...", "confidence": null}
    4. {"type": "final", "text": "...", "confidence": 0.95}
    5. {"type": "error", "message": "..."}
    6. {"type": "closed"}
    """
    # Origin 검증 (CORS)
    origin = websocket.headers.get("origin")
    if origin and origin not in ALLOWED_ORIGINS:
        logger.warning(f"WebSocket connection rejected: invalid origin {origin}")
        await websocket.close(code=1008, reason="Invalid origin")
        return

    # WebSocket 연결 수락
    await websocket.accept()
    logger.info("WebSocket connection established")

    user = None
    recognizer = None
    push_stream = None
    result_queue = None
    service = None

    try:
        # 1. 인증 단계 (첫 메시지로 토큰 받기)
        auth_message = await asyncio.wait_for(
            websocket.receive_json(),
            timeout=10.0  # 10초 내에 인증 필요
        )

        if auth_message.get("type") != "auth":
            await websocket.send_json({
                "type": "auth_error",
                "message": "First message must be authentication"
            })
            await websocket.close()
            return

        # JWT 검증
        try:
            user = verify_token_from_string(auth_message.get("token", ""))
            user_id = user["user_id"]
            username = user["username"]

            logger.info(f"User authenticated: {username} ({user_id})")

            # 동시 연결 수 제한
            if active_connections.get(user_id, 0) >= MAX_CONNECTIONS_PER_USER:
                await websocket.send_json({
                    "type": "auth_error",
                    "message": f"Maximum {MAX_CONNECTIONS_PER_USER} concurrent connections exceeded"
                })
                await websocket.close()
                return

            # 연결 카운트 증가
            active_connections[user_id] = active_connections.get(user_id, 0) + 1

            # 인증 성공 응답
            await websocket.send_json({
                "type": "auth_success",
                "user_id": user_id,
                "username": username
            })

        except jwt.JWTError as e:
            await websocket.send_json({
                "type": "auth_error",
                "message": f"Authentication failed: {str(e)}"
            })
            await websocket.close()
            return

        # 2. 설정 단계 (언어, diarization 등)
        config_message = await asyncio.wait_for(
            websocket.receive_json(),
            timeout=10.0
        )

        if config_message.get("type") != "config":
            await websocket.send_json({
                "type": "error",
                "message": "Second message must be configuration"
            })
            await websocket.close()
            return

        language = config_message.get("language", "en-US")
        enable_diarization = config_message.get("enable_diarization", False)

        logger.info(f"STT configured: language={language}, diarization={enable_diarization}")

        # 3. Service 호출 (recognizer, push_stream, result_queue 생성)
        service = VoiceSTTService()
        recognizer, push_stream, result_queue = await service.transcribe_stream(
            language=language,
            enable_diarization=enable_diarization,
            user_id=user_id
        )

        logger.info("STT streaming started")

        # 4. 동시 작업: 오디오 수신 + 결과 전송
        async def receive_audio():
            """오디오 청크 수신 및 PushStream에 쓰기"""
            try:
                while True:
                    message = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=30.0  # 30초 동안 메시지 없으면 타임아웃
                    )

                    msg_type = message.get("type")

                    if msg_type == "audio":
                        # Base64 디코딩
                        audio_data = base64.b64decode(message.get("data", ""))

                        # 크기 제한 (100KB per chunk)
                        if len(audio_data) > 100 * 1024:
                            await websocket.send_json({
                                "type": "error",
                                "message": "Audio chunk too large (max 100KB)"
                            })
                            continue

                        # PushStream에 쓰기
                        push_stream.write(audio_data)
                        logger.info(f"Audio chunk received: {len(audio_data)} bytes")

                    elif msg_type == "stop":
                        logger.info("Stop signal received from client")
                        break

                    else:
                        logger.warning(f"Unknown message type: {msg_type}")

            except asyncio.TimeoutError:
                logger.warning("Audio receive timeout (30s)")
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected by client")
            except Exception as e:
                logger.error(f"Error receiving audio: {str(e)}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error receiving audio: {str(e)}"
                })

        async def send_results():
            """결과 큐에서 읽어서 WebSocket으로 전송"""
            try:
                while True:
                    # 큐에서 결과 가져오기
                    result = await result_queue.get()

                    # WebSocket으로 전송
                    await websocket.send_json(result)

                    # session_stopped 시그널이면 종료
                    if result.get("type") == "session_stopped":
                        logger.info("Session stopped signal received")
                        break

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected while sending results")
            except Exception as e:
                logger.error(f"Error sending results: {str(e)}", exc_info=True)

        # 동시 실행
        await asyncio.gather(
            receive_audio(),
            send_results(),
            return_exceptions=True
        )

    except asyncio.TimeoutError:
        logger.warning("Authentication timeout")
        await websocket.send_json({
            "type": "error",
            "message": "Authentication timeout"
        })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Server error: {str(e)}"
            })
        except:
            pass

    finally:
        # 리소스 정리
        logger.info("Cleaning up WebSocket resources")

        # recognizer 정리
        if recognizer:
            try:
                recognizer.stop_continuous_recognition()
                logger.info("Recognizer stopped")
            except Exception as e:
                logger.error(f"Error stopping recognizer: {str(e)}")

        # push_stream 정리
        if push_stream:
            try:
                push_stream.close()
                logger.info("Push stream closed")
            except Exception as e:
                logger.error(f"Error closing push stream: {str(e)}")

        # 연결 카운트 감소
        if user:
            user_id = user["user_id"]
            if user_id in active_connections:
                active_connections[user_id] -= 1
                if active_connections[user_id] <= 0:
                    del active_connections[user_id]
                logger.info(f"Connection count for user {user_id}: {active_connections.get(user_id, 0)}")

        # WebSocket 종료
        try:
            await websocket.send_json({"type": "closed"})
            await websocket.close()
        except:
            pass

        logger.info("WebSocket connection closed")
