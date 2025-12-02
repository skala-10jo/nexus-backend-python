"""
API endpoints for expression pronunciation assessment and TTS.
"""
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from sqlalchemy.orm import Session
from uuid import UUID
import json

from app.database import get_db
from app.config import Settings
from app.schemas.expression_speech import (
    TTSRequest,
    TTSTextRequest,
    TTSResponse
)
from app.services.expression_speech_service import ExpressionSpeechService
from app.models.expression import Expression
from agent.voice.pronunciation_streaming_agent import PronunciationStreamingAgent
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


def get_settings() -> Settings:
    """Get application settings"""
    return Settings()


@router.post("/expression/speech/synthesize")
async def synthesize_speech(
    request: TTSRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings)
):
    """
    TTS 음성 합성

    Args:
        request: TTS 요청 (expression_id, voice_name)
        db: 데이터베이스 세션
        settings: 애플리케이션 설정

    Returns:
        WAV 오디오 파일 (바이너리)

    Example:
        >>> POST /api/ai/expression/speech/synthesize
        >>> Content-Type: application/json
        >>> {
        >>>   "expression_id": "123e4567-e89b-12d3-a456-426614174000",
        >>>   "voice_name": "en-US-JennyNeural"
        >>> }
        >>> Response: <binary WAV audio>
    """
    logger.info(f"TTS 요청: expression_id={request.expression_id}, voice={request.voice_name}")

    # 서비스 호출
    service = ExpressionSpeechService(settings)
    audio_data = await service.synthesize_speech(
        expression_id=request.expression_id,
        voice_name=request.voice_name,
        db=db
    )

    logger.info(f"TTS 완료: {len(audio_data)} bytes")

    # 오디오 파일 반환
    return Response(
        content=audio_data,
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=expression.wav"
        }
    )


@router.post("/expression/speech/synthesize-text")
async def synthesize_text(
    request: TTSTextRequest,
    settings: Settings = Depends(get_settings)
):
    """
    텍스트를 직접 음성으로 합성

    Args:
        request: TTS 요청 (text, voice_name)
        settings: 애플리케이션 설정

    Returns:
        WAV 오디오 파일 (바이너리)

    Example:
        >>> POST /api/ai/expression/speech/synthesize-text
        >>> Content-Type: application/json
        >>> {
        >>>   "text": "I am writing to inform you of the meeting schedule",
        >>>   "voice_name": "en-US-JennyNeural"
        >>> }
        >>> Response: <binary WAV audio>
    """
    logger.info(f"TTS text 요청: text_length={len(request.text)}, voice={request.voice_name}")

    # 서비스 호출
    service = ExpressionSpeechService(settings)
    audio_data = await service.synthesize_text(
        text=request.text,
        voice_name=request.voice_name
    )

    logger.info(f"TTS 완료: {len(audio_data)} bytes")

    # 오디오 파일 반환
    return Response(
        content=audio_data,
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=speech.wav"
        }
    )


@router.websocket("/expression/speech/assess-realtime")
async def assess_pronunciation_realtime(
    websocket: WebSocket,
    settings: Settings = Depends(get_settings)
):
    """
    실시간 스트리밍 발음 평가 (WebSocket)

    Protocol:
        1. Client → Server: {"expression_id": "uuid", "type": "expression" or "example", "example_index": 0}
        2. Client → Server: audio chunks (binary)
        3. Server → Client: {"status": "processing"}
        4. Server → Client: {"result": {...}} (최종 결과)

    Example:
        >>> ws = new WebSocket('ws://localhost:8000/api/ai/expression/speech/assess-realtime')
        >>> ws.send(JSON.stringify({expression_id: "...", type: "expression"}))
        >>> ws.send(audioChunk)  // binary
        >>> ws.send(audioChunk)  // binary
        >>> ws.close()
    """
    await websocket.accept()
    logger.info("WebSocket 연결 수립")

    db = next(get_db())
    recognizer = None
    push_stream = None

    try:
        # 1. 첫 메시지: expression_id 및 타입 수신
        init_message = await websocket.receive_text()
        init_data = json.loads(init_message)

        expression_id = UUID(init_data["expression_id"])
        eval_type = init_data.get("type", "expression")
        example_index = init_data.get("example_index", 0)

        logger.info(f"발음 평가 시작: expression_id={expression_id}, type={eval_type}")

        # 2. DB에서 표현 조회
        expression = db.query(Expression).filter(Expression.id == expression_id).first()
        if not expression:
            await websocket.send_json({"error": "표현을 찾을 수 없습니다"})
            await websocket.close()
            return

        # 3. reference_text 결정
        if eval_type == "expression":
            reference_text = expression.expression
        elif eval_type == "example":
            # JSONB는 SQLAlchemy가 자동으로 Python list로 변환
            examples = expression.examples
            if example_index >= len(examples):
                await websocket.send_json({"error": "잘못된 example_index"})
                await websocket.close()
                return
            reference_text = examples[example_index]["text"]
        else:
            await websocket.send_json({"error": "잘못된 type (expression 또는 example)"})
            await websocket.close()
            return

        logger.info(f"Reference text: {reference_text}")

        # 4. 스트리밍 Agent 생성
        agent = PronunciationStreamingAgent(
            speech_key=settings.AZURE_SPEECH_KEY,
            speech_region=settings.AZURE_SPEECH_REGION
        )

        recognizer, push_stream = agent.create_recognizer(
            reference_text=reference_text,
            language="en-US"
        )

        # 5. 인식 시작 (비동기)
        done = False
        final_result = None

        def recognized_callback(evt):
            nonlocal final_result, done
            final_result = agent.parse_result(evt.result)
            done = True

        recognizer.recognized.connect(recognized_callback)
        recognizer.start_continuous_recognition()

        await websocket.send_json({"status": "ready", "reference_text": reference_text})

        # 6. 오디오 청크 수신 및 전송
        while not done:
            try:
                message = await websocket.receive()

                if "bytes" in message:
                    # 오디오 청크
                    audio_chunk = message["bytes"]
                    push_stream.write(audio_chunk)
                    await websocket.send_json({"status": "processing"})

                elif "text" in message:
                    # 종료 신호
                    text = message["text"]
                    if text == "END":
                        push_stream.close()
                        break

            except WebSocketDisconnect:
                break

        # 7. 인식 완료 대기
        recognizer.stop_continuous_recognition()

        # 8. 결과 전송
        if final_result:
            await websocket.send_json({"result": final_result})
            logger.info(f"발음 평가 완료: score={final_result.get('pronunciation_score', 0)}")
        else:
            await websocket.send_json({"error": "발음 평가 실패"})

    except WebSocketDisconnect:
        logger.info("WebSocket 연결 종료 (클라이언트)")
    except Exception as e:
        logger.error(f"WebSocket 오류: {str(e)}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        if recognizer:
            recognizer.stop_continuous_recognition()
        if push_stream:
            push_stream.close()
        db.close()
        try:
            await websocket.close()
        except:
            pass
