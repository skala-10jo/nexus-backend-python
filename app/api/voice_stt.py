"""
Voice STT API 엔드포인트

실시간 WebSocket 기반 음성 인식(STT) API
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import logging
import asyncio
from typing import Optional
import azure.cognitiveservices.speech as speechsdk
from agent.stt_translation.stt_agent import STTAgent
from app.schemas.voice import STTRequest, STTResponse, STTStreamMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stt", tags=["Voice STT"])


@router.post(
    "",
    response_model=dict,
    summary="음성 파일을 텍스트로 변환",
    description="""
    업로드된 음성 파일을 텍스트로 변환합니다.

    - 지원 형식: WAV, MP3, OGG
    - 화자 분리(Speaker Diarization) 지원
    - BCP-47 언어 코드 사용 (예: ko-KR, en-US, ja-JP)
    """
)
async def speech_to_text(
    file: UploadFile = File(..., description="음성 파일 (WAV/MP3/OGG)"),
    language: str = Form(default="ko-KR", description="BCP-47 언어 코드"),
    enable_diarization: bool = Form(default=True, description="화자 분리 활성화")
):
    """
    음성 파일을 텍스트로 변환 (POST 업로드)

    Args:
        file: 음성 파일
        language: BCP-47 언어 코드
        enable_diarization: 화자 분리 활성화 여부

    Returns:
        dict: STT 결과
    """
    try:
        logger.info(f"STT request: filename={file.filename}, language={language}, diarization={enable_diarization}")

        # 파일 읽기
        audio_data = await file.read()

        if not audio_data:
            raise HTTPException(status_code=400, detail="음성 파일이 비어있습니다")

        # STT Agent 실행
        agent = STTAgent.get_instance()
        result = await agent.process(
            audio_data=audio_data,
            language=language,
            enable_diarization=enable_diarization
        )

        return JSONResponse(
            content={
                "success": True,
                "message": "음성 인식 완료",
                "data": result
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"음성 인식 실패: {str(e)}")


@router.websocket("/stream")
async def speech_to_text_stream(websocket: WebSocket):
    """
    실시간 WebSocket 기반 STT 스트리밍

    클라이언트가 오디오 청크를 실시간으로 전송하면
    서버가 STT 결과를 실시간으로 반환합니다.

    Protocol:
        1. Client → Server: 설정 메시지 {"language": "ko-KR", "enable_diarization": true}
        2. Client → Server: 오디오 청크 (binary)
        3. Server → Client: STT 결과 {"type": "recognizing", "text": "안녕..."}
        4. Client → Server: 종료 메시지 {"type": "end"}
        5. Server → Client: 종료 확인 {"type": "end"}
    """
    await websocket.accept()
    logger.info("WebSocket STT connection established")

    recognizer: Optional[speechsdk.SpeechRecognizer] = None
    push_stream: Optional[speechsdk.audio.PushAudioInputStream] = None

    try:
        # 1. 초기 설정 메시지 수신
        config_message = await websocket.receive_json()
        language = config_message.get("language", "ko-KR")
        enable_diarization = config_message.get("enable_diarization", True)

        logger.info(f"WebSocket STT config: language={language}, diarization={enable_diarization}")

        # 2. STT Agent 설정 (스트리밍)
        agent = STTAgent.get_instance()
        recognizer, push_stream = await agent.process_stream(
            language=language,
            enable_diarization=enable_diarization
        )

        # 3. 이벤트 핸들러 설정
        def on_recognizing(evt):
            """중간 인식 결과 (실시간 스트리밍)"""
            asyncio.create_task(websocket.send_json({
                "type": "recognizing",
                "text": evt.result.text,
                "speaker_id": None,
                "confidence": None
            }))

        def on_recognized(evt):
            """최종 인식 결과"""
            speaker_id = evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult, {}
            ).get("SpeakerId", "Unknown")

            asyncio.create_task(websocket.send_json({
                "type": "recognized",
                "text": evt.result.text,
                "speaker_id": speaker_id,
                "confidence": 1.0
            }))

        def on_canceled(evt):
            """인식 취소/에러"""
            error_msg = f"STT canceled: {evt.result.cancellation_details.reason}"
            logger.error(error_msg)
            asyncio.create_task(websocket.send_json({
                "type": "error",
                "error": error_msg
            }))

        # 이벤트 핸들러 등록
        recognizer.recognizing.connect(on_recognizing)
        recognizer.recognized.connect(on_recognized)
        recognizer.canceled.connect(on_canceled)

        # 4. 연속 인식 시작
        recognizer.start_continuous_recognition()
        logger.info("Continuous recognition started")

        # 5. 오디오 스트림 수신 루프
        while True:
            try:
                # 메시지 타입 확인
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=30.0  # 30초 타임아웃
                )

                # 종료 메시지 확인
                if "text" in message:
                    json_data = await websocket.receive_json()
                    if json_data.get("type") == "end":
                        logger.info("Client requested end of stream")
                        break

                # 오디오 데이터 수신
                elif "bytes" in message:
                    audio_chunk = message["bytes"]
                    if audio_chunk:
                        push_stream.write(audio_chunk)

            except asyncio.TimeoutError:
                logger.warning("WebSocket receive timeout (30s)")
                break
            except WebSocketDisconnect:
                logger.info("Client disconnected")
                break

        # 6. 정리 및 종료
        push_stream.close()
        recognizer.stop_continuous_recognition()

        await websocket.send_json({"type": "end"})
        logger.info("WebSocket STT session ended")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected during setup")

    except Exception as e:
        logger.error(f"WebSocket STT error: {str(e)}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "error": f"STT 오류: {str(e)}"
            })
        except:
            pass

    finally:
        # 리소스 정리
        if recognizer:
            try:
                recognizer.stop_continuous_recognition()
            except:
                pass

        if push_stream:
            try:
                push_stream.close()
            except:
                pass

        try:
            await websocket.close()
        except:
            pass

        logger.info("WebSocket STT resources cleaned up")
