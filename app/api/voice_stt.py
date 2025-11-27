"""
Voice STT API 엔드포인트

실시간 WebSocket 기반 음성 인식(STT) API
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import logging
import asyncio
from typing import Optional, List
import azure.cognitiveservices.speech as speechsdk
from agent.stt_translation.stt_agent import STTAgent
from agent.stt_translation.translation_agent import TranslationAgent
from app.schemas.voice import STTRequest, STTResponse, STTStreamMessage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Voice STT"])


# ===== Helper Functions for Language Code Conversion =====

def bcp47_to_iso639(bcp47_lang: str) -> str:
    """
    BCP-47 언어 코드를 ISO 639-1 코드로 변환

    Args:
        bcp47_lang: BCP-47 코드 (예: ko-KR, en-US, ja-JP, vi-VN)

    Returns:
        str: ISO 639-1 코드 (예: ko, en, ja, vi)
    """
    return bcp47_lang.split('-')[0]


def iso639_to_bcp47(iso_lang: str) -> str:
    """
    ISO 639-1 언어 코드를 BCP-47 코드로 변환

    Args:
        iso_lang: ISO 639-1 코드 (예: ko, en, ja, vi)

    Returns:
        str: BCP-47 코드 (예: ko-KR, en-US, ja-JP, vi-VN)
    """
    mapping = {
        "ko": "ko-KR",
        "en": "en-US",
        "ja": "ja-JP",
        "vi": "vi-VN"
    }
    return mapping.get(iso_lang, f"{iso_lang}-XX")


async def translate_to_multiple_languages(
    text: str,
    source_lang: str,
    target_langs: List[str],
    translator: TranslationAgent
) -> List[dict]:
    """
    여러 언어로 병렬 번역

    Args:
        text: 원본 텍스트
        source_lang: 원본 언어 (ISO 639-1 코드)
        target_langs: 목표 언어 리스트 (ISO 639-1 코드)
        translator: TranslationAgent 인스턴스

    Returns:
        List[dict]: [{"lang": "en-US", "text": "Hello"}, ...]
    """
    if not target_langs:
        return []

    try:
        # 병렬 번역 실행
        tasks = [
            translator.process(text, source_lang, target_lang)
            for target_lang in target_langs
        ]

        translated_texts = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 포맷팅
        results = []
        for target_lang, translated_text in zip(target_langs, translated_texts):
            if isinstance(translated_text, Exception):
                logger.error(f"Translation failed for {target_lang}: {str(translated_text)}")
                results.append({
                    "lang": iso639_to_bcp47(target_lang),
                    "text": f"[번역 실패: {target_lang}]"
                })
            else:
                results.append({
                    "lang": iso639_to_bcp47(target_lang),
                    "text": translated_text
                })

        return results

    except Exception as e:
        logger.error(f"Batch translation error: {str(e)}", exc_info=True)
        return []


@router.post(
    "/api/ai/voice/stt",
    response_model=dict,
    summary="음성 파일을 텍스트로 변환",
    description="""
    업로드된 음성 파일을 텍스트로 변환합니다.

    - 지원 형식: WAV, MP3, OGG
    - BCP-47 언어 코드 사용 (예: ko-KR, en-US, ja-JP)
    """
)
async def speech_to_text(
    file: UploadFile = File(..., description="음성 파일 (WAV/MP3/OGG)"),
    language: str = Form(default="ko-KR", description="BCP-47 언어 코드")
):
    """
    음성 파일을 텍스트로 변환 (POST 업로드)

    Args:
        file: 음성 파일
        language: BCP-47 언어 코드

    Returns:
        dict: STT 결과
    """
    try:
        logger.info(f"STT request: filename={file.filename}, language={language}")

        # 파일 읽기
        audio_data = await file.read()

        if not audio_data:
            raise HTTPException(status_code=400, detail="음성 파일이 비어있습니다")

        # STT Agent 실행
        agent = STTAgent.get_instance()
        result = await agent.process(
            audio_data=audio_data,
            language=language
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


@router.websocket("/api/ai/voice/stt/stream")
async def speech_to_text_stream(websocket: WebSocket):
    """
    실시간 WebSocket 기반 STT 스트리밍 (다국어 번역 지원)

    클라이언트가 오디오 청크를 실시간으로 전송하면
    서버가 STT 결과와 다국어 번역을 실시간으로 반환합니다.

    Protocol:
        1. Client → Server: 설정 메시지
           {
               "selected_languages": ["ko-KR", "en-US", "ja-JP", "vi-VN"]
           }
        2. Client → Server: 오디오 청크 (binary)
        3. Server → Client: STT 중간 결과
           {"type": "recognizing", "text": "안녕..."}
        4. Server → Client: STT 최종 결과 + 번역
           {
               "type": "recognized",
               "text": "안녕하세요",
               "detected_language": "ko-KR",
               "translations": [
                   {"lang": "en-US", "text": "Hello"},
                   {"lang": "ja-JP", "text": "こんにちは"}
               ]
           }
        5. Client → Server: 종료 메시지 {"type": "end"}
        6. Server → Client: 종료 확인 {"type": "end"}
    """
    await websocket.accept()
    logger.info("WebSocket STT connection established (multi-language mode)")

    recognizer: Optional[speechsdk.SpeechRecognizer] = None
    push_stream: Optional[speechsdk.audio.PushAudioInputStream] = None

    try:
        # 1. 초기 설정 메시지 수신
        config_message = await websocket.receive_json()
        selected_languages = config_message.get("selected_languages", ["ko-KR"])

        logger.info(f"WebSocket STT config: selected_languages={selected_languages}")

        # 2. STT Agent 설정 (자동 언어 감지)
        stt_agent = STTAgent.get_instance()
        recognizer, push_stream = await stt_agent.process_stream_with_auto_detect(
            candidate_languages=selected_languages
        )

        # 3. Translation Agent 준비
        translator = TranslationAgent.get_instance()

        # 4. 이벤트 핸들러 설정
        def on_recognizing(evt):
            """중간 인식 결과 (번역 없음)"""
            asyncio.create_task(websocket.send_json({
                "type": "recognizing",
                "text": evt.result.text,
                "confidence": None
            }))

        def on_recognized(evt):
            """최종 인식 결과 + 다국어 번역"""
            # 감지된 언어 추출
            detected_lang = evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult,
                "ko-KR"  # 기본값
            )

            logger.info(f"Detected language: {detected_lang}, text: {evt.result.text}")

            # 번역 대상 언어 (detected_lang 제외)
            target_bcp47_langs = [lang for lang in selected_languages if lang != detected_lang]
            target_iso_langs = [bcp47_to_iso639(lang) for lang in target_bcp47_langs]

            # 번역 비동기 실행 (별도 태스크)
            async def translate_and_send():
                translations = await translate_to_multiple_languages(
                    text=evt.result.text,
                    source_lang=bcp47_to_iso639(detected_lang),
                    target_langs=target_iso_langs,
                    translator=translator
                )

                # WebSocket으로 결과 전송
                await websocket.send_json({
                    "type": "recognized",
                    "text": evt.result.text,
                    "confidence": 1.0,
                    "detected_language": detected_lang,
                    "translations": translations
                })

            # 번역 태스크 실행
            asyncio.create_task(translate_and_send())

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

        # 5. 연속 인식 시작
        recognizer.start_continuous_recognition()
        logger.info("Continuous recognition with auto-detect started")

        # 6. 오디오 스트림 수신 루프
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

        # 7. 정리 및 종료
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
