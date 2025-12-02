"""
음성 STT 서비스

Agent를 조율하고 비즈니스 로직을 처리합니다.
"""
import logging
import asyncio
from typing import Dict, Any, AsyncGenerator
from agent.stt_translation.stt_agent import STTAgent
import azure.cognitiveservices.speech as speechsdk

logger = logging.getLogger(__name__)


class VoiceSTTService:
    """
    음성 STT 비즈니스 로직 처리 서비스

    역할:
    - STT Agent 조율
    - 비즈니스 로직 처리 (향후 DB 저장, 사용량 추적 등)
    - 에러 핸들링 및 로깅
    """

    def __init__(self):
        """서비스 초기화"""
        self.agent = STTAgent.get_instance()
        logger.info("VoiceSTTService initialized")

    async def transcribe_audio(
        self,
        audio_data: bytes,
        language: str = "en-US",
        enable_diarization: bool = False,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        음성을 텍스트로 변환

        Args:
            audio_data: 오디오 데이터 (WAV 형식)
            language: 언어 코드 (기본값: en-US)
            enable_diarization: 화자 분리 활성화 여부
            user_id: 사용자 ID (사용량 추적용, 선택사항)

        Returns:
            Dict[str, Any]: {
                "text": str,           # 인식된 텍스트
                "confidence": float,   # 신뢰도 (0.0 ~ 1.0)
                "language": str,       # 언어 코드
                "speaker_id": str      # 화자 ID (diarization 활성화 시)
            }

        Raises:
            Exception: STT 처리 실패 시
        """
        try:
            logger.info(
                f"STT transcription started: "
                f"user={user_id or 'anonymous'}, "
                f"language={language}, "
                f"diarization={enable_diarization}"
            )

            # 1. Agent 호출 (순수 AI 로직)
            result = await self.agent.process(
                audio_data=audio_data,
                language=language,
                enable_diarization=enable_diarization
            )

            # 2. 비즈니스 로직 처리 (향후 확장)
            # - DB에 결과 저장 (user_id 사용)
            # - 사용량 추적
            # - 통계 수집
            # - 알림 발송 등

            # 향후 DB 저장 예시:
            # if user_id:
            #     transcription = Transcription(
            #         user_id=user_id,
            #         text=result['text'],
            #         confidence=result.get('confidence', 0.0),
            #         language=language
            #     )
            #     db.add(transcription)
            #     db.commit()

            logger.info(
                f"STT transcription completed: "
                f"user={user_id or 'anonymous'}, "
                f"text_length={len(result['text'])}, "
                f"confidence={result.get('confidence', 0.0):.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"STT transcription failed: {str(e)}", exc_info=True)
            raise

    async def get_supported_languages(self) -> Dict[str, str]:
        """
        지원하는 언어 목록 조회

        Returns:
            Dict[str, str]: 언어 코드 → 언어명 매핑
        """
        # 현재는 영어만 지원
        # 향후 DB나 설정 파일에서 동적으로 로드 가능
        supported_languages = {
            "en-US": "English (US)"
        }

        logger.debug(f"Supported languages requested: {len(supported_languages)} languages")
        return supported_languages

    async def transcribe_stream(
        self,
        language: str = "en-US",
        enable_diarization: bool = False,
        user_id: str = None
    ) -> tuple:
        """
        실시간 스트리밍 STT을 위한 Recognizer, PushStream, 결과 Queue 반환

        WebSocket에서 사용하기 위한 스트리밍 인터페이스입니다.

        Args:
            language: 언어 코드 (기본값: en-US)
            enable_diarization: 화자 분리 활성화 여부
            user_id: 사용자 ID (사용량 추적용, 선택사항)

        Returns:
            tuple: (recognizer, push_stream, result_queue)
                - recognizer: SpeechRecognizer 인스턴스 (호출자가 관리)
                - push_stream: PushAudioInputStream (호출자가 write)
                - result_queue: asyncio.Queue (결과 수신용)

        Example:
            >>> recognizer, push_stream, queue = await service.transcribe_stream()
            >>> # WebSocket에서 오디오 청크 수신 시
            >>> push_stream.write(audio_chunk)
            >>> # 결과 큐에서 읽기
            >>> result = await queue.get()
            >>> # 종료 시
            >>> push_stream.close()
            >>> recognizer.stop_continuous_recognition()
        """
        try:
            logger.info(
                f"Streaming STT started: "
                f"user={user_id or 'anonymous'}, "
                f"language={language}, "
                f"diarization={enable_diarization}"
            )

            # 1. Agent 호출 (recognizer, push_stream 생성)
            # 자동 언어 감지 사용 (발화자 언어를 자동 인식)
            # enable_diarization은 Azure SDK 실시간 스트리밍에서 미지원
            recognizer, push_stream = await self.agent.process_stream_with_auto_detect(
                candidate_languages=["ko-KR", "en-US", "ja-JP", "vi-VN", "zh-CN"]
            )

            # 2. 결과 전달을 위한 asyncio Queue
            result_queue = asyncio.Queue()

            # 현재 event loop 참조 저장 (다른 스레드에서 사용)
            loop = asyncio.get_event_loop()

            # 3. 이벤트 핸들러 등록 (Azure SDK → Queue)
            def recognizing_handler(evt: speechsdk.SpeechRecognitionEventArgs):
                """실시간 인식 중 (interim result)"""
                try:
                    # 다른 스레드에서 안전하게 coroutine 실행
                    asyncio.run_coroutine_threadsafe(
                        result_queue.put({
                            "type": "interim",
                            "text": evt.result.text,
                            "confidence": None  # interim은 confidence 없음
                        }),
                        loop
                    )
                    logger.debug(f"Interim result: {evt.result.text}")
                except Exception as e:
                    logger.error(f"Error in recognizing_handler: {str(e)}")

            def recognized_handler(evt: speechsdk.SpeechRecognitionEventArgs):
                """최종 인식 완료 (final result)"""
                try:
                    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                        # 감지된 언어 추출 (자동 언어 감지 시)
                        detected_language = None
                        try:
                            detected_language = evt.result.properties.get(
                                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
                            )
                        except Exception:
                            pass

                        # 다른 스레드에서 안전하게 coroutine 실행
                        asyncio.run_coroutine_threadsafe(
                            result_queue.put({
                                "type": "final",
                                "text": evt.result.text,
                                "confidence": 0.95,  # Azure는 confidence를 직접 제공하지 않음
                                "detected_language": detected_language or language
                            }),
                            loop
                        )
                        logger.info(f"Final result: {evt.result.text}, language={detected_language}")
                except Exception as e:
                    logger.error(f"Error in recognized_handler: {str(e)}")

            def canceled_handler(evt: speechsdk.SpeechRecognitionCanceledEventArgs):
                """인식 취소 또는 에러"""
                try:
                    if evt.reason == speechsdk.CancellationReason.Error:
                        asyncio.run_coroutine_threadsafe(
                            result_queue.put({
                                "type": "error",
                                "message": f"Recognition error: {evt.error_details}"
                            }),
                            loop
                        )
                        logger.error(f"Recognition canceled: {evt.error_details}")
                except Exception as e:
                    logger.error(f"Error in canceled_handler: {str(e)}")

            def session_stopped_handler(evt: speechsdk.SessionEventArgs):
                """세션 종료"""
                try:
                    asyncio.run_coroutine_threadsafe(
                        result_queue.put({
                            "type": "session_stopped"
                        }),
                        loop
                    )
                    logger.info("Recognition session stopped")
                except Exception as e:
                    logger.error(f"Error in session_stopped_handler: {str(e)}")

            # 이벤트 핸들러 연결
            recognizer.recognizing.connect(recognizing_handler)
            recognizer.recognized.connect(recognized_handler)
            recognizer.canceled.connect(canceled_handler)
            recognizer.session_stopped.connect(session_stopped_handler)

            # 4. Continuous Recognition 시작
            recognizer.start_continuous_recognition()
            logger.info("Continuous recognition started")

            # 5. recognizer, push_stream, queue 반환 (호출자가 관리)
            return recognizer, push_stream, result_queue

        except Exception as e:
            logger.error(f"Streaming STT setup failed: {str(e)}", exc_info=True)
            raise
