"""
Azure Speech STT Agent (싱글톤 패턴)

Azure Speech SDK를 사용하여 음성을 텍스트로 변환하는 Agent입니다.
화자 분리(Speaker Diarization) 기능을 지원합니다.
"""
import logging
import asyncio
from typing import Optional, Dict, Any
import azure.cognitiveservices.speech as speechsdk
from agent.base_agent import BaseAgent
from agent.stt_translation.azure_speech_agent import AzureSpeechAgent

logger = logging.getLogger(__name__)


class STTAgent(BaseAgent):
    """
    Azure Speech STT Agent (싱글톤)

    음성을 텍스트로 변환하고 화자를 분리하는 Agent입니다.

    Features:
    - Azure Speech SDK STT
    - 화자 분리 (Speaker Diarization)
    - 실시간 스트리밍 지원 (PushAudioInputStream)
    - 싱글톤 패턴

    Example:
        >>> agent = STTAgent.get_instance()
        >>> result = await agent.process(audio_data, language="ko-KR", enable_diarization=True)
        >>> print(result["text"])
    """

    _instance: Optional['STTAgent'] = None

    def __init__(self):
        """
        Initialize STT Agent.

        Note: 직접 호출하지 말고 get_instance()를 사용하세요.
        """
        super().__init__()
        self.speech_agent = AzureSpeechAgent.get_instance()
        logger.info("STT Agent initialized")

    @classmethod
    def get_instance(cls) -> 'STTAgent':
        """
        싱글톤 인스턴스 반환

        Returns:
            STTAgent 싱글톤 인스턴스
        """
        if cls._instance is None:
            cls._instance = cls()
            logger.info("Created new STTAgent singleton instance")
        return cls._instance

    async def process(
        self,
        audio_data: bytes,
        language: str = "ko-KR",
        enable_diarization: bool = True
    ) -> Dict[str, Any]:
        """
        음성을 텍스트로 변환 (화자 분리 포함)

        Args:
            audio_data: 오디오 데이터 (WAV/PCM 형식)
            language: BCP-47 언어 코드 (예: ko-KR, en-US, ja-JP)
            enable_diarization: 화자 분리 활성화 여부

        Returns:
            Dict[str, Any]: {
                "text": str,           # 인식된 텍스트
                "speaker_id": str,     # 화자 ID (diarization 활성화 시)
                "confidence": float,   # 신뢰도 (0.0 ~ 1.0)
                "language": str        # 인식된 언어
            }

        Raises:
            Exception: STT 처리 실패 시
        """
        try:
            logger.info(f"Starting STT processing: language={language}, diarization={enable_diarization}")

            # Azure Speech 토큰 가져오기
            token, region = await self.speech_agent.get_token()

            # Speech Config 생성 (토큰 기반)
            speech_config = speechsdk.SpeechConfig(
                subscription=None,
                region=region,
                auth_token=token
            )
            speech_config.speech_recognition_language = language

            # 화자 분리 활성화
            if enable_diarization:
                speech_config.set_property(
                    speechsdk.PropertyId.SpeechServiceConnection_EnableSpeakerDiarization,
                    "true"
                )
                logger.info("Speaker diarization enabled")

            # PushAudioInputStream 생성 (스트리밍용)
            push_stream = speechsdk.audio.PushAudioInputStream()
            audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

            # Speech Recognizer 생성
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config
            )

            # 오디오 데이터 푸시
            push_stream.write(audio_data)
            push_stream.close()

            # 음성 인식 실행 (비동기)
            result = await asyncio.to_thread(recognizer.recognize_once)

            # 결과 처리
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                response = {
                    "text": result.text,
                    "confidence": result.confidence if hasattr(result, 'confidence') else 1.0,
                    "language": language
                }

                # 화자 분리 결과 추가
                if enable_diarization and hasattr(result, 'speaker_id'):
                    response["speaker_id"] = result.speaker_id
                else:
                    response["speaker_id"] = "Unknown"

                logger.info(f"STT success: text='{result.text[:50]}...', speaker={response['speaker_id']}")
                return response

            elif result.reason == speechsdk.ResultReason.NoMatch:
                logger.warning("No speech recognized in audio")
                return {
                    "text": "",
                    "speaker_id": "Unknown",
                    "confidence": 0.0,
                    "language": language
                }

            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                error_msg = f"STT canceled: {cancellation.reason}, {cancellation.error_details}"
                logger.error(error_msg)
                raise Exception(error_msg)

            else:
                error_msg = f"Unexpected STT result reason: {result.reason}"
                logger.error(error_msg)
                raise Exception(error_msg)

        except Exception as e:
            logger.error(f"STT processing failed: {str(e)}", exc_info=True)
            raise Exception(f"STT 처리 실패: {str(e)}")

    async def process_stream(
        self,
        language: str = "ko-KR",
        enable_diarization: bool = True
    ) -> tuple:
        """
        실시간 스트리밍 STT을 위한 Recognizer 및 PushStream 반환

        WebSocket에서 사용하기 위한 스트리밍 인터페이스입니다.

        Args:
            language: BCP-47 언어 코드
            enable_diarization: 화자 분리 활성화 여부

        Returns:
            tuple: (recognizer, push_stream) - 호출자가 직접 관리

        Example:
            >>> recognizer, push_stream = await agent.process_stream(language="ko-KR")
            >>> # WebSocket에서 오디오 청크 수신 시
            >>> push_stream.write(audio_chunk)
            >>> # 종료 시
            >>> push_stream.close()
            >>> recognizer.stop_continuous_recognition()
        """
        try:
            logger.info(f"Setting up streaming STT: language={language}, diarization={enable_diarization}")

            # Azure Speech 토큰 가져오기
            token, region = await self.speech_agent.get_token()

            # Speech Config 생성
            speech_config = speechsdk.SpeechConfig(
                subscription=None,
                region=region,
                auth_token=token
            )
            speech_config.speech_recognition_language = language

            # 화자 분리 활성화
            if enable_diarization:
                speech_config.set_property(
                    speechsdk.PropertyId.SpeechServiceConnection_EnableSpeakerDiarization,
                    "true"
                )

            # PushAudioInputStream 생성
            push_stream = speechsdk.audio.PushAudioInputStream()
            audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

            # Speech Recognizer 생성
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config
            )

            logger.info("Streaming STT setup complete")
            return recognizer, push_stream

        except Exception as e:
            logger.error(f"Streaming STT setup failed: {str(e)}", exc_info=True)
            raise Exception(f"스트리밍 STT 설정 실패: {str(e)}")


# 싱글톤 인스턴스 생성 함수
def get_stt_agent() -> STTAgent:
    """
    STT Agent 싱글톤 인스턴스 반환

    Returns:
        STTAgent: 싱글톤 인스턴스
    """
    return STTAgent.get_instance()
