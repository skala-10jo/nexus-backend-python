"""
Azure Speech STT Agent (싱글톤 패턴)

Azure Speech SDK를 사용하여 음성을 텍스트로 변환하는 Agent입니다.
"""
import logging
import asyncio
from typing import Optional, Dict, Any, List
import azure.cognitiveservices.speech as speechsdk
from agent.base_agent import BaseAgent
from app.core.azure_speech_token_manager import AzureSpeechTokenManager as AzureSpeechAgent

logger = logging.getLogger(__name__)


class STTAgent(BaseAgent):
    """
    Azure Speech STT Agent (싱글톤)

    음성을 텍스트로 변환하고 자동 언어 감지는 Agent입니다.

    Features:
    - Azure Speech SDK STT
    - 실시간 스트리밍 지원 (PushAudioInputStream)
    - 싱글톤 패턴

    Example:
        >>> agent = STTAgent.get_instance()
        >>> result = await agent.process(audio_data, language="ko-KR")
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
        language: str = "ko-KR"
    ) -> Dict[str, Any]:
        """
        음성을 텍스트로 변환

        Args:
            audio_data: 오디오 데이터 (WAV/PCM 형식)
            language: BCP-47 언어 코드 (예: ko-KR, en-US, ja-JP)

        Returns:
            Dict[str, Any]: {
                "text": str,           # 인식된 텍스트
                "confidence": float,   # 신뢰도 (0.0 ~ 1.0)
                "language": str        # 인식된 언어
            }

        Raises:
            Exception: STT 처리 실패 시
        """
        try:
            logger.info(f"Starting STT processing: language={language}")

            # Azure Speech 토큰 가져오기
            token, region = await self.speech_agent.get_token()

            # Speech Config 생성 (토큰 기반)
            speech_config = speechsdk.SpeechConfig(
                subscription=None,
                region=region,
                auth_token=token
            )
            speech_config.speech_recognition_language = language

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

                logger.info(f"STT success: text='{result.text[:50]}...'")
                return response

            elif result.reason == speechsdk.ResultReason.NoMatch:
                logger.warning("No speech recognized in audio")
                return {
                    "text": "",
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
        language: str = "ko-KR"
    ) -> tuple:
        """
        실시간 스트리밍 STT을 위한 Recognizer 및 PushStream 반환

        WebSocket에서 사용하기 위한 스트리밍 인터페이스입니다.

        Args:
            language: BCP-47 언어 코드

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
            logger.info(f"Setting up streaming STT: language={language}")

            # Azure Speech 토큰 가져오기
            token, region = await self.speech_agent.get_token()

            # Speech Config 생성
            speech_config = speechsdk.SpeechConfig(
                subscription=None,
                region=region,
                auth_token=token
            )
            speech_config.speech_recognition_language = language

            # PushAudioInputStream 생성 (포맷 지정 없음 - Azure가 자동 감지)
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

    async def process_stream_with_auto_detect(
        self,
        candidate_languages: List[str] = None
    ) -> tuple:
        """
        자동 언어 감지 + 실시간 스트리밍 STT

        모든 주요 언어를 자동 감지하여 음성을 텍스트로 변환합니다.
        candidate_languages는 번역 대상 언어 판별용으로만 사용되고,
        실제 STT 언어 감지는 모든 언어에 대해 자유롭게 수행됩니다.

        Args:
            candidate_languages: 번역 목표 언어 목록 (BCP-47 코드)
                이 파라미터는 번역 대상 필터링용이며, STT 감지 언어에는 영향 없음
                기본값: ["ko-KR", "en-US", "ja-JP", "vi-VN"]

        Returns:
            tuple: (recognizer, push_stream) - 호출자가 직접 관리

        Example:
            >>> recognizer, push_stream = await agent.process_stream_with_auto_detect(
            ...     candidate_languages=["ko-KR", "en-US", "ja-JP"]
            ... )
            >>> # on_recognized 이벤트에서 감지된 언어 추출:
            >>> # detected_lang = evt.result.properties.get(
            >>> #     speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
            >>> # )
        """
        if candidate_languages is None:
            candidate_languages = ["ko-KR", "en-US", "ja-JP", "vi-VN"]

        try:
            logger.info(f"Setting up auto-detect streaming STT")
            logger.info(f"Candidate languages: {candidate_languages}")

            # Azure Speech 토큰 가져오기
            token, region = await self.speech_agent.get_token()

            # Speech Config 생성
            speech_config = speechsdk.SpeechConfig(
                subscription=None,
                region=region,
                auth_token=token
            )

            # 자동 언어 감지 설정 (Azure 제한: 최대 4개 언어만 지원)
            # 사용자가 선택한 언어 중 최대 4개까지만 자동 감지에 사용
            languages_to_detect = candidate_languages[:4] if len(candidate_languages) > 4 else candidate_languages

            if len(candidate_languages) > 4:
                logger.warning(f"⚠️ Azure Speech SDK supports max 4 languages for auto-detect. Using first 4: {languages_to_detect}")

            auto_detect_config = speechsdk.AutoDetectSourceLanguageConfig(
                languages=languages_to_detect
            )

            # Continuous 언어 감지 모드 (문장마다 언어 재감지)
            speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
                "Continuous"
            )

            # 발화 종료 판단 시간 (기본 500ms → 300ms로 단축)
            speech_config.set_property(
                speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
                "300"
            )

            # 끝 침묵 타임아웃 (300ms)
            speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,
                "300"
            )

            # 초기 침묵 타임아웃 (기본 5000ms → 2000ms로 단축)
            # 처음 말하기 전 대기 시간
            speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
                "2000"
            )

            # 오디오 포맷 설정: 16kHz, 16bit, Mono PCM (프론트엔드 AudioWorklet과 일치)
            audio_format = speechsdk.audio.AudioStreamFormat(
                samples_per_second=16000,
                bits_per_sample=16,
                channels=1
            )

            # PushAudioInputStream 생성 (PCM 포맷 지정)
            push_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
            audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

            # Speech Recognizer 생성 (자동 언어 감지 포함)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                auto_detect_source_language_config=auto_detect_config,
                audio_config=audio_config
            )

            logger.info("Auto-detect streaming STT setup complete (PCM 16kHz 16bit Mono)")
            return recognizer, push_stream

        except Exception as e:
            logger.error(f"Auto-detect streaming STT setup failed: {str(e)}", exc_info=True)
            raise Exception(f"자동 감지 스트리밍 STT 설정 실패: {str(e)}")


# 싱글톤 인스턴스 생성 함수
def get_stt_agent() -> STTAgent:
    """
    STT Agent 싱글톤 인스턴스 반환

    Returns:
        STTAgent: 싱글톤 인스턴스
    """
    return STTAgent.get_instance()
