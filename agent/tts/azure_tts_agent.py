"""
Azure TTS Agent (싱글톤 패턴)

Azure Speech SDK를 사용하여 텍스트를 음성으로 변환하는 Agent입니다.
SSML을 통한 고급 음성 제어를 지원합니다.
"""
import logging
import asyncio
from typing import Optional, Dict, Any
import azure.cognitiveservices.speech as speechsdk
from agent.base_agent import BaseAgent
from agent.stt_translation.azure_speech_agent import AzureSpeechAgent

logger = logging.getLogger(__name__)


class AzureTTSAgent(BaseAgent):
    """
    Azure TTS Agent (싱글톤)

    텍스트를 음성으로 변환하는 Agent입니다.

    Features:
    - Azure Speech SDK TTS
    - SSML 지원 (속도, 음높이, 음량 제어)
    - 다양한 뉴럴 음성 지원
    - 싱글톤 패턴

    Example:
        >>> agent = AzureTTSAgent.get_instance()
        >>> audio_data = await agent.process(
        ...     text="안녕하세요",
        ...     voice_name="ko-KR-SunHiNeural",
        ...     rate=1.0,
        ...     pitch=0,
        ...     volume=100
        ... )
        >>> # audio_data: bytes (WAV/PCM 형식)
    """

    _instance: Optional['AzureTTSAgent'] = None

    def __init__(self):
        """
        Initialize TTS Agent.

        Note: 직접 호출하지 말고 get_instance()를 사용하세요.
        """
        super().__init__()
        self.speech_agent = AzureSpeechAgent.get_instance()
        logger.info("Azure TTS Agent initialized")

    @classmethod
    def get_instance(cls) -> 'AzureTTSAgent':
        """
        싱글톤 인스턴스 반환

        Returns:
            AzureTTSAgent 싱글톤 인스턴스
        """
        if cls._instance is None:
            cls._instance = cls()
            logger.info("Created new AzureTTSAgent singleton instance")
        return cls._instance

    async def process(
        self,
        text: str,
        voice_name: str = "ko-KR-SunHiNeural",
        rate: float = 1.0,
        pitch: int = 0,
        volume: int = 100
    ) -> bytes:
        """
        텍스트를 음성으로 변환

        Args:
            text: 변환할 텍스트
            voice_name: Azure 뉴럴 음성 이름 (예: ko-KR-SunHiNeural, en-US-JennyNeural)
            rate: 말하기 속도 (0.5 - 2.0, 기본값 1.0)
            pitch: 음높이 (-50% ~ +50%, 기본값 0)
            volume: 음량 (0 - 100, 기본값 100)

        Returns:
            bytes: WAV 형식 오디오 데이터

        Raises:
            Exception: TTS 처리 실패 시
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return b''

        try:
            logger.info(f"Starting TTS: voice={voice_name}, rate={rate}, pitch={pitch}")

            # Azure Speech 토큰 가져오기
            token, region = await self.speech_agent.get_token()

            # Speech Config 생성 (토큰 기반)
            speech_config = speechsdk.SpeechConfig(
                subscription=None,
                region=region,
                auth_token=token
            )

            # 음성 이름 설정
            speech_config.speech_synthesis_voice_name = voice_name

            # 오디오 출력 설정 (메모리 스트림)
            audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False)

            # Speech Synthesizer 생성
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=audio_config
            )

            # SSML 생성 (고급 음성 제어)
            ssml = self._build_ssml(text, voice_name, rate, pitch, volume)

            # TTS 실행 (비동기)
            result = await asyncio.to_thread(synthesizer.speak_ssml, ssml)

            # 결과 처리
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                audio_data = result.audio_data
                logger.info(f"TTS success: generated {len(audio_data)} bytes")
                return audio_data

            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                error_msg = f"TTS canceled: {cancellation.reason}, {cancellation.error_details}"
                logger.error(error_msg)
                raise Exception(error_msg)

            else:
                error_msg = f"Unexpected TTS result reason: {result.reason}"
                logger.error(error_msg)
                raise Exception(error_msg)

        except Exception as e:
            logger.error(f"TTS processing failed: {str(e)}", exc_info=True)
            raise Exception(f"TTS 처리 실패: {str(e)}")

    def _build_ssml(
        self,
        text: str,
        voice_name: str,
        rate: float,
        pitch: int,
        volume: int
    ) -> str:
        """
        SSML (Speech Synthesis Markup Language) 생성

        고급 음성 제어를 위한 SSML 마크업을 생성합니다.

        Args:
            text: 변환할 텍스트
            voice_name: 음성 이름
            rate: 말하기 속도 (0.5 - 2.0)
            pitch: 음높이 (-50% ~ +50%)
            volume: 음량 (0 - 100)

        Returns:
            str: SSML 문자열
        """
        # 속도를 퍼센트로 변환 (1.0 = 100%)
        rate_percent = int(rate * 100)

        # 음높이가 범위 내에 있는지 확인
        pitch_percent = max(-50, min(50, pitch))

        # 음량 범위 확인
        volume = max(0, min(100, volume))

        # XML 특수 문자 이스케이프
        escaped_text = self._escape_xml(text)

        # SSML 생성
        ssml = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
            <voice name="{voice_name}">
                <prosody rate="{rate_percent}%" pitch="{'+' if pitch_percent >= 0 else ''}{pitch_percent}%" volume="{volume}">
                    {escaped_text}
                </prosody>
            </voice>
        </speak>
        """.strip()

        return ssml

    def _escape_xml(self, text: str) -> str:
        """
        XML 특수 문자 이스케이프

        Args:
            text: 이스케이프할 텍스트

        Returns:
            str: 이스케이프된 텍스트
        """
        return (
            text.replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&apos;')
        )

    async def process_with_options(
        self,
        text: str,
        options: Dict[str, Any]
    ) -> bytes:
        """
        옵션 딕셔너리를 사용한 TTS 처리

        Args:
            text: 변환할 텍스트
            options: TTS 옵션 딕셔너리 {
                "voice_name": str,
                "rate": float,
                "pitch": int,
                "volume": int
            }

        Returns:
            bytes: WAV 형식 오디오 데이터
        """
        voice_name = options.get('voice_name', 'ko-KR-SunHiNeural')
        rate = options.get('rate', 1.0)
        pitch = options.get('pitch', 0)
        volume = options.get('volume', 100)

        return await self.process(
            text=text,
            voice_name=voice_name,
            rate=rate,
            pitch=pitch,
            volume=volume
        )


# 싱글톤 인스턴스 생성 함수
def get_azure_tts_agent() -> AzureTTSAgent:
    """
    Azure TTS Agent 싱글톤 인스턴스 반환

    Returns:
        AzureTTSAgent: 싱글톤 인스턴스
    """
    return AzureTTSAgent.get_instance()
