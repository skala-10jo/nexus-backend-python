"""
Azure Speech TTS (Text-to-Speech) Agent

Azure Cognitive Services Speech SDK를 사용하여
텍스트를 음성으로 합성합니다.
"""
from typing import Dict
import azure.cognitiveservices.speech as speechsdk


class TTSAgent:
    """
    Text-to-Speech Agent

    Azure Speech Service를 사용하여 텍스트를 음성으로 변환합니다.
    """

    def __init__(self, speech_key: str, speech_region: str):
        """
        Args:
            speech_key: Azure Speech Service API Key
            speech_region: Azure 리전 (예: koreacentral)
        """
        self.speech_key = speech_key
        self.speech_region = speech_region

    def synthesize(
        self,
        text: str,
        voice_name: str = "en-US-JennyNeural",
        audio_format: str = "audio-16khz-32kbitrate-mono-mp3"
    ) -> Dict:
        """
        텍스트를 음성으로 합성합니다

        Args:
            text: 음성으로 변환할 텍스트
            voice_name: 음성 이름 (기본값: en-US-JennyNeural)
                - en-US-JennyNeural (여성, 자연스러운 음성)
                - en-US-GuyNeural (남성, 자연스러운 음성)
                - en-US-AriaNeural (여성, 뉴스 스타일)
            audio_format: 오디오 형식
                - audio-16khz-32kbitrate-mono-mp3 (기본값, 압축률 좋음)
                - audio-16khz-128kbitrate-mono-mp3 (고음질)
                - riff-16khz-16bit-mono-pcm (WAV 무손실)

        Returns:
            {
                "audio_data": bytes,  # 음성 데이터
                "audio_format": "mp3",  # 파일 확장자
                "duration_ms": 1500  # 재생 시간 (밀리초)
            }

        Raises:
            Exception: TTS 합성 실패 시
        """
        try:
            # 1. Speech Config
            speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key,
                region=self.speech_region
            )
            speech_config.speech_synthesis_voice_name = voice_name
            speech_config.set_speech_synthesis_output_format(
                self._get_output_format(audio_format)
            )

            # 2. Speech Synthesizer (audio_config=None이면 메모리로 반환)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=None  # None이면 result.audio_data로 받음
            )

            # 4. 음성 합성 실행
            result = synthesizer.speak_text_async(text).get()

            # 5. 결과 처리
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                audio_data = result.audio_data
                duration_ms = result.audio_duration.total_seconds() * 1000

                # 파일 확장자 결정
                file_extension = "mp3" if "mp3" in audio_format else "wav"

                return {
                    "audio_data": audio_data,
                    "audio_format": file_extension,
                    "duration_ms": int(duration_ms)
                }

            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                error_msg = f"TTS 합성 실패: {cancellation.reason}"
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    error_msg += f" - {cancellation.error_details}"
                raise Exception(error_msg)

            else:
                raise Exception(f"알 수 없는 결과 상태: {result.reason}")

        except Exception as e:
            raise Exception(f"TTS 합성 중 오류 발생: {str(e)}")

    def _get_output_format(self, format_str: str):
        """
        오디오 형식 문자열을 Azure SDK enum으로 변환
        """
        format_map = {
            "audio-16khz-32kbitrate-mono-mp3": speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3,
            "audio-16khz-128kbitrate-mono-mp3": speechsdk.SpeechSynthesisOutputFormat.Audio16Khz128KBitRateMonoMp3,
            "riff-16khz-16bit-mono-pcm": speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm,
        }
        return format_map.get(
            format_str,
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )
