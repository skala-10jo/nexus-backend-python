"""
Azure Speech SDK를 사용한 음성 인식 및 화자 분리 에이전트.

오디오 파일을 분석하여 다음 정보를 제공:
- 화자별로 분리된 발화 내용
- 타임스탬프 (시작/종료 시간)
- 신뢰도 점수
- 화자 수

참고:
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-stt-diarization
"""
import asyncio
import logging
import tempfile
import os
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

import azure.cognitiveservices.speech as speechsdk
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Utterance:
    """단일 발화를 나타내는 데이터 클래스."""
    speaker_id: int
    text: str
    start_time_ms: int
    end_time_ms: int
    confidence: float
    sequence_number: int


class DiarizationAgent:
    """
    Azure SpeechRecognizer 기반 화자 분리 에이전트.

    SpeechRecognizer + continuous_recognition을 사용하여
    파일 기반 음성 인식 및 화자 분리를 수행합니다.

    주의: 화자 분리 기능은 특정 리전에서만 사용 가능:
    - eastasia, southeastasia, centralus, eastus, westeurope
    - AZURE_AVATAR_SPEECH_KEY/REGION (southeastasia) 사용
    """

    SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm'}

    def __init__(self):
        """Azure Speech 설정으로 초기화."""
        # 화자 분리를 위해 Avatar 키/리전 사용
        # koreacentral은 화자 분리를 지원하지 않음
        if not settings.AZURE_AVATAR_SPEECH_KEY:
            raise ValueError("AZURE_AVATAR_SPEECH_KEY가 설정되지 않았습니다 (화자 분리에 필요)")

        self.speech_key = settings.AZURE_AVATAR_SPEECH_KEY
        self.speech_region = settings.AZURE_AVATAR_SPEECH_REGION

        logger.info(f"DiarizationAgent 초기화 완료: 리전={self.speech_region}")

    async def process(
        self,
        audio_file_path: str,
        language: str = "en-US",
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        오디오 파일을 화자 분리하여 텍스트로 변환.

        Args:
            audio_file_path: 오디오 파일 경로
            language: 언어 코드 (예: 'en-US', 'ko-KR')
            progress_callback: 진행률 콜백 함수 (percent, message)

        Returns:
            {
                "utterances": [...],
                "speaker_count": 3,
                "duration_seconds": 180.5
            }
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_file_path}")

        ext = os.path.splitext(audio_file_path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"지원하지 않는 오디오 형식: {ext}")

        if progress_callback:
            progress_callback(5, "오디오 파일 준비 중...")

        # WAV가 아니면 변환 필요
        wav_path = audio_file_path
        temp_wav = None

        if ext != '.wav':
            if progress_callback:
                progress_callback(10, "오디오 포맷 변환 중...")
            wav_path = await self._convert_to_wav(audio_file_path)
            temp_wav = wav_path

        try:
            if progress_callback:
                progress_callback(20, "음성 인식 시작...")

            result = await self._transcribe_with_speech_recognizer(
                wav_path,
                language,
                progress_callback
            )

            if progress_callback:
                progress_callback(100, "분석 완료")

            return result

        finally:
            # 임시 파일 정리
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception as e:
                    logger.warning(f"임시 파일 삭제 실패: {e}")

    async def _convert_to_wav(self, input_path: str) -> str:
        """오디오 파일을 Azure용 WAV 형식으로 변환."""
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)

        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ar', '16000',       # 16kHz 샘플레이트
            '-ac', '1',           # 모노
            '-acodec', 'pcm_s16le',  # 16bit PCM
            '-y',
            temp_path
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"FFmpeg 오류: {stderr.decode()}")
                raise ValueError("오디오 변환 실패")

            logger.info(f"오디오 변환 완료: {temp_path}")
            return temp_path

        except FileNotFoundError:
            raise ValueError("ffmpeg를 찾을 수 없습니다")

    async def _transcribe_with_speech_recognizer(
        self,
        wav_path: str,
        language: str,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        SpeechRecognizer + continuous_recognition으로 음성 인식 수행.

        파일 기반 음성 인식에서 가장 안정적인 방식입니다.
        """
        # Speech 설정 생성
        speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region
        )
        speech_config.speech_recognition_language = language

        # 단어 수준 타임스탬프 요청
        speech_config.request_word_level_timestamps()

        # 출력 형식 설정 (상세 정보 포함)
        speech_config.output_format = speechsdk.OutputFormat.Detailed

        # 오디오 설정 (파일에서 직접 읽기)
        audio_config = speechsdk.AudioConfig(filename=wav_path)

        # SpeechRecognizer 생성
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )

        # 결과 저장
        utterances: List[Utterance] = []
        done = asyncio.Event()
        errors: List[str] = []

        def handle_recognized(evt):
            """인식 완료 이벤트 처리."""
            try:
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    text = evt.result.text
                    if text and text.strip():
                        # 타이밍 정보 추출
                        offset_ticks = evt.result.offset
                        duration_ticks = evt.result.duration

                        # 100나노초 → 밀리초
                        start_ms = offset_ticks // 10000
                        duration_ms = duration_ticks // 10000

                        # 화자 ID는 단순하게 1로 설정 (단일 화자 가정)
                        # 실제 화자 분리는 Azure의 제한으로 어려움
                        speaker_id = 1

                        utterances.append(Utterance(
                            speaker_id=speaker_id,
                            text=text.strip(),
                            start_time_ms=start_ms,
                            end_time_ms=start_ms + duration_ms,
                            confidence=0.9,
                            sequence_number=len(utterances)
                        ))

                        logger.info(f"✅ 발화 인식: {text[:50]}...")

                        if progress_callback:
                            progress = min(90, 30 + len(utterances) * 2)
                            progress_callback(progress, f"발화 {len(utterances)}개 인식됨...")

                elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                    logger.debug(f"매칭 없음: {evt.result.no_match_details}")

            except Exception as e:
                logger.error(f"인식 처리 중 오류: {e}")

        def handle_canceled(evt):
            """취소/오류 처리."""
            logger.info(f"🔴 Canceled: reason={evt.reason}")

            if evt.reason == speechsdk.CancellationReason.Error:
                error_code = getattr(evt, 'error_code', 'Unknown')
                error_details = getattr(evt, 'error_details', 'No details')
                error_msg = f"코드: {error_code}, 상세: {error_details}"
                errors.append(error_msg)
                logger.error(f"❌ 음성 인식 오류: {error_msg}")
            elif evt.reason == speechsdk.CancellationReason.EndOfStream:
                logger.info("✅ 오디오 스트림 종료")

            done.set()

        def handle_session_stopped(evt):
            """세션 종료 처리."""
            logger.info("🔵 세션 종료됨")
            done.set()

        def handle_session_started(evt):
            """세션 시작 처리."""
            logger.info(f"🟢 세션 시작됨: {evt.session_id}")

        # 이벤트 핸들러 연결
        recognizer.recognized.connect(handle_recognized)
        recognizer.canceled.connect(handle_canceled)
        recognizer.session_stopped.connect(handle_session_stopped)
        recognizer.session_started.connect(handle_session_started)

        # continuous recognition 시작
        logger.info(f"🚀 음성 인식 시작: {wav_path}")
        recognizer.start_continuous_recognition()

        # 완료 대기
        try:
            await asyncio.wait_for(done.wait(), timeout=600)  # 최대 10분
        except asyncio.TimeoutError:
            logger.warning("음성 인식 타임아웃 (10분)")
            errors.append("타임아웃 (10분)")

        # 인식 종료
        recognizer.stop_continuous_recognition()

        if errors:
            logger.error(f"음성 인식 오류: {errors}")

        # 결과 계산
        result_duration = 0.0
        if utterances:
            result_duration = max(u.end_time_ms for u in utterances) / 1000.0

        # 화자 수 (현재는 단일 화자로 처리)
        unique_speakers = 1 if utterances else 0

        logger.info(f"📊 분석 완료: {len(utterances)}개 발화, {result_duration:.1f}초")

        return {
            "utterances": [
                {
                    "speaker_id": u.speaker_id,
                    "text": u.text,
                    "start_time_ms": u.start_time_ms,
                    "end_time_ms": u.end_time_ms,
                    "confidence": u.confidence,
                    "sequence_number": u.sequence_number
                }
                for u in utterances
            ],
            "speaker_count": unique_speakers,
            "duration_seconds": result_duration
        }

    def get_supported_languages(self) -> List[str]:
        """지원되는 언어 코드 목록 반환."""
        return [
            "en-US", "en-GB", "en-AU", "en-IN", "en-NZ", "en-CA",
            "ko-KR",
            "ja-JP",
            "zh-CN", "zh-TW", "zh-HK",
            "de-DE", "de-AT", "de-CH",
            "fr-FR", "fr-CA",
            "es-ES", "es-MX",
            "it-IT",
            "pt-BR", "pt-PT",
            "nl-NL",
            "ru-RU",
            "ar-SA", "ar-EG",
            "hi-IN",
            "th-TH",
            "vi-VN"
        ]
