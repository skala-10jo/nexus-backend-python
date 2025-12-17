"""
Azure Speech SDK를 사용한 음성 인식 및 화자 분리 에이전트.

ConversationTranscriber를 사용하여 실제 화자 분리(Speaker Diarization) 수행.

오디오 파일을 분석하여 다음 정보를 제공:
- 화자별로 분리된 발화 내용 (음성 특성 기반 자동 식별)
- 타임스탬프 (시작/종료 시간)
- 신뢰도 점수

Docker/AWS 환경 호환:
- PushAudioInputStream을 사용하여 안정적으로 작동

참고:
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-stt-diarization
"""
import asyncio
import logging
import tempfile
import os
import wave
import threading
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

import azure.cognitiveservices.speech as speechsdk
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Utterance:
    """단일 발화를 나타내는 데이터 클래스."""
    speaker_id: str  # Changed to str for ConversationTranscriber (Guest-1, Guest-2, etc.)
    text: str
    start_time_ms: int
    end_time_ms: int
    confidence: float
    sequence_number: int


class DiarizationAgent:
    """
    Azure ConversationTranscriber 기반 화자 분리 에이전트.

    ConversationTranscriber를 사용하여 음성 특성 기반 실제 화자 분리 수행.

    주의: 화자 분리 기능은 특정 리전에서만 사용 가능:
    - eastasia, southeastasia, centralus, eastus, westeurope
    - AZURE_AVATAR_SPEECH_KEY/REGION (southeastasia) 사용
    """

    SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm'}

    def __init__(self):
        """Azure Speech 설정으로 초기화."""
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
                "speaker_count": 2,
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

        # 항상 16kHz/16bit/mono WAV로 변환 (Docker 환경 호환성)
        if progress_callback:
            progress_callback(10, "오디오 포맷 변환 중...")

        wav_path = await self._convert_to_wav(audio_file_path)

        try:
            if progress_callback:
                progress_callback(20, "화자 분리 시작...")

            result = await self._transcribe_with_diarization(
                wav_path,
                language,
                progress_callback
            )

            if progress_callback:
                progress_callback(100, "분석 완료")

            return result

        finally:
            # 변환된 임시 파일 정리
            if wav_path != audio_file_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                    logger.debug(f"임시 파일 삭제: {wav_path}")
                except Exception as e:
                    logger.warning(f"임시 파일 삭제 실패: {e}")

    async def _convert_to_wav(self, input_path: str) -> str:
        """
        오디오 파일을 Azure용 WAV 형식으로 변환.

        항상 16kHz, 16bit, mono PCM으로 변환하여 일관성 보장.
        """
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)

        # Azure Speech SDK 요구 사항: 16kHz, 16bit, mono PCM
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ar', '16000',          # 16kHz 샘플레이트
            '-ac', '1',              # 모노
            '-acodec', 'pcm_s16le',  # 16bit PCM little-endian
            '-y',                    # 덮어쓰기
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

    async def _transcribe_with_diarization(
        self,
        wav_path: str,
        language: str,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        ConversationTranscriber를 사용하여 화자 분리 수행.
        """
        # WAV 파일 정보 확인
        try:
            with wave.open(wav_path, 'rb') as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                n_frames = wf.getnframes()
                duration_sec = n_frames / sample_rate

            logger.info(
                f"WAV 파일 정보: {sample_rate}Hz, {channels}ch, "
                f"{sample_width*8}bit, {duration_sec:.1f}초"
            )
        except Exception as e:
            logger.error(f"WAV 파일 읽기 실패: {e}")
            raise ValueError(f"WAV 파일 읽기 실패: {e}")

        # Speech 설정 생성
        speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region
        )
        speech_config.speech_recognition_language = language
        # Note: ConversationTranscriber는 자동으로 화자 분리를 수행함
        # SpeechServiceResponse_DiarizeIntermediateResults는 SpeechRecognizer용이므로 사용하지 않음

        # 파일 기반 AudioConfig
        audio_config = speechsdk.audio.AudioConfig(filename=wav_path)

        logger.info(f"🔧 ConversationTranscriber 설정: region={self.speech_region}, language={language}")

        # ConversationTranscriber 생성 (화자 분리 지원)
        transcriber = speechsdk.transcription.ConversationTranscriber(
            speech_config=speech_config,
            audio_config=audio_config
        )

        logger.info("✅ ConversationTranscriber 인스턴스 생성 완료")

        # 결과 저장 (thread-safe)
        utterances: List[Utterance] = []
        done = threading.Event()
        errors: List[str] = []
        lock = threading.Lock()
        speaker_set = set()

        def handle_transcribed(evt):
            """화자 분리된 발화 인식 이벤트 처리."""
            try:
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    text = evt.result.text
                    speaker_id = evt.result.speaker_id  # 실제 화자 ID (Guest-1, Guest-2, etc.)

                    if text and text.strip():
                        offset_ticks = evt.result.offset
                        duration_ticks = evt.result.duration

                        # 100나노초 → 밀리초
                        start_ms = offset_ticks // 10000
                        duration_ms = duration_ticks // 10000

                        with lock:
                            speaker_set.add(speaker_id)
                            utterances.append(Utterance(
                                speaker_id=speaker_id,
                                text=text.strip(),
                                start_time_ms=start_ms,
                                end_time_ms=start_ms + duration_ms,
                                confidence=0.9,
                                sequence_number=len(utterances)
                            ))
                            count = len(utterances)
                            num_speakers = len(speaker_set)

                        logger.info(
                            f"✅ [{speaker_id}] 발화 인식 [{start_ms}ms]: {text[:50]}..."
                        )

                        if progress_callback:
                            progress = min(90, 30 + count * 2)
                            progress_callback(
                                progress,
                                f"발화 {count}개 인식, 화자 {num_speakers}명 감지..."
                            )

                elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                    logger.debug(f"매칭 없음: {evt.result.no_match_details}")

            except Exception as e:
                logger.error(f"인식 처리 중 오류: {e}")

        def handle_canceled(evt):
            """취소/오류 처리."""
            try:
                # ConversationTranscriber의 canceled 이벤트 처리
                cancellation = speechsdk.CancellationDetails(evt.result)
                logger.info(f"🔴 Canceled: reason={cancellation.reason}")

                if cancellation.reason == speechsdk.CancellationReason.Error:
                    error_msg = f"코드: {cancellation.error_code}, 상세: {cancellation.error_details}"
                    with lock:
                        errors.append(error_msg)
                    logger.error(f"❌ 화자 분리 오류: {error_msg}")
                elif cancellation.reason == speechsdk.CancellationReason.EndOfStream:
                    logger.info("✅ 오디오 스트림 종료 (정상)")
            except Exception as e:
                logger.error(f"❌ canceled 이벤트 처리 중 오류: {e}")

            done.set()

        def handle_session_stopped(evt):
            """세션 종료 처리."""
            logger.info("🔵 세션 종료됨")
            done.set()

        def handle_session_started(evt):
            """세션 시작 처리."""
            logger.info(f"🟢 ConversationTranscriber 세션 시작됨: {evt.session_id}")

        # 이벤트 핸들러 연결
        transcriber.transcribed.connect(handle_transcribed)
        transcriber.canceled.connect(handle_canceled)
        transcriber.session_stopped.connect(handle_session_stopped)
        transcriber.session_started.connect(handle_session_started)

        # 비동기 화자 분리 시작
        logger.info(f"🚀 ConversationTranscriber 화자 분리 시작: {wav_path}")
        try:
            start_future = transcriber.start_transcribing_async()
            logger.info("⏳ start_transcribing_async() 호출 완료, 결과 대기 중...")
            start_future.get()
            logger.info("✅ 화자 분리 시작됨")
        except Exception as e:
            logger.error(f"❌ start_transcribing_async() 실패: {e}")
            raise

        # 완료 대기 (threading.Event 사용)
        logger.info("⏳ 화자 분리 완료 대기 중...")
        completed = done.wait(timeout=600)  # 최대 10분

        if not completed:
            logger.warning("⚠️ 화자 분리 타임아웃 (10분)")
            with lock:
                errors.append("타임아웃 (10분)")
        else:
            logger.info("✅ 화자 분리 이벤트 수신 완료")

        # 인식 종료
        logger.info("🛑 화자 분리 중지 중...")
        try:
            transcriber.stop_transcribing_async().get()
            logger.info("✅ 화자 분리 중지됨")
        except Exception as e:
            logger.warning(f"⚠️ stop_transcribing_async() 경고: {e}")

        # 에러 확인
        with lock:
            if errors and "EndOfStream" not in str(errors):
                logger.error(f"화자 분리 오류: {errors}")

        # 결과 계산
        with lock:
            result_duration = 0.0
            if utterances:
                result_duration = max(u.end_time_ms for u in utterances) / 1000.0

            # 화자 ID를 숫자로 변환 (Guest-1 → 1, Guest-2 → 2)
            speaker_mapping = {}
            for i, spk in enumerate(sorted(speaker_set), start=1):
                speaker_mapping[spk] = i

            unique_speakers = len(speaker_set)
            utterance_count = len(utterances)

            logger.info(
                f"📊 화자 분리 완료: {utterance_count}개 발화, "
                f"{unique_speakers}명 화자, {result_duration:.1f}초"
            )

            return {
                "utterances": [
                    {
                        "speaker_id": speaker_mapping.get(u.speaker_id, 1),
                        "speaker_label": u.speaker_id,  # 원본 화자 레이블 (Guest-1 등)
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
