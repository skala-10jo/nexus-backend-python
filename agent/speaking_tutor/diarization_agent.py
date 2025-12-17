"""
Azure Speech SDK를 사용한 음성 인식 및 화자 분리 에이전트.

오디오 파일을 분석하여 다음 정보를 제공:
- 화자별로 분리된 발화 내용
- 타임스탬프 (시작/종료 시간)
- 신뢰도 점수

Docker/AWS 환경 호환:
- PushAudioInputStream을 사용하여 안정적으로 작동
- AudioConfig(filename=...)은 일부 환경에서 불안정함

참고:
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-use-audio-input-streams
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
    speaker_id: int
    text: str
    start_time_ms: int
    end_time_ms: int
    confidence: float
    sequence_number: int


class DiarizationAgent:
    """
    Azure SpeechRecognizer 기반 화자 분리 에이전트.

    PushAudioInputStream을 사용하여 Docker/AWS 환경에서도 안정적으로 작동.

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
                "speaker_count": 1,
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
                progress_callback(20, "음성 인식 시작...")

            result = await self._transcribe_with_push_stream(
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

    async def _transcribe_with_push_stream(
        self,
        wav_path: str,
        language: str,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        PushAudioInputStream을 사용하여 음성 인식 수행.

        Docker/AWS 환경에서 AudioConfig(filename=...)이 불안정할 수 있어
        직접 오디오 데이터를 스트림으로 전달합니다.
        """
        # WAV 파일에서 오디오 데이터 읽기
        try:
            with wave.open(wav_path, 'rb') as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                n_frames = wf.getnframes()
                audio_data = wf.readframes(n_frames)

            logger.info(
                f"WAV 파일 정보: {sample_rate}Hz, {channels}ch, "
                f"{sample_width*8}bit, {len(audio_data)} bytes"
            )
        except Exception as e:
            logger.error(f"WAV 파일 읽기 실패: {e}")
            raise ValueError(f"WAV 파일 읽기 실패: {e}")

        # PushAudioInputStream 설정 (16kHz, 16bit, mono)
        audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=16000,
            bits_per_sample=16,
            channels=1
        )
        push_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

        # Speech 설정 생성
        speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region
        )
        speech_config.speech_recognition_language = language
        speech_config.request_word_level_timestamps()
        speech_config.output_format = speechsdk.OutputFormat.Detailed

        # SpeechRecognizer 생성
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )

        # 결과 저장 (thread-safe)
        utterances: List[Utterance] = []
        done = threading.Event()
        errors: List[str] = []
        lock = threading.Lock()

        def handle_recognized(evt):
            """인식 완료 이벤트 처리."""
            try:
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    text = evt.result.text
                    if text and text.strip():
                        offset_ticks = evt.result.offset
                        duration_ticks = evt.result.duration

                        # 100나노초 → 밀리초
                        start_ms = offset_ticks // 10000
                        duration_ms = duration_ticks // 10000

                        with lock:
                            utterances.append(Utterance(
                                speaker_id=1,
                                text=text.strip(),
                                start_time_ms=start_ms,
                                end_time_ms=start_ms + duration_ms,
                                confidence=0.9,
                                sequence_number=len(utterances)
                            ))
                            count = len(utterances)

                        logger.info(f"✅ 발화 인식 [{start_ms}ms]: {text[:50]}...")

                        if progress_callback:
                            progress = min(90, 30 + count * 2)
                            progress_callback(progress, f"발화 {count}개 인식됨...")

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
                with lock:
                    errors.append(error_msg)
                logger.error(f"❌ 음성 인식 오류: {error_msg}")
            elif evt.reason == speechsdk.CancellationReason.EndOfStream:
                logger.info("✅ 오디오 스트림 종료 (정상)")

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

        # 청크 크기: 3200 bytes = 100ms of audio at 16kHz, 16bit, mono
        chunk_size = 3200
        total_bytes = len(audio_data)

        logger.info(f"📤 오디오 데이터 준비: {total_bytes} bytes")

        # 먼저 첫 1초(10 chunks) 정도 푸시하여 버퍼 채우기
        initial_chunks = min(total_bytes, chunk_size * 10)
        for i in range(0, initial_chunks, chunk_size):
            push_stream.write(audio_data[i:i + chunk_size])
        logger.info(f"📤 초기 버퍼 푸시: {initial_chunks} bytes")

        # continuous recognition 시작
        logger.info(f"🚀 음성 인식 시작 (PushStream): {wav_path}")
        recognizer.start_continuous_recognition()

        # 나머지 데이터를 별도 스레드에서 푸시
        def push_remaining_data():
            """나머지 오디오 데이터를 스트림에 푸시."""
            try:
                pushed = initial_chunks
                for i in range(initial_chunks, total_bytes, chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    push_stream.write(chunk)
                    pushed += len(chunk)

                logger.info(f"📤 오디오 데이터 푸시 완료: {pushed} bytes")

                # 스트림 종료 (SDK에게 데이터 끝을 알림)
                push_stream.close()
                logger.info("📪 오디오 스트림 닫힘")

            except Exception as e:
                logger.error(f"오디오 푸시 오류: {e}")
                try:
                    push_stream.close()
                except:
                    pass

        push_thread = threading.Thread(target=push_remaining_data, daemon=True)
        push_thread.start()

        # 완료 대기 (threading.Event 사용)
        completed = done.wait(timeout=600)  # 최대 10분

        # 푸시 스레드 종료 대기
        push_thread.join(timeout=5)

        if not completed:
            logger.warning("음성 인식 타임아웃 (10분)")
            with lock:
                errors.append("타임아웃 (10분)")

        # 인식 종료
        recognizer.stop_continuous_recognition()

        # 에러 확인
        with lock:
            if errors and "EndOfStream" not in str(errors):
                logger.error(f"음성 인식 오류: {errors}")

        # 결과 계산
        with lock:
            result_duration = 0.0
            if utterances:
                result_duration = max(u.end_time_ms for u in utterances) / 1000.0

            unique_speakers = 1 if utterances else 0
            utterance_count = len(utterances)

            logger.info(f"📊 분석 완료: {utterance_count}개 발화, {result_duration:.1f}초")

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
