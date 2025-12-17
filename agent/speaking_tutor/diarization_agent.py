"""
Azure Conversation Transcription API를 사용한 화자 분리 에이전트.

오디오 파일을 분석하여 다음 정보를 제공:
- 화자별로 분리된 발화 내용
- 타임스탬프 (시작/종료 시간)
- 신뢰도 점수
- 화자 수

참고:
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/conversation-transcription
"""
import asyncio
import logging
import tempfile
import os
import wave
import threading
import time
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
    Azure Conversation Transcription 에이전트.

    오디오 파일을 처리하여 화자별로 분리된 텍스트를 반환.

    주의: Conversation Transcription은 특정 리전에서만 사용 가능:
    - eastasia, southeastasia, centralus, eastus, westeurope
    - 화자 분리 기능을 위해 AZURE_AVATAR_SPEECH_KEY/REGION (southeastasia) 사용
    """

    SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm'}

    def __init__(self):
        """Azure Speech 설정으로 초기화."""
        # Conversation Transcription을 위해 Avatar 키/리전 사용
        # koreacentral은 Conversation Transcription을 지원하지 않음
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
                "utterances": [
                    {
                        "speaker_id": 1,
                        "text": "Hello everyone",
                        "start_time_ms": 0,
                        "end_time_ms": 1500,
                        "confidence": 0.95,
                        "sequence_number": 0
                    },
                    ...
                ],
                "speaker_count": 3,
                "duration_seconds": 180.5
            }

        Raises:
            FileNotFoundError: 오디오 파일이 없을 때
            ValueError: 지원하지 않는 오디오 형식일 때
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
                progress_callback(20, "화자 분리 분석 시작...")

            result = await self._transcribe_with_push_stream(
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
        """
        오디오 파일을 Azure용 WAV 형식으로 변환.

        16kHz, 16bit, mono PCM 형식으로 변환.
        """
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)

        # 입력 파일의 채널 수 확인
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=channels',
            '-of', 'csv=p=0',
            input_path
        ]

        try:
            probe_process = await asyncio.create_subprocess_exec(
                *probe_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            probe_stdout, _ = await probe_process.communicate()
            input_channels = int(probe_stdout.decode().strip()) if probe_stdout.decode().strip() else 1
        except Exception:
            input_channels = 1  # 실패 시 모노로 가정

        # Azure는 16kHz mono를 권장
        output_channels = 1  # 화자 분리를 위해 mono 사용

        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ar', '16000',       # 16kHz 샘플레이트
            '-ac', str(output_channels),  # 모노
            '-acodec', 'pcm_s16le',  # 16bit PCM
            '-y',  # 덮어쓰기
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
                raise ValueError("오디오 변환 실패. ffmpeg가 설치되어 있는지 확인하세요.")

            logger.info(f"오디오 변환 완료: {input_channels}ch -> {output_channels}ch")
            return temp_path

        except FileNotFoundError:
            raise ValueError("ffmpeg를 찾을 수 없습니다. ffmpeg를 설치해주세요.")

    async def _transcribe_with_push_stream(
        self,
        wav_path: str,
        language: str,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        PushAudioInputStream을 사용하여 화자 분리 수행.

        Azure ConversationTranscriber는 파일 직접 입력보다
        스트림 방식이 더 안정적으로 동작함.
        """
        # WAV 파일 정보 읽기
        with wave.open(wav_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            audio_data = wav_file.readframes(n_frames)

        bits_per_sample = sample_width * 8
        duration_seconds = n_frames / sample_rate

        logger.info(
            f"WAV 파일 정보: {sample_rate}Hz, {channels}ch, "
            f"{bits_per_sample}bit, {len(audio_data)} bytes, {duration_seconds:.1f}초"
        )

        # Speech 설정 생성
        speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region
        )
        speech_config.speech_recognition_language = language

        # 화자 분리 향상을 위한 설정
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
            "15000"  # 15초 초기 침묵 허용
        )
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,
            "5000"   # 5초 종료 침묵 허용
        )

        # 단어 수준 타임스탬프 요청
        speech_config.request_word_level_timestamps()

        # PushAudioInputStream 생성
        audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=sample_rate,
            bits_per_sample=bits_per_sample,
            channels=channels
        )
        push_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

        # ConversationTranscriber 생성
        transcriber = speechsdk.transcription.ConversationTranscriber(
            speech_config=speech_config,
            audio_config=audio_config
        )

        # 결과 저장용
        utterances: List[Utterance] = []
        speaker_map: Dict[str, int] = {}
        done = asyncio.Event()
        errors: List[str] = []
        transcribing_started = threading.Event()

        def get_speaker_number(azure_speaker_id: str) -> int:
            """Azure 화자 ID를 순차 번호로 변환."""
            if not azure_speaker_id or azure_speaker_id.lower() == "unknown":
                return 1

            if azure_speaker_id not in speaker_map:
                speaker_map[azure_speaker_id] = len(speaker_map) + 1
            return speaker_map[azure_speaker_id]

        def handle_transcribing(evt):
            """중간 결과 처리 (디버그용)."""
            logger.debug(f"[중간] {evt.result.text[:50]}..." if evt.result.text else "[중간] (빈 텍스트)")

        def handle_transcribed(evt):
            """최종 인식 결과 처리."""
            try:
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    text = evt.result.text

                    if text and text.strip():
                        # 화자 ID 추출
                        azure_speaker_id = getattr(evt.result, 'speaker_id', None) or "Unknown"
                        speaker_num = get_speaker_number(azure_speaker_id)

                        # 타이밍 정보 추출
                        offset_ticks = evt.result.offset
                        duration_ticks = evt.result.duration

                        # 100나노초 단위 → 밀리초로 변환
                        start_ms = offset_ticks // 10000
                        duration_ms = duration_ticks // 10000

                        utterances.append(Utterance(
                            speaker_id=speaker_num,
                            text=text.strip(),
                            start_time_ms=start_ms,
                            end_time_ms=start_ms + duration_ms,
                            confidence=0.9,
                            sequence_number=len(utterances)
                        ))

                        logger.info(
                            f"✅ 발화 인식: 화자{speaker_num} ({azure_speaker_id}): "
                            f"{text[:50]}..."
                        )

                        if progress_callback:
                            progress = min(90, 30 + len(utterances) * 2)
                            progress_callback(
                                progress,
                                f"발화 {len(utterances)}개 인식됨 (화자 {len(speaker_map)}명)..."
                            )

            except Exception as e:
                logger.error(f"발화 처리 중 오류: {e}")

        def handle_canceled(evt):
            """취소/오류 처리."""
            logger.info(f"🔴 Canceled 이벤트: reason={evt.reason}")

            if evt.reason == speechsdk.CancellationReason.Error:
                error_code = getattr(evt, 'error_code', 'Unknown')
                error_details = getattr(evt, 'error_details', 'No details')
                error_msg = f"오류 코드: {error_code}, 상세: {error_details}"
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
            transcribing_started.set()

        # 이벤트 핸들러 연결
        transcriber.transcribing.connect(handle_transcribing)
        transcriber.transcribed.connect(handle_transcribed)
        transcriber.canceled.connect(handle_canceled)
        transcriber.session_stopped.connect(handle_session_stopped)
        transcriber.session_started.connect(handle_session_started)

        # 오디오 푸시를 위한 스레드 함수
        def push_audio_data():
            """별도 스레드에서 오디오 데이터를 청크로 푸시."""
            # 세션 시작 대기
            if not transcribing_started.wait(timeout=10):
                logger.error("세션 시작 타임아웃")
                push_stream.close()
                return

            logger.info(f"🎵 오디오 데이터 푸시 시작: {len(audio_data)} bytes")

            # 청크 크기: 100ms 분량의 오디오 (16kHz, 16bit, mono = 3200 bytes/100ms)
            chunk_size = int(sample_rate * sample_width * channels * 0.1)  # 100ms
            total_chunks = len(audio_data) // chunk_size

            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                push_stream.write(chunk)

                # 실시간 처리 시뮬레이션을 위해 약간의 딜레이
                # (너무 빠르게 푸시하면 Azure가 처리하지 못할 수 있음)
                time.sleep(0.05)  # 50ms 딜레이

                # 진행률 업데이트 (선택적)
                current_chunk = i // chunk_size
                if current_chunk % 50 == 0:  # 매 50청크마다
                    logger.debug(f"푸시 진행: {current_chunk}/{total_chunks}")

            logger.info("🎵 오디오 데이터 푸시 완료, 스트림 종료")
            push_stream.close()

        # Transcription 시작
        logger.info(f"🚀 화자 분리 시작: {wav_path}")
        transcriber.start_transcribing_async().get()

        # 별도 스레드에서 오디오 푸시
        push_thread = threading.Thread(target=push_audio_data, daemon=True)
        push_thread.start()

        # 완료 대기 (타임아웃: 오디오 길이 + 여유분)
        timeout = max(300, duration_seconds * 2 + 60)  # 최소 5분, 또는 오디오 길이의 2배 + 1분
        try:
            await asyncio.wait_for(done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"음성 인식 타임아웃 ({timeout}초)")
            errors.append(f"타임아웃 ({timeout}초)")

        # Transcription 종료
        transcriber.stop_transcribing_async().get()

        # 푸시 스레드 종료 대기
        push_thread.join(timeout=5)

        if errors:
            logger.error(f"음성 인식 오류 목록: {errors}")

        # 결과 계산
        result_duration = 0.0
        if utterances:
            result_duration = max(u.end_time_ms for u in utterances) / 1000.0
        else:
            result_duration = duration_seconds  # 발화가 없으면 원본 길이 사용

        unique_speakers = len(speaker_map) if speaker_map else (1 if utterances else 0)

        logger.info(
            f"📊 분석 완료: {len(utterances)}개 발화, "
            f"{unique_speakers}명 화자, {result_duration:.1f}초"
        )

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
