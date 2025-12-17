"""
Speaker Diarization Agent using Azure Conversation Transcription API.

Transcribes audio files with speaker separation, providing:
- Speaker-labeled utterances
- Timestamps (start/end)
- Confidence scores
- Speaker count

Reference:
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/conversation-transcription
"""
import asyncio
import logging
import subprocess
import tempfile
import os
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

import azure.cognitiveservices.speech as speechsdk
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Utterance:
    """Represents a single utterance from a speaker."""
    speaker_id: int
    text: str
    start_time_ms: int
    end_time_ms: int
    confidence: float
    sequence_number: int


class DiarizationAgent:
    """
    Azure Conversation Transcription Agent for speaker diarization.

    Processes audio files and returns speaker-separated transcriptions
    with timing information.

    Uses ConversationTranscriber for proper speaker diarization support.
    Does not inherit from BaseAgent as it uses Azure Speech SDK, not OpenAI.

    Note: Conversation Transcription is only available in specific regions:
    - eastasia, southeastasia, centralus, eastus, westeurope
    - We use AZURE_AVATAR_SPEECH_KEY/REGION (southeastasia) for this feature
    """

    SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm'}

    def __init__(self):
        """Initialize with Azure Speech configuration for Conversation Transcription."""
        # Use Avatar's speech key/region for Conversation Transcription
        # because it's in southeastasia which supports this feature
        # (koreacentral does NOT support Conversation Transcription)
        if not settings.AZURE_AVATAR_SPEECH_KEY:
            raise ValueError("AZURE_AVATAR_SPEECH_KEY is not configured (required for speaker diarization)")

        self.speech_key = settings.AZURE_AVATAR_SPEECH_KEY
        self.speech_region = settings.AZURE_AVATAR_SPEECH_REGION

        logger.info(f"DiarizationAgent initialized with region: {self.speech_region}")

    async def process(
        self,
        audio_file_path: str,
        language: str = "en-US",
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio file with speaker diarization.

        Args:
            audio_file_path: Path to the audio file
            language: Language code (e.g., 'en-US', 'ko-KR')
            progress_callback: Optional callback(progress_percent, message)

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
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio format is not supported
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        ext = os.path.splitext(audio_file_path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {ext}")

        if progress_callback:
            progress_callback(5, "오디오 파일 준비 중...")

        # Convert to WAV if necessary (Azure requires specific format)
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

            result = await self._transcribe_with_conversation_transcriber(
                wav_path,
                language,
                progress_callback
            )

            if progress_callback:
                progress_callback(100, "분석 완료")

            return result

        finally:
            # Cleanup temp file
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file: {e}")

    async def _convert_to_wav(self, input_path: str) -> str:
        """
        Convert audio file to WAV format for Azure.

        For speaker diarization, we preserve stereo if available,
        or keep mono with 16kHz sample rate.
        """
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)

        # First, check if input is stereo
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
            input_channels = 1  # Default to mono if probe fails

        # For diarization, 16kHz mono is recommended by Azure
        # But if stereo is available, we keep it as Azure can use channel separation
        output_channels = min(input_channels, 2)  # Max 2 channels

        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ar', '16000',  # 16kHz sample rate (Azure requirement)
            '-ac', str(output_channels),  # Preserve stereo if available
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-y',  # Overwrite
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
                logger.error(f"FFmpeg error: {stderr.decode()}")
                raise ValueError("Audio conversion failed. Please ensure ffmpeg is installed.")

            logger.info(f"Audio converted: {input_channels} -> {output_channels} channels")
            return temp_path

        except FileNotFoundError:
            raise ValueError("ffmpeg not found. Please install ffmpeg.")

    async def _transcribe_with_conversation_transcriber(
        self,
        wav_path: str,
        language: str,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Use Azure ConversationTranscriber for proper speaker diarization.

        ConversationTranscriber is Azure's dedicated API for multi-speaker
        transcription with automatic speaker identification.
        """
        # Create speech config
        speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region
        )
        speech_config.speech_recognition_language = language

        # Set properties for better diarization
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
            "15000"
        )
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,
            "5000"
        )

        # Request word-level timing for better accuracy
        speech_config.request_word_level_timestamps()

        # Audio config from file
        audio_config = speechsdk.AudioConfig(filename=wav_path)

        # Create ConversationTranscriber (key change from SpeechRecognizer)
        conversation_transcriber = speechsdk.transcription.ConversationTranscriber(
            speech_config=speech_config,
            audio_config=audio_config
        )

        # Results storage
        utterances: List[Utterance] = []
        speaker_map: Dict[str, int] = {}  # Map Azure speaker IDs to sequential numbers
        done = asyncio.Event()
        errors: List[str] = []

        def get_speaker_number(azure_speaker_id: str) -> int:
            """Convert Azure speaker ID (e.g., 'Guest-1') to sequential number."""
            if not azure_speaker_id or azure_speaker_id.lower() == "unknown":
                # If speaker is unknown, assign based on context or default
                return 1

            if azure_speaker_id not in speaker_map:
                speaker_map[azure_speaker_id] = len(speaker_map) + 1
            return speaker_map[azure_speaker_id]

        def handle_transcribed(evt):
            """Handle final transcription result with speaker ID."""
            try:
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    text = evt.result.text

                    if text and text.strip():
                        # Get speaker ID from ConversationTranscriber result
                        azure_speaker_id = getattr(evt.result, 'speaker_id', None) or "Unknown"
                        speaker_num = get_speaker_number(azure_speaker_id)

                        # Get timing from result
                        offset_ticks = evt.result.offset
                        duration_ticks = evt.result.duration

                        # Convert from 100-nanosecond units to milliseconds
                        start_ms = offset_ticks // 10000
                        duration_ms = duration_ticks // 10000

                        utterances.append(Utterance(
                            speaker_id=speaker_num,
                            text=text.strip(),
                            start_time_ms=start_ms,
                            end_time_ms=start_ms + duration_ms,
                            confidence=0.9,  # Azure doesn't expose confidence for CT
                            sequence_number=len(utterances)
                        ))

                        logger.debug(
                            f"Transcribed: Speaker {speaker_num} ({azure_speaker_id}): "
                            f"{text[:50]}..."
                        )

                        if progress_callback:
                            progress = min(90, 30 + len(utterances) * 2)
                            progress_callback(
                                progress,
                                f"발화 {len(utterances)}개 인식됨 (화자 {len(speaker_map)}명)..."
                            )

            except Exception as e:
                logger.error(f"Error in transcribed handler: {e}")

        def handle_canceled(evt):
            """Handle cancellation/error."""
            if evt.reason == speechsdk.CancellationReason.Error:
                error_msg = f"Error: {evt.error_details}"
                errors.append(error_msg)
                logger.error(f"Transcription error: {evt.error_details}")
            elif evt.reason == speechsdk.CancellationReason.EndOfStream:
                logger.info("End of audio stream reached")
            done.set()

        def handle_session_stopped(evt):
            """Handle session end."""
            logger.info("Transcription session stopped")
            done.set()

        def handle_session_started(evt):
            """Handle session start."""
            logger.info(f"Transcription session started: {evt.session_id}")

        # Connect event handlers
        conversation_transcriber.transcribed.connect(handle_transcribed)
        conversation_transcriber.canceled.connect(handle_canceled)
        conversation_transcriber.session_stopped.connect(handle_session_stopped)
        conversation_transcriber.session_started.connect(handle_session_started)

        # Start transcription - must call .get() to actually start
        logger.info(f"Starting conversation transcription for: {wav_path}")
        conversation_transcriber.start_transcribing_async().get()

        # Wait for completion with timeout
        try:
            # Set a reasonable timeout based on audio duration
            # For now, use 10 minutes max
            await asyncio.wait_for(done.wait(), timeout=600)
        except asyncio.TimeoutError:
            logger.warning("Transcription timed out after 10 minutes")
            errors.append("Transcription timed out")

        # Stop transcription
        conversation_transcriber.stop_transcribing_async().get()

        if errors:
            logger.error(f"Transcription errors: {errors}")

        # Calculate results
        duration_seconds = 0.0
        if utterances:
            duration_seconds = max(u.end_time_ms for u in utterances) / 1000.0

        # Get unique speakers count
        unique_speakers = len(speaker_map) if speaker_map else (1 if utterances else 0)

        logger.info(
            f"Transcription complete: {len(utterances)} utterances, "
            f"{unique_speakers} speakers, {duration_seconds:.1f}s duration"
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
            "duration_seconds": duration_seconds
        }

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes for conversation transcription."""
        # Languages supported by Azure Conversation Transcription
        # Reference: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support
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
