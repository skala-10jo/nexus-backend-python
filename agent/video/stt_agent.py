"""
Video STT Agent (Speech-to-Text)

OpenAI Whisper API를 사용하여 영상에서 음성을 텍스트로 변환하는 Agent.
ffmpeg를 통해 영상에서 오디오를 추출한 후 Whisper로 STT 수행.

재사용 시나리오:
- 영상 자막 생성
- 회의록 작성
- 강의 스크립트 추출
"""

import logging
import os
import tempfile
import subprocess
from typing import List, Dict, Any
from pathlib import Path
from agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class VideoSTTAgent(BaseAgent):
    """
    영상 음성 인식 Agent (Whisper API 사용)

    책임: 영상 파일 → 타임스탬프가 포함된 텍스트 세그먼트

    ffmpeg로 영상에서 오디오를 추출하고,
    OpenAI Whisper API로 음성을 텍스트로 변환합니다.

    예시:
        >>> agent = VideoSTTAgent()
        >>> segments = await agent.process(
        ...     video_file_path="/path/to/video.mp4",
        ...     source_language="ko"
        ... )
        >>> print(segments[0])
        {
            "sequence_number": 1,
            "start_time_ms": 0,
            "end_time_ms": 3500,
            "text": "안녕하세요...",
            "confidence": 0.95
        }
    """

    def _extract_audio_from_video(self, video_path: str, output_audio_path: str) -> None:
        """
        ffmpeg를 사용하여 영상에서 오디오 추출

        Args:
            video_path: 영상 파일 경로
            output_audio_path: 출력 오디오 파일 경로 (MP3)

        Raises:
            RuntimeError: ffmpeg 실행 실패 시
            FileNotFoundError: ffmpeg가 설치되지 않은 경우
        """
        logger.info(f"🎬 ffmpeg로 오디오 추출 시작: {video_path}")

        try:
            # ffmpeg 명령어: 영상 → MP3 추출 (오디오만, 비디오 제거)
            command = [
                'ffmpeg',
                '-i', video_path,           # 입력 영상
                '-vn',                       # 비디오 스트림 제거
                '-acodec', 'libmp3lame',    # MP3 코덱
                '-ar', '16000',             # 샘플링 레이트 16kHz (Whisper 최적화)
                '-ac', '1',                 # 모노 채널
                '-b:a', '64k',              # 비트레이트 64kbps
                '-y',                        # 덮어쓰기 허용
                output_audio_path
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=300  # 5분 제한
            )

            logger.info(f"✅ 오디오 추출 완료: {output_audio_path}")

        except FileNotFoundError:
            logger.error("❌ ffmpeg가 설치되지 않았습니다")
            raise FileNotFoundError(
                "ffmpeg가 설치되지 않았습니다. "
                "설치: brew install ffmpeg (macOS) 또는 apt install ffmpeg (Ubuntu)"
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ ffmpeg 실행 실패: {e.stderr.decode()}")
            raise RuntimeError(f"ffmpeg 오디오 추출 실패: {e.stderr.decode()}")
        except subprocess.TimeoutExpired:
            logger.error("❌ ffmpeg 타임아웃 (5분 초과)")
            raise RuntimeError("오디오 추출 시간 초과 (5분 제한)")

    def _convert_timestamp_to_ms(self, seconds: float) -> int:
        """
        초 단위 타임스탬프를 밀리초로 변환

        Args:
            seconds: 초 단위 시간

        Returns:
            밀리초 단위 시간
        """
        return int(seconds * 1000)

    async def process(
        self,
        video_file_path: str,
        source_language: str = "ko"
    ) -> List[Dict[str, Any]]:
        """
        영상 파일에서 음성을 텍스트로 변환

        Args:
            video_file_path: 영상 파일 경로 (MP4, AVI, MOV 등)
            source_language: 음성 언어 코드 (ko, en, ja, vi 등)

        Returns:
            타임스탬프가 포함된 텍스트 세그먼트 리스트:
            [
                {
                    "sequence_number": 1,
                    "start_time_ms": 0,
                    "end_time_ms": 3500,
                    "text": "안녕하세요...",
                    "confidence": 0.95
                },
                ...
            ]

        Raises:
            FileNotFoundError: 영상 파일이 존재하지 않을 때
            ValueError: 영상 파일 경로가 비어있을 때
            RuntimeError: STT 처리 실패 시
        """
        if not video_file_path or not video_file_path.strip():
            raise ValueError("영상 파일 경로가 비어있습니다")

        if not os.path.exists(video_file_path):
            raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_file_path}")

        logger.info(f"🎥 STT 처리 시작: {video_file_path} ({source_language})")

        temp_audio_path = None

        try:
            # Step 1: 임시 오디오 파일 생성
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name

            # Step 2: ffmpeg로 오디오 추출
            self._extract_audio_from_video(video_file_path, temp_audio_path)

            # Step 3: Whisper API 호출 (타임스탬프 포함)
            logger.info("🎤 Whisper API 호출 중...")

            with open(temp_audio_path, 'rb') as audio_file:
                response = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=source_language,
                    response_format="verbose_json",  # 타임스탬프 포함 형식
                    timestamp_granularities=["segment"]  # 세그먼트 단위 타임스탬프
                )

            # Step 4: 응답 파싱
            segments = []
            for idx, segment in enumerate(response.segments, start=1):
                segments.append({
                    "sequence_number": idx,
                    "start_time_ms": self._convert_timestamp_to_ms(segment.start),
                    "end_time_ms": self._convert_timestamp_to_ms(segment.end),
                    "text": segment.text.strip(),
                    "confidence": getattr(segment, 'avg_logprob', 0.0)  # Whisper는 avg_logprob 제공
                })

            logger.info(f"✅ STT 완료: {len(segments)}개 세그먼트 추출")

            return segments

        except Exception as e:
            logger.error(f"❌ STT 처리 실패: {str(e)}")
            raise RuntimeError(f"STT 처리 중 오류 발생: {str(e)}")

        finally:
            # 임시 파일 삭제
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                    logger.debug(f"🗑️ 임시 오디오 파일 삭제: {temp_audio_path}")
                except Exception as e:
                    logger.warning(f"⚠️ 임시 파일 삭제 실패: {str(e)}")
