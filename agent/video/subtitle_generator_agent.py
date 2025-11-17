"""
Subtitle Generator Agent

타임스탬프가 포함된 텍스트 세그먼트를 SRT 형식 자막 파일로 변환하는 Agent.

재사용 시나리오:
- 원본 자막 생성 (STT 결과)
- 번역된 자막 생성 (번역 결과)
- 다국어 자막 파일 생성
"""

import logging
from typing import List, Dict, Any
from pathlib import Path
from agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SubtitleGeneratorAgent(BaseAgent):
    """
    자막 파일 생성 Agent (SRT 형식)

    책임: 타임스탬프 세그먼트 → SRT 자막 파일

    STT 또는 번역 결과를 SRT(SubRip) 형식의 자막 파일로 생성합니다.

    예시:
        >>> agent = SubtitleGeneratorAgent()
        >>> file_path = await agent.process(
        ...     segments=[
        ...         {"sequence_number": 1, "start_time_ms": 0, "end_time_ms": 3500, "text": "안녕하세요"},
        ...         {"sequence_number": 2, "start_time_ms": 3500, "end_time_ms": 7000, "text": "반갑습니다"}
        ...     ],
        ...     output_path="/path/to/subtitle.srt"
        ... )
        >>> print(file_path)
        "/path/to/subtitle.srt"
    """

    def _format_timestamp(self, milliseconds: int) -> str:
        """
        밀리초를 SRT 타임스탬프 형식으로 변환

        Args:
            milliseconds: 밀리초 단위 시간

        Returns:
            SRT 형식 타임스탬프 (예: "00:00:03,500")
        """
        hours = milliseconds // 3600000
        milliseconds %= 3600000
        minutes = milliseconds // 60000
        milliseconds %= 60000
        seconds = milliseconds // 1000
        milliseconds %= 1000

        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def _validate_segments(self, segments: List[Dict[str, Any]]) -> None:
        """
        세그먼트 데이터 유효성 검증

        Args:
            segments: 텍스트 세그먼트 리스트

        Raises:
            ValueError: 세그먼트 데이터가 유효하지 않을 때
        """
        if not segments:
            raise ValueError("세그먼트 리스트가 비어있습니다")

        for idx, segment in enumerate(segments):
            required_fields = ["sequence_number", "start_time_ms", "end_time_ms", "text"]
            missing_fields = [field for field in required_fields if field not in segment]

            if missing_fields:
                raise ValueError(
                    f"세그먼트 {idx}에 필수 필드가 없습니다: {', '.join(missing_fields)}"
                )

            # 타임스탬프 검증
            start = segment["start_time_ms"]
            end = segment["end_time_ms"]

            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError(f"세그먼트 {idx}의 타임스탬프는 정수여야 합니다")

            if start < 0 or end < 0:
                raise ValueError(f"세그먼트 {idx}의 타임스탬프는 0 이상이어야 합니다")

            if start >= end:
                raise ValueError(
                    f"세그먼트 {idx}의 시작 시간({start}ms)이 종료 시간({end}ms)보다 크거나 같습니다"
                )

    def _generate_srt_content(self, segments: List[Dict[str, Any]]) -> str:
        """
        SRT 형식 자막 내용 생성

        Args:
            segments: 텍스트 세그먼트 리스트

        Returns:
            SRT 형식 자막 문자열
        """
        srt_lines = []

        for segment in segments:
            sequence = segment["sequence_number"]
            start = self._format_timestamp(segment["start_time_ms"])
            end = self._format_timestamp(segment["end_time_ms"])
            text = segment["text"].strip()

            # SRT 형식:
            # 1
            # 00:00:00,000 --> 00:00:03,500
            # 안녕하세요, 오늘은 인공지능에 대해 말씀드리겠습니다.
            #
            srt_lines.append(f"{sequence}")
            srt_lines.append(f"{start} --> {end}")
            srt_lines.append(text)
            srt_lines.append("")  # 빈 줄 (세그먼트 구분)

        return "\n".join(srt_lines)

    async def process(
        self,
        segments: List[Dict[str, Any]],
        output_path: str,
        subtitle_type: str = "original"
    ) -> str:
        """
        SRT 형식 자막 파일 생성

        Args:
            segments: 타임스탬프가 포함된 텍스트 세그먼트 리스트
                [
                    {
                        "sequence_number": 1,
                        "start_time_ms": 0,
                        "end_time_ms": 3500,
                        "text": "안녕하세요..."
                    },
                    ...
                ]
            output_path: 출력 파일 경로 (예: "/path/to/subtitle.srt")
            subtitle_type: 자막 유형 ("original" 또는 "translated")

        Returns:
            생성된 파일 경로

        Raises:
            ValueError: 세그먼트 데이터가 유효하지 않을 때
            OSError: 파일 쓰기 실패 시
        """
        logger.info(f"📝 자막 파일 생성 시작: {subtitle_type} ({len(segments)}개 세그먼트)")

        # Step 1: 데이터 검증
        self._validate_segments(segments)

        # Step 2: SRT 내용 생성
        srt_content = self._generate_srt_content(segments)

        # Step 3: 파일 저장
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)  # 디렉토리 생성

            output_file.write_text(srt_content, encoding="utf-8")

            logger.info(f"✅ 자막 파일 생성 완료: {output_path}")

            return str(output_file)

        except OSError as e:
            logger.error(f"❌ 파일 쓰기 실패: {str(e)}")
            raise OSError(f"자막 파일 저장 실패: {str(e)}")
