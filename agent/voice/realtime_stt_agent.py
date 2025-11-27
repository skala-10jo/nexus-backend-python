"""
실시간 음성-텍스트 변환 Agent (OpenAI Whisper API)

최적화 기법:
- 작은 청크 처리 (500ms - 2초)
- 임시 파일 즉시 삭제
- 에러 처리 강화
"""
import tempfile
import base64
import os
from typing import Dict
from agent.base_agent import BaseAgent


class RealtimeSTTAgent(BaseAgent):
    """
    실시간 STT Agent

    OpenAI Whisper API를 사용하여 오디오를 텍스트로 변환합니다.
    실시간 처리를 위해 작은 청크 단위로 최적화되었습니다.
    """

    def __init__(self):
        super().__init__()
        self.supported_languages = {
            'ko': '한국어',
            'en': '영어',
            'vi': '베트남어'
        }

    async def process(
        self,
        audio_data,  # bytes 또는 str (base64) 모두 허용
        input_language: str = 'ko',
        audio_format: Dict = None
    ) -> Dict[str, any]:
        """
        오디오 청크를 텍스트로 변환

        Args:
            audio_data: 오디오 데이터 (bytes 또는 base64 인코딩된 문자열)
            input_language: 입력 언어 코드 (ko, en, vi)
            audio_format: 오디오 형식 정보 (mimeType, extension)
                예: {"mimeType": "audio/webm;codecs=opus", "extension": "webm"}

        Returns:
            {
                "text": "변환된 텍스트",
                "confidence": 0.95,
                "language": "ko",
                "duration": 1.5  # 오디오 길이 (초)
            }

        Raises:
            ValueError: 지원하지 않는 언어일 때
            Exception: Whisper API 호출 실패 시
        """
        # 1. 언어 검증
        if input_language not in self.supported_languages:
            raise ValueError(
                f"지원하지 않는 언어입니다: {input_language}. "
                f"지원 언어: {list(self.supported_languages.keys())}"
            )

        # 2. audio_data 타입에 따라 처리
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"🔍 RealtimeSTTAgent.process: type={type(audio_data)}, isinstance(str)={isinstance(audio_data, str)}, isinstance(bytes)={isinstance(audio_data, bytes)}")

        if isinstance(audio_data, str):
            # base64 인코딩된 문자열인 경우
            logger.debug(f"🔍 Attempting base64 decode of string (length={len(audio_data)})")
            try:
                audio_bytes = base64.b64decode(audio_data)
                logger.debug(f"🔍 Successfully decoded base64, resulting bytes length={len(audio_bytes)}")
            except Exception as e:
                logger.error(f"🔍 Base64 decode failed: {str(e)}")
                raise ValueError(f"오디오 데이터 디코딩 실패: {str(e)}")
        elif isinstance(audio_data, bytes):
            # 이미 bytes인 경우 (WebSocket binary message)
            logger.debug(f"🔍 Using raw bytes (length={len(audio_data)})")
            audio_bytes = audio_data
        else:
            raise ValueError(f"지원하지 않는 오디오 데이터 타입: {type(audio_data)}")

        # 오디오가 너무 작으면 무시 (< 0.5초, 대략 8KB)
        if len(audio_bytes) < 8000:
            return {
                "text": "",
                "confidence": 0.0,
                "language": input_language,
                "duration": 0.0
            }

        # 3. 오디오 형식에 맞는 확장자 결정
        # Whisper API는 mp3, mp4, mpeg, mpga, m4a, wav, webm 모두 지원
        if audio_format and "extension" in audio_format:
            file_extension = audio_format["extension"]
        else:
            # 기본값: webm
            file_extension = "webm"

        # 4. 임시 파일로 저장
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=f'.{file_extension}',
                delete=False
            ) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name

            # 4. Whisper API 호출
            with open(temp_file_path, 'rb') as audio_file:
                transcript = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=input_language,
                    response_format="verbose_json",  # 상세 정보 포함
                    temperature=0.0,  # 결정론적 결과
                    prompt="이것은 일상 대화입니다."  # Hallucination 방지 프롬프트
                )

            # 5. 결과 파싱
            text = transcript.text.strip()

            # Hallucination 감지: segments의 no_speech_prob 체크
            segments = getattr(transcript, 'segments', [])
            if segments:
                # 모든 segment의 평균 no_speech_prob 계산
                # TranscriptionSegment 객체이므로 getattr 사용
                avg_no_speech_prob = sum(
                    getattr(seg, 'no_speech_prob', 0.0) for seg in segments
                ) / len(segments)

                # no_speech_prob가 0.85 이상이면 무음/hallucination으로 간주
                if avg_no_speech_prob > 0.85:
                    return {
                        "text": "",
                        "confidence": 0.0,
                        "language": input_language,
                        "duration": getattr(transcript, 'duration', 0.0)
                    }

            # Confidence 계산
            confidence = 0.9 if text else 0.0

            return {
                "text": text,
                "confidence": confidence,
                "language": transcript.language or input_language,
                "duration": getattr(transcript, 'duration', 0.0)
            }

        except Exception as e:
            # Whisper API 에러 처리
            error_msg = str(e)

            # 일반적인 에러 메시지를 한글로 변환
            if "Invalid file format" in error_msg or "invalid_file_format" in error_msg.lower():
                raise Exception(
                    f"지원하지 않는 오디오 형식입니다. "
                    f"형식: {audio_format.get('mimeType', 'unknown') if audio_format else 'unknown'}, "
                    f"확장자: {file_extension}. "
                    f"Whisper API는 mp3, mp4, wav, webm 등을 지원합니다. "
                    f"원본 에러: {error_msg}"
                )
            elif "File too large" in error_msg:
                raise Exception("오디오 파일이 너무 큽니다. (최대 25MB)")
            elif "No audio data" in error_msg:
                # 무음 구간은 빈 결과 반환
                return {
                    "text": "",
                    "confidence": 0.0,
                    "language": input_language,
                    "duration": 0.0
                }
            else:
                raise Exception(f"음성 인식 실패: {error_msg}")

        finally:
            # 6. 임시 파일 즉시 삭제 (보안 및 디스크 공간 관리)
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass  # 삭제 실패는 무시

    def get_supported_languages(self) -> Dict[str, str]:
        """
        지원하는 언어 목록 반환

        Returns:
            {
                'ko': '한국어',
                'en': '영어',
                'vi': '베트남어'
            }
        """
        return self.supported_languages.copy()
