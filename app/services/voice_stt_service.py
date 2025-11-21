"""
음성 STT 서비스

Agent를 조율하고 비즈니스 로직을 처리합니다.
"""
import logging
from typing import Dict, Any
from agent.stt_translation.stt_agent import STTAgent

logger = logging.getLogger(__name__)


class VoiceSTTService:
    """
    음성 STT 비즈니스 로직 처리 서비스

    역할:
    - STT Agent 조율
    - 비즈니스 로직 처리 (향후 DB 저장, 사용량 추적 등)
    - 에러 핸들링 및 로깅
    """

    def __init__(self):
        """서비스 초기화"""
        self.agent = STTAgent.get_instance()
        logger.info("VoiceSTTService initialized")

    async def transcribe_audio(
        self,
        audio_data: bytes,
        language: str = "en-US",
        enable_diarization: bool = False,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        음성을 텍스트로 변환

        Args:
            audio_data: 오디오 데이터 (WAV 형식)
            language: 언어 코드 (기본값: en-US)
            enable_diarization: 화자 분리 활성화 여부
            user_id: 사용자 ID (사용량 추적용, 선택사항)

        Returns:
            Dict[str, Any]: {
                "text": str,           # 인식된 텍스트
                "confidence": float,   # 신뢰도 (0.0 ~ 1.0)
                "language": str,       # 언어 코드
                "speaker_id": str      # 화자 ID (diarization 활성화 시)
            }

        Raises:
            Exception: STT 처리 실패 시
        """
        try:
            logger.info(
                f"STT transcription started: "
                f"user={user_id or 'anonymous'}, "
                f"language={language}, "
                f"diarization={enable_diarization}"
            )

            # 1. Agent 호출 (순수 AI 로직)
            result = await self.agent.process(
                audio_data=audio_data,
                language=language,
                enable_diarization=enable_diarization
            )

            # 2. 비즈니스 로직 처리 (향후 확장)
            # - DB에 결과 저장 (user_id 사용)
            # - 사용량 추적
            # - 통계 수집
            # - 알림 발송 등

            # 향후 DB 저장 예시:
            # if user_id:
            #     transcription = Transcription(
            #         user_id=user_id,
            #         text=result['text'],
            #         confidence=result.get('confidence', 0.0),
            #         language=language
            #     )
            #     db.add(transcription)
            #     db.commit()

            logger.info(
                f"STT transcription completed: "
                f"user={user_id or 'anonymous'}, "
                f"text_length={len(result['text'])}, "
                f"confidence={result.get('confidence', 0.0):.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"STT transcription failed: {str(e)}", exc_info=True)
            raise

    async def get_supported_languages(self) -> Dict[str, str]:
        """
        지원하는 언어 목록 조회

        Returns:
            Dict[str, str]: 언어 코드 → 언어명 매핑
        """
        # 현재는 영어만 지원
        # 향후 DB나 설정 파일에서 동적으로 로드 가능
        supported_languages = {
            "en-US": "English (US)"
        }

        logger.debug(f"Supported languages requested: {len(supported_languages)} languages")
        return supported_languages
