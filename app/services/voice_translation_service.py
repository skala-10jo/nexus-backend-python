"""
실시간 음성 번역 서비스

역할:
- STT Agent 및 Translation Agent 관리
- 비즈니스 로직 처리 (Agent 조율)
- WebSocket Session은 이 Service를 통해서만 Agent 접근

AI Agent 아키텍처 가이드 준수:
- API → Service → Agent 계층 구조
- Service에서 Agent 인스턴스화 및 호출
"""

import logging
from typing import List, Dict, Any, Tuple
import azure.cognitiveservices.speech as speechsdk

from agent.stt_translation.stt_agent import STTAgent
from agent.stt_translation.translation_agent import TranslationAgent

logger = logging.getLogger(__name__)


class VoiceTranslationService:
    """
    실시간 음성 번역 비즈니스 로직

    책임:
    - Agent 인스턴스 생성 및 관리
    - STT 스트림 설정
    - 멀티 타겟 번역 수행

    금지:
    - DB 접근 (현재는 DB 저장 없음)
    - HTTP 응답 구성 (API 계층 역할)
    """

    def __init__(self):
        """Agent 인스턴스화 (싱글톤)"""
        self.stt_agent = STTAgent.get_instance()
        self.translation_agent = TranslationAgent.get_instance()
        logger.info("VoiceTranslationService initialized")

    async def setup_stream_with_auto_detect(
        self,
        candidate_languages: List[str]
    ) -> Tuple[speechsdk.SpeechRecognizer, speechsdk.audio.PushAudioInputStream]:
        """
        자동 언어 감지 기반 STT 스트림 설정

        Args:
            candidate_languages: 후보 언어 목록 (BCP-47 코드)
                예: ["ko-KR", "en-US", "ja-JP"]

        Returns:
            tuple: (recognizer, push_stream)
                - recognizer: Azure Speech Recognizer
                - push_stream: 오디오 입력 스트림

        Raises:
            Exception: STT 스트림 설정 실패 시
        """
        try:
            logger.info(f"Setting up STT stream with auto-detect: {candidate_languages}")

            # STT Agent를 통한 스트림 설정
            recognizer, push_stream = await self.stt_agent.process_stream_with_auto_detect(
                candidate_languages=candidate_languages
            )

            logger.info("STT stream setup complete")
            return recognizer, push_stream

        except Exception as e:
            logger.error(f"Failed to setup STT stream: {str(e)}", exc_info=True)
            raise Exception(f"STT 스트림 설정 실패: {str(e)}")

    async def translate_to_multiple_languages(
        self,
        text: str,
        source_lang: str,
        target_langs: List[str]
    ) -> List[Dict[str, str]]:
        """
        멀티 타겟 번역 (한 번의 API 호출)

        Args:
            text: 원본 텍스트
            source_lang: 원본 언어 (ISO 639-1 코드, 예: ko, en, ja)
            target_langs: 목표 언어 리스트 (ISO 639-1 코드)

        Returns:
            List[Dict[str, str]]: [
                {"lang": "en", "text": "Hello"},
                {"lang": "ja", "text": "こんにちは"}
            ]

        Raises:
            Exception: 번역 실패 시
        """
        try:
            logger.info(
                f"Multi-target translation: {source_lang} → {target_langs}, "
                f"text='{text[:50]}...'"
            )

            # Translation Agent를 통한 멀티 타겟 번역
            translations = await self.translation_agent.process_multi(
                text=text,
                source_lang=source_lang,
                target_langs=target_langs
            )

            logger.info(f"Translation complete: {len(translations)} languages")
            return translations

        except Exception as e:
            logger.error(f"Translation failed: {str(e)}", exc_info=True)
            raise Exception(f"번역 실패: {str(e)}")
