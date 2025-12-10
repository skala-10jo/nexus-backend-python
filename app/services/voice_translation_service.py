"""
실시간 음성 번역 서비스

역할:
- STT Agent 및 Translation Agent 관리
- 비즈니스 로직 처리 (Agent 조율)
- WebSocket Session은 이 Service를 통해서만 Agent 접근
- 프로젝트 선택 시 전문용어사전 반영 번역

AI Agent 아키텍처 가이드 준수:
- API → Service → Agent 계층 구조
- Service에서 Agent 인스턴스화 및 호출
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from uuid import UUID
from dataclasses import asdict
import azure.cognitiveservices.speech as speechsdk
from sqlalchemy.orm import Session
from sqlalchemy import text

from agent.stt_translation.stt_agent import STTAgent
from agent.stt_translation.translation_agent import TranslationAgent
from agent.term_detection.optimized_term_detector_agent import OptimizedTermDetectorAgent, DetectedTerm
from app.core.glossary_cache import glossary_cache

logger = logging.getLogger(__name__)


class VoiceTranslationService:
    """
    실시간 음성 번역 비즈니스 로직

    책임:
    - Agent 인스턴스 생성 및 관리
    - STT 스트림 설정
    - 멀티 타겟 번역 수행
    - 프로젝트별 용어집 후처리 적용

    금지:
    - HTTP 응답 구성 (API 계층 역할)
    """

    def __init__(self):
        """Agent 인스턴스화 (싱글톤)"""
        self.stt_agent = STTAgent.get_instance()
        self.translation_agent = TranslationAgent.get_instance()
        self.term_detector = OptimizedTermDetectorAgent()  # 용어 탐지 Agent 추가
        self._glossary_cache = glossary_cache
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

    def fetch_project_glossary(
        self,
        project_id: UUID,
        db: Session,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        프로젝트의 용어집 조회 (캐싱 적용)

        TranslationService._fetch_project_glossary()와 동일한 로직을 재사용합니다.

        Args:
            project_id: 프로젝트 ID
            db: DB 세션
            use_cache: 캐시 사용 여부 (기본값: True)

        Returns:
            용어집 리스트 (딕셔너리 형태)

        Note:
            - 캐시 TTL: 5분 (glossary_cache 기본값)
            - project_id = NULL인 용어도 조회하기 위해
              glossary_term_documents → project_files 경로로 조인합니다.

        Performance:
            - 캐시 히트 시: O(1)
            - 캐시 미스 시: DB 쿼리 수행
        """
        # 캐시 확인
        if use_cache:
            cached = self._glossary_cache.get(project_id)
            if cached is not None:
                logger.debug(f"📦 용어집 캐시 히트: project={project_id}, terms={len(cached)}개")
                return cached

        # Native SQL 쿼리: Java의 findTermsByProjectFiles()와 동일한 로직
        query = text("""
            SELECT DISTINCT t.*
            FROM glossary_terms t
            INNER JOIN glossary_term_documents gtd ON t.id = gtd.term_id
            INNER JOIN project_files pf ON gtd.file_id = pf.file_id
            WHERE pf.project_id = :project_id
        """)

        result = db.execute(query, {"project_id": str(project_id)})
        rows = result.fetchall()

        # Row를 딕셔너리로 변환
        glossary_terms = []
        for row in rows:
            glossary_terms.append({
                "id": row.id,
                "korean_term": row.korean_term,
                "english_term": row.english_term,
                "vietnamese_term": row.vietnamese_term,
                "japanese_term": getattr(row, 'japanese_term', None),
                "chinese_term": getattr(row, 'chinese_term', None),
                "definition": row.definition,
                "context": row.context,
                "example_sentence": row.example_sentence,
                "note": row.note,
                "domain": row.domain,
                "confidence_score": float(row.confidence_score) if row.confidence_score else None
            })

        # 캐시 저장
        if use_cache and glossary_terms:
            self._glossary_cache.set(project_id, glossary_terms)
            logger.debug(f"💾 용어집 캐시 저장: project={project_id}, terms={len(glossary_terms)}개")

        return glossary_terms

    def apply_glossary_to_translation(
        self,
        translation_text: str,
        source_lang: str,
        target_lang: str,
        glossary_terms: List[Dict[str, Any]]
    ) -> str:
        """
        번역 결과에 용어집 후처리 적용 (용어 치환)

        Azure Translator의 번역 결과에서 용어집에 등록된 용어를
        지정된 번역으로 치환합니다.

        Args:
            translation_text: Azure Translator 번역 결과
            source_lang: 원본 언어 (ISO 639-1: ko, en, ja, vi, zh)
            target_lang: 목표 언어 (ISO 639-1)
            glossary_terms: 용어집 리스트

        Returns:
            용어집이 적용된 번역 결과

        Example:
            >>> text = "Cloud computing is important"
            >>> glossary = [{"english_term": "Cloud computing", "korean_term": "클라우드 컴퓨팅"}]
            >>> result = apply_glossary_to_translation(text, "en", "ko", glossary)
            >>> # 번역 결과에서 "Cloud computing" 관련 번역이 "클라우드 컴퓨팅"으로 치환됨
        """
        if not glossary_terms or not translation_text:
            return translation_text

        # 언어 코드 → 용어집 필드 매핑
        lang_field_map = {
            "ko": "korean_term",
            "en": "english_term",
            "ja": "japanese_term",
            "vi": "vietnamese_term",
            "zh": "chinese_term"
        }

        target_field = lang_field_map.get(target_lang)
        if not target_field:
            logger.warning(f"⚠️ 지원하지 않는 목표 언어: {target_lang}")
            return translation_text

        replaced_text = translation_text
        replacement_count = 0

        # 긴 용어부터 치환 (더 긴 용어가 우선)
        sorted_terms = sorted(
            glossary_terms,
            key=lambda t: len(t.get(target_field, "") or ""),
            reverse=True
        )

        for term in sorted_terms:
            target_term = term.get(target_field)
            korean_term = term.get("korean_term")

            if not target_term or not korean_term:
                continue

            # 목표 언어 용어가 번역 결과에 있는지 확인
            # 대소문자 무시 검색 (영어의 경우)
            import re

            if target_lang == "en":
                # 영어: 단어 경계 포함 검색
                pattern = re.compile(r'\b' + re.escape(target_term) + r'\b', re.IGNORECASE)
            else:
                # 한국어/일본어/베트남어/중국어: 단순 문자열 검색
                pattern = re.compile(re.escape(target_term))

            if pattern.search(replaced_text):
                # 용어집에 정의된 번역으로 치환
                replaced_text = pattern.sub(target_term, replaced_text)
                replacement_count += 1
                logger.debug(f"🔄 용어 치환: '{korean_term}' → '{target_term}'")

        if replacement_count > 0:
            logger.info(f"✅ 용어집 후처리 완료: {replacement_count}개 용어 적용")

        return replaced_text

    async def detect_terms_in_text(
        self,
        text: str,
        source_lang: str,
        glossary_terms: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        텍스트에서 전문용어 탐지

        OptimizedTermDetectorAgent를 사용하여 원문에서 용어를 탐지합니다.

        Args:
            text: 분석할 텍스트
            source_lang: 원본 언어 (ISO 639-1 코드: ko, en, vi 등)
            glossary_terms: 용어집 리스트

        Returns:
            탐지된 용어 리스트 (딕셔너리 형태)
            [
                {
                    "matchedText": "클라우드",
                    "positionStart": 0,
                    "positionEnd": 4,
                    "koreanTerm": "클라우드",
                    "englishTerm": "Cloud",
                    "vietnameseTerm": "Đám mây",
                    "definition": "...",
                    "domain": "IT"
                }
            ]
        """
        if not glossary_terms or not text:
            return []

        try:
            # OptimizedTermDetectorAgent로 용어 탐지
            detected_terms = await self.term_detector.process(
                text=text,
                glossary_terms=glossary_terms,
                source_lang=source_lang
            )

            # DetectedTerm → 딕셔너리 변환 (프론트엔드 형식에 맞게)
            result = []
            for term in detected_terms:
                # 용어집에서 추가 정보 조회 (definition, domain 등)
                term_info = self._find_term_details(term.korean_term, glossary_terms)

                result.append({
                    "matchedText": term.matched_text,
                    "positionStart": term.position_start,
                    "positionEnd": term.position_end,
                    "koreanTerm": term.korean_term,
                    "englishTerm": term.english_term,
                    "vietnameseTerm": term.vietnamese_term,
                    "japaneseTerm": term_info.get("japanese_term"),
                    "chineseTerm": term_info.get("chinese_term"),
                    "definition": term_info.get("definition"),
                    "domain": term_info.get("domain"),
                    "glossaryTermId": str(term_info.get("id")) if term_info.get("id") else None
                })

            logger.info(f"🔍 용어 탐지 완료: {len(result)}개 용어 발견")
            return result

        except Exception as e:
            logger.error(f"용어 탐지 실패: {str(e)}", exc_info=True)
            return []

    def _find_term_details(
        self,
        korean_term: str,
        glossary_terms: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        용어집에서 해당 용어의 상세 정보 조회

        Args:
            korean_term: 한글 용어
            glossary_terms: 용어집 리스트

        Returns:
            용어 상세 정보 (없으면 빈 딕셔너리)
        """
        if not korean_term:
            return {}

        for term in glossary_terms:
            if term.get("korean_term") == korean_term:
                return term

        return {}

    async def translate_to_multiple_languages_with_glossary(
        self,
        text: str,
        source_lang: str,
        target_langs: List[str],
        project_id: Optional[UUID],
        db: Optional[Session]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        """
        멀티 타겟 번역 + 용어집 후처리 + 용어 탐지

        프로젝트가 선택된 경우 용어집을 조회하여 번역 결과에 후처리를 적용하고,
        원문에서 탐지된 전문용어 목록도 함께 반환합니다.

        Args:
            text: 원본 텍스트
            source_lang: 원본 언어 (ISO 639-1 코드)
            target_langs: 목표 언어 리스트 (ISO 639-1 코드)
            project_id: 프로젝트 ID (None이면 용어집 미적용)
            db: DB 세션 (project_id가 있는 경우 필수)

        Returns:
            Tuple[translations, detected_terms]:
            - translations: [{"lang": "en", "text": "Hello"}, ...]
            - detected_terms: [{"matchedText": "클라우드", ...}, ...]

        Raises:
            Exception: 번역 실패 시
        """
        detected_terms = []

        try:
            logger.info(
                f"Multi-target translation with glossary: {source_lang} → {target_langs}, "
                f"project={project_id}, text='{text[:50]}...'"
            )

            # 1. Azure Translator로 기본 번역 수행
            translations = await self.translation_agent.process_multi(
                text=text,
                source_lang=source_lang,
                target_langs=target_langs
            )

            # 2. 프로젝트가 선택된 경우 용어집 후처리 및 용어 탐지 적용
            if project_id and db:
                glossary_terms = self.fetch_project_glossary(project_id, db)

                if glossary_terms:
                    logger.info(f"📚 용어집 조회: {len(glossary_terms)}개 → 후처리 및 용어 탐지 적용")

                    # 2-1. 원문에서 용어 탐지
                    detected_terms = await self.detect_terms_in_text(
                        text=text,
                        source_lang=source_lang,
                        glossary_terms=glossary_terms
                    )

                    # 2-2. 각 번역 결과에 용어집 후처리 적용
                    for translation in translations:
                        target_lang = translation.get("lang")
                        original_text = translation.get("text")

                        processed_text = self.apply_glossary_to_translation(
                            translation_text=original_text,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            glossary_terms=glossary_terms
                        )

                        translation["text"] = processed_text
                else:
                    logger.info("📚 프로젝트에 연결된 용어집이 없습니다")

            logger.info(f"Translation complete: {len(translations)} languages, {len(detected_terms)} terms detected")
            return translations, detected_terms

        except Exception as e:
            logger.error(f"Translation with glossary failed: {str(e)}", exc_info=True)
            raise Exception(f"용어집 번역 실패: {str(e)}")
