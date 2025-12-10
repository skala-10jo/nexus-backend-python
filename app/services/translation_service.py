"""
번역 Service

여러 Micro Agent를 조율하여 프로젝트 컨텍스트 기반 번역을 제공합니다.

성능 최적화:
- OptimizedTermDetectorAgent: Aho-Corasick 알고리즘 기반 O(M+Z) 용어 탐지
- GlossaryCache: 프로젝트별 용어집 TTL 캐싱
"""

import logging
from typing import Optional, Dict, List, Any
from uuid import UUID
from sqlalchemy.orm import Session

# Agent imports
from agent.translate.simple_translation_agent import SimpleTranslationAgent
from agent.translate.context_enhanced_translation_agent import ContextEnhancedTranslationAgent
from agent.term_detection.optimized_term_detector_agent import OptimizedTermDetectorAgent
from agent.term_detection.glossary_matcher_agent import GlossaryMatcherAgent
from agent.summarization.document_summarizer_agent import DocumentSummarizerAgent

# Core imports
from app.core.glossary_cache import glossary_cache

# Model imports
from app.models.translation import Translation, TranslationTerm
from app.models.glossary import GlossaryTerm
from app.models.file import File

logger = logging.getLogger(__name__)


class TranslationService:
    """
    번역 비즈니스 로직

    책임:
    - Micro Agent들을 조율하여 번역 수행
    - 프로젝트 컨텍스트 및 용어집 데이터 조회
    - 번역 결과를 DB에 저장
    - 탐지된 용어 매핑 관리

    성능 최적화:
    - OptimizedTermDetectorAgent: Aho-Corasick 기반 O(M+Z) 용어 탐지
    - GlossaryCache: 프로젝트별 용어집 캐싱 (TTL 5분)
    """

    def __init__(self):
        """Micro Agent 인스턴스화"""
        self.simple_translator = SimpleTranslationAgent()
        self.context_translator = ContextEnhancedTranslationAgent()
        # 최적화된 용어 탐지 Agent 사용 (Aho-Corasick 알고리즘)
        self.term_detector = OptimizedTermDetectorAgent()
        self.glossary_matcher = GlossaryMatcherAgent()
        self.document_summarizer = DocumentSummarizerAgent()
        # 용어집 캐시 참조
        self._glossary_cache = glossary_cache

    def _fetch_project_glossary(
        self,
        project_id: UUID,
        db: Session,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        프로젝트의 용어집 조회 (캐싱 적용)

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

        from sqlalchemy import text

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

    def _fetch_project_documents_text(
        self,
        project_id: UUID,
        db: Session,
        max_docs: int = 5
    ) -> List[str]:
        """
        프로젝트의 문서 텍스트 조회

        Args:
            project_id: 프로젝트 ID
            db: DB 세션
            max_docs: 최대 문서 수 (기본값: 5개)

        Returns:
            문서 텍스트 리스트
        """
        # 프로젝트와 연결된 문서 조회
        documents = db.query(File).join(
            File.projects
        ).filter(
            File.projects.any(id=project_id)
        ).limit(max_docs).all()

        document_texts = []
        for doc in documents:
            # DocumentContent에서 텍스트 추출
            if doc.contents:
                # 각 페이지의 텍스트를 합침
                full_text = "\n".join([
                    content.content_text
                    for content in doc.contents
                    if content.content_text
                ])
                if full_text:
                    # 길이 제한 (각 문서당 2000자)
                    document_texts.append(full_text[:2000])

        return document_texts

    async def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        user_id: UUID,
        project_id: Optional[UUID],
        db: Session
    ) -> Dict[str, Any]:
        """
        텍스트 번역 (프로젝트 선택 여부에 따라 Agent 조합 변경)

        Args:
            text: 번역할 텍스트
            source_lang: 원본 언어
            target_lang: 목표 언어
            user_id: 사용자 ID
            project_id: 프로젝트 ID (None이면 기본 번역)
            db: DB 세션

        Returns:
            번역 결과 딕셔너리
        """
        logger.info(f"🌐 번역 요청: user={user_id}, project={project_id}, {source_lang}→{target_lang}")

        # Case 1: 프로젝트 없음 → SimpleTranslationAgent만 사용
        if project_id is None:
            logger.info("📝 기본 번역 모드 (프로젝트 없음)")

            translated_text = await self.simple_translator.process(
                text, source_lang, target_lang
            )

            # DB 저장
            translation = Translation(
                user_id=user_id,
                project_id=None,
                original_text=text,
                translated_text=translated_text,
                source_language=source_lang,
                target_language=target_lang,
                context_used=False,
                context_summary=None,
                terms_detected=0
            )
            db.add(translation)
            db.commit()
            db.refresh(translation)

            return {
                "translation_id": str(translation.id),
                "original_text": text,
                "translated_text": translated_text,
                "source_language": source_lang,
                "target_language": target_lang,
                "context_used": False,
                "context_summary": None,
                "detected_terms": [],
                "terms_count": 0
            }

        # Case 2: 프로젝트 있음 → Multiple Agents 조합
        logger.info(f"🚀 컨텍스트 기반 번역 모드 (프로젝트: {project_id})")

        # Step 1: 프로젝트의 용어집 조회
        glossary_terms = self._fetch_project_glossary(project_id, db)
        logger.info(f"📚 용어집 조회: {len(glossary_terms)}개")

        # Step 2: 용어 탐지 (source_lang에 따라 해당 언어 용어 매칭)
        detected_terms = await self.term_detector.process(
            text,
            [{"korean_term": t["korean_term"],
              "english_term": t.get("english_term"),
              "vietnamese_term": t.get("vietnamese_term")}
             for t in glossary_terms],
            source_lang=source_lang
        )
        logger.info(f"🔍 용어 탐지: {len(detected_terms)}개 (source_lang={source_lang})")

        # Step 3: 용어 매칭 (상세 정보)
        matched_terms = await self.glossary_matcher.process(detected_terms, glossary_terms)
        logger.info(f"✅ 용어 매칭: {len(matched_terms)}개")

        # Step 4: 문서 컨텍스트 생성
        project_documents = self._fetch_project_documents_text(project_id, db)
        context_summary = ""
        if project_documents:
            logger.info(f"📄 문서 조회: {len(project_documents)}개")
            context_summary = await self.document_summarizer.process(
                project_documents,
                max_length=500
            )
            logger.info(f"📝 컨텍스트 요약 완료: {len(context_summary)}자")
        else:
            logger.warning("⚠️ 프로젝트 문서가 없습니다")

        # Step 5: 컨텍스트 기반 번역
        translated_text = await self.context_translator.process(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            context=context_summary or "프로젝트 문서가 없습니다",
            glossary_terms=[{"korean_term": t["korean_term"],
                             "english_term": t.get("english_term"),
                             "vietnamese_term": t.get("vietnamese_term")}
                            for t in glossary_terms],
            detected_terms=detected_terms
        )

        # Step 6: DB 저장
        translation = Translation(
            user_id=user_id,
            project_id=project_id,
            original_text=text,
            translated_text=translated_text,
            source_language=source_lang,
            target_language=target_lang,
            context_used=True,
            context_summary=context_summary if context_summary else None,
            terms_detected=len(matched_terms)
        )
        db.add(translation)
        db.flush()  # ID 생성

        # 탐지된 용어 저장
        for matched_term in matched_terms:
            if matched_term.glossary_term_id:
                term_mapping = TranslationTerm(
                    translation_id=translation.id,
                    glossary_term_id=UUID(matched_term.glossary_term_id),
                    position_start=matched_term.position_start,
                    position_end=matched_term.position_end,
                    matched_text=matched_term.matched_text
                )
                db.add(term_mapping)

        db.commit()
        db.refresh(translation)

        logger.info(f"✅ 번역 완료 및 저장: translation_id={translation.id}")

        # 응답 구성
        return {
            "translation_id": str(translation.id),
            "original_text": text,
            "translated_text": translated_text,
            "source_language": source_lang,
            "target_language": target_lang,
            "context_used": True,
            "context_summary": context_summary,
            "detected_terms": [
                {
                    "matched_text": term.matched_text,
                    "position_start": term.position_start,
                    "position_end": term.position_end,
                    "glossary_term_id": term.glossary_term_id,
                    "korean_term": term.korean_term,
                    "english_term": term.english_term,
                    "vietnamese_term": term.vietnamese_term,
                    "definition": term.definition,
                    "domain": term.domain
                }
                for term in matched_terms
            ],
            "terms_count": len(matched_terms)
        }
