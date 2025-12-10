"""
Video Translation Service

영상 자막 STT 및 번역 비즈니스 로직을 담당합니다.
Java Backend V22 스키마와 호환되도록 설계되었습니다 (files/video_files 시스템).
"""

import logging
import math
import os
from typing import List, Dict, Any, Optional
from uuid import UUID
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

# Agent imports
from agent.video.stt_agent import VideoSTTAgent
from agent.video.subtitle_generator_agent import SubtitleGeneratorAgent
from agent.translate.context_enhanced_translation_agent import ContextEnhancedTranslationAgent
from agent.term_detection.term_detector_agent import TermDetectorAgent

# Model imports
from app.models.file import File  # For context documents
from app.models.video_file import VideoFile
from app.models.video_subtitle import VideoSubtitle

# Config
from app.config import settings

logger = logging.getLogger(__name__)


class VideoTranslationService:
    """
    영상 번역 비즈니스 로직

    책임:
    - STT Agent를 통한 음성 인식
    - 번역 Agent를 통한 자막 번역
    - 자막 파일 생성 및 DB 저장
    """

    def __init__(self):
        """Agent 인스턴스화"""
        self.stt_agent = VideoSTTAgent()
        self.subtitle_generator = SubtitleGeneratorAgent()
        self.context_translator = ContextEnhancedTranslationAgent()
        self.term_detector = TermDetectorAgent()

    def _get_video_file_by_file_id(self, file_id: UUID, db: Session) -> VideoFile:
        """
        파일 ID로 영상 파일 조회

        Args:
            file_id: 파일 ID (File)
            db: DB 세션

        Returns:
            VideoFile 객체

        Raises:
            ValueError: 영상 파일을 찾을 수 없을 때
        """
        video_file = db.query(VideoFile).filter(
            VideoFile.id == file_id
        ).first()

        if not video_file:
            raise ValueError(f"영상 파일을 찾을 수 없습니다: {file_id}")

        return video_file

    def _get_file_by_id(self, file_id: UUID, db: Session) -> File:
        """
        파일 ID로 파일 조회

        Args:
            file_id: 파일 ID
            db: DB 세션

        Returns:
            File 객체

        Raises:
            ValueError: 파일을 찾을 수 없을 때
        """
        file = db.query(File).filter(File.id == file_id).first()

        if not file:
            raise ValueError(f"파일을 찾을 수 없습니다: {file_id}")

        return file

    def _fetch_project_glossary(
        self,
        project_id: UUID,
        db: Session
    ) -> List[Dict[str, Any]]:
        """
        프로젝트의 용어집 조회 (translation_service.py와 동일한 로직)

        Args:
            project_id: 프로젝트 ID
            db: DB 세션

        Returns:
            용어집 리스트 (딕셔너리 형태)

        Note:
            project_files 조인을 통해 프로젝트에 연결된 모든 문서의 용어를 조회합니다.
        """
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
                "id": str(row.id),
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

        logger.info(f"📚 프로젝트 용어집 조회: {len(glossary_terms)}개 용어")
        return glossary_terms

    def _fetch_project_documents_text(
        self,
        project_id: UUID,
        db: Session,
        max_docs: int = 5
    ) -> str:
        """
        프로젝트의 문서 텍스트 조회 (컨텍스트 생성용)

        Args:
            project_id: 프로젝트 ID
            db: DB 세션
            max_docs: 최대 문서 수 (기본값: 5개)

        Returns:
            통합된 문서 텍스트
        """
        # 프로젝트와 연결된 문서 조회
        documents = db.query(File).join(
            File.projects
        ).filter(
            File.projects.any(id=project_id)
        ).limit(max_docs).all()

        document_texts = []
        for doc in documents:
            if doc.contents:
                full_text = "\n".join([
                    content.content_text
                    for content in doc.contents
                    if content.content_text
                ])
                if full_text:
                    document_texts.append(full_text[:2000])  # 각 문서 2000자 제한

        combined_text = "\n\n".join(document_texts)
        logger.info(f"📄 프로젝트 문서 조회: {len(documents)}개, 총 {len(combined_text)}자")

        return combined_text

    def _fetch_context_documents_text(
        self,
        document_ids: List[UUID],
        db: Session
    ) -> str:
        """
        컨텍스트 문서 텍스트 조회

        Args:
            document_ids: 문서 ID 리스트
            db: DB 세션

        Returns:
            통합된 문서 텍스트
        """
        if not document_ids:
            return ""

        documents = db.query(File).filter(File.id.in_(document_ids)).all()

        document_texts = []
        for doc in documents:
            if doc.contents:
                full_text = "\n".join([
                    content.content_text
                    for content in doc.contents
                    if content.content_text
                ])
                if full_text:
                    document_texts.append(full_text[:2000])  # 각 문서 2000자 제한

        combined_text = "\n\n".join(document_texts)
        logger.info(f"📄 컨텍스트 문서 조회: {len(documents)}개, 총 {len(combined_text)}자")

        return combined_text

    async def process_stt(
        self,
        video_file_id: UUID,
        source_language: str,
        user_id: UUID,
        db: Session
    ) -> Dict[str, Any]:
        """
        영상 STT 처리 및 DB 저장

        Args:
            video_file_id: 영상 파일 ID (File ID, not VideoFile ID)
            source_language: 음성 언어 코드
            user_id: 사용자 ID
            db: DB 세션

        Returns:
            STT 결과 딕셔너리
        """
        logger.info(f"🎥 STT 처리 시작: file={video_file_id}, lang={source_language}")

        # Step 1: File 및 VideoFile 조회
        file = self._get_file_by_id(video_file_id, db)
        video_file = self._get_video_file_by_file_id(video_file_id, db)

        # 절대 경로로 변환 (환경에 따라 자동 감지)
        upload_base_dir = settings.upload_dir
        full_video_path = os.path.join(upload_base_dir, file.file_path)

        if not os.path.exists(full_video_path):
            raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {full_video_path}")

        # Step 2: STT 처리 (Agent 호출)
        segments = await self.stt_agent.process(
            video_file_path=full_video_path,
            source_language=source_language
        )

        logger.info(f"🎤 STT 완료: {len(segments)}개 세그먼트")

        # Step 3: 기존 자막 삭제 (재처리 허용)
        deleted_count = db.query(VideoSubtitle).filter(
            VideoSubtitle.video_file_id == video_file.id
        ).delete()

        if deleted_count > 0:
            logger.info(f"🗑️ 기존 자막 삭제: {deleted_count}개")

        # Step 4: DB 저장 (각 세그먼트를 VideoSubtitle row로 저장)
        for segment_data in segments:
            # Whisper avg_logprob은 음수값이므로 exp()로 0-1 확률로 변환
            raw_confidence = segment_data.get("confidence")
            if raw_confidence is not None and raw_confidence < 0:
                # avg_logprob → probability: exp(logprob)
                confidence = math.exp(raw_confidence)
                # 0-1 범위로 클리핑
                confidence = max(0.0, min(1.0, confidence))
            else:
                confidence = raw_confidence

            subtitle = VideoSubtitle(
                video_file_id=video_file.id,
                sequence_number=segment_data["sequence_number"],
                start_time_ms=segment_data["start_time_ms"],
                end_time_ms=segment_data["end_time_ms"],
                original_text=segment_data["text"],
                original_language=source_language,  # 원본 언어 저장
                translations={},  # 빈 번역 딕셔너리로 초기화
                translated_text=None,  # 레거시 필드 (하위 호환성)
                confidence_score=confidence
            )
            db.add(subtitle)

        db.commit()

        # Step 5: 저장된 자막 조회
        saved_subtitles = db.query(VideoSubtitle).filter(
            VideoSubtitle.video_file_id == video_file.id
        ).order_by(VideoSubtitle.sequence_number).all()

        logger.info(f"✅ STT 결과 저장 완료: {len(saved_subtitles)}개 자막")

        # 응답 구성
        return {
            "video_file_id": video_file_id,
            "language": source_language,
            "segments": [
                {
                    "sequence_number": sub.sequence_number,
                    "start_time_ms": sub.start_time_ms,
                    "end_time_ms": sub.end_time_ms,
                    "text": sub.original_text,
                    "confidence": float(sub.confidence_score) if sub.confidence_score else None
                }
                for sub in saved_subtitles
            ],
            "total_segments": len(saved_subtitles),
            "created_at": saved_subtitles[0].created_at if saved_subtitles else None
        }

    async def process_translation(
        self,
        video_file_id: UUID,
        project_id: Optional[UUID],  # Text.vue 방식: projectId로 용어집 자동 조회
        source_language: str,
        target_language: str,
        user_id: UUID,
        db: Session
    ) -> Dict[str, Any]:
        """
        영상 자막 번역 및 DB 저장 (Text.vue 방식과 동일하게 projectId로 용어집 조회)

        Args:
            video_file_id: 영상 파일 ID (File ID)
            project_id: 프로젝트 ID (용어집 컨텍스트 자동 조회) - None이면 기본 번역
            source_language: 원본 언어
            target_language: 목표 언어
            user_id: 사용자 ID
            db: DB 세션

        Returns:
            번역 결과 딕셔너리
        """
        logger.info(f"🌐 자막 번역 시작: {source_language} → {target_language}, project={project_id}")

        # Step 1: VideoFile 조회
        video_file = self._get_video_file_by_file_id(video_file_id, db)

        # Step 2: 원본 자막 조회
        original_subtitles = db.query(VideoSubtitle).filter(
            VideoSubtitle.video_file_id == video_file.id
        ).order_by(VideoSubtitle.sequence_number).all()

        if not original_subtitles:
            raise ValueError(
                f"원본 자막을 찾을 수 없습니다. STT를 먼저 실행하세요. "
                f"(file={video_file_id})"
            )

        logger.info(f"📝 원본 자막 조회: {len(original_subtitles)}개 세그먼트")

        # Step 3: 프로젝트 기반 용어집 및 컨텍스트 조회 (Text.vue 방식)
        glossary_terms = []
        context_text = ""
        context_used = False

        if project_id:
            # 프로젝트 용어집 조회
            glossary_terms = self._fetch_project_glossary(project_id, db)
            logger.info(f"📚 용어집 조회: {len(glossary_terms)}개 용어")

            # 프로젝트 문서 컨텍스트 조회
            context_text = self._fetch_project_documents_text(project_id, db)
            context_used = len(context_text) > 0 or len(glossary_terms) > 0

        # Step 4: 각 세그먼트 번역 (ContextEnhancedTranslationAgent 사용)
        # 용어집을 번역용 포맷으로 변환 (한 번만 수행)
        glossary_for_translation = [
            {
                "korean_term": t["korean_term"],
                "english_term": t.get("english_term"),
                "vietnamese_term": t.get("vietnamese_term")
            }
            for t in glossary_terms
        ]

        total_detected_count = 0
        for subtitle in original_subtitles:
            # 용어 탐지 (TermDetectorAgent 사용)
            detected_terms = []
            if glossary_terms:
                detected_terms = await self.term_detector.process(
                    text=subtitle.original_text,
                    glossary_terms=glossary_for_translation,
                    source_lang=source_language
                )
                total_detected_count += len(detected_terms)

                # 탐지된 용어를 DB에 저장 (JSONB 형식)
                if detected_terms:
                    subtitle.detected_terms = [
                        {
                            "matched_text": t.matched_text,
                            "korean_term": t.korean_term,
                            "english_term": t.english_term,
                            "vietnamese_term": t.vietnamese_term
                        }
                        for t in detected_terms
                    ]
                    flag_modified(subtitle, "detected_terms")

            # 번역 (컨텍스트 + 용어집 + 탐지된 용어 포함)
            translated_text = await self.context_translator.process(
                text=subtitle.original_text,
                source_lang=source_language,
                target_lang=target_language,
                context=context_text if context_used else "컨텍스트 없음",
                glossary_terms=glossary_for_translation,
                detected_terms=detected_terms
            )

            # DB 업데이트 (translations JSON에 target_language 추가)
            if subtitle.translations is None:
                subtitle.translations = {}

            subtitle.translations[target_language] = translated_text
            flag_modified(subtitle, "translations")  # JSONB 변경 추적

            # 레거시 필드도 업데이트 (하위 호환성)
            subtitle.translated_text = translated_text

        db.commit()

        logger.info(
            f"✅ 번역 완료: {len(original_subtitles)}개 세그먼트 "
            f"(용어집: {len(glossary_terms)}개, 탐지된 용어: {total_detected_count}개)"
        )

        # 응답 구성
        return {
            "video_file_id": video_file_id,
            "source_language": source_language,
            "target_language": target_language,
            "segments": [
                {
                    "sequence_number": sub.sequence_number,
                    "start_time_ms": sub.start_time_ms,
                    "end_time_ms": sub.end_time_ms,
                    "original_text": sub.original_text,
                    "translated_text": sub.translated_text,
                    "confidence": float(sub.confidence_score) if sub.confidence_score else None
                }
                for sub in original_subtitles
            ],
            "total_segments": len(original_subtitles),
            "context_used": context_used,
            "context_document_count": len(glossary_terms),  # 용어 수로 변경
            "created_at": original_subtitles[0].created_at if original_subtitles else None
        }

    async def get_subtitles(
        self,
        video_file_id: UUID,
        db: Session
    ) -> Dict[str, Any]:
        """
        다국어 자막 조회

        Args:
            video_file_id: 영상 파일 ID (File ID)
            db: DB 세션

        Returns:
            다국어 자막 정보 딕셔너리
        """
        logger.info(f"📖 다국어 자막 조회: video_file_id={video_file_id}")

        # Step 1: VideoFile 조회
        video_file = self._get_video_file_by_file_id(video_file_id, db)

        # Step 2: 자막 조회
        subtitles = db.query(VideoSubtitle).filter(
            VideoSubtitle.video_file_id == video_file.id
        ).order_by(VideoSubtitle.sequence_number).all()

        if not subtitles:
            raise ValueError(f"자막을 찾을 수 없습니다: {video_file_id}")

        # Step 3: 사용 가능한 언어 목록 추출
        original_language = subtitles[0].original_language if subtitles else "ko"
        available_languages = {original_language}  # 원본 언어는 항상 포함

        # 모든 자막의 translations에서 언어 추출
        for subtitle in subtitles:
            if subtitle.translations:
                available_languages.update(subtitle.translations.keys())

        logger.info(f"✅ 자막 조회 완료: {len(subtitles)}개 세그먼트, {len(available_languages)}개 언어")

        # 응답 구성
        return {
            "video_file_id": video_file_id,
            "original_language": original_language,
            "available_languages": sorted(list(available_languages)),
            "segments": [
                {
                    "sequence_number": sub.sequence_number,
                    "start_time_ms": sub.start_time_ms,
                    "end_time_ms": sub.end_time_ms,
                    "original_text": sub.original_text,
                    "translations": sub.translations or {},
                    "confidence": float(sub.confidence_score) if sub.confidence_score else None,
                    "detected_terms": sub.detected_terms or []
                }
                for sub in subtitles
            ],
            "total_segments": len(subtitles)
        }

    async def generate_subtitle_file(
        self,
        video_file_id: UUID,
        language_type: str,  # "original" or "translated"
        db: Session
    ) -> Dict[str, Any]:
        """
        SRT 자막 파일 생성

        Args:
            video_file_id: 영상 파일 ID (File ID)
            language_type: "original" (원본) or "translated" (번역)
            db: DB 세션

        Returns:
            파일 정보 딕셔너리
        """
        logger.info(f"📄 SRT 파일 생성 시작: video_file_id={video_file_id}, type={language_type}")

        # Step 1: VideoFile 조회
        video_file = self._get_video_file_by_file_id(video_file_id, db)

        # Step 2: 자막 조회
        subtitles = db.query(VideoSubtitle).filter(
            VideoSubtitle.video_file_id == video_file.id
        ).order_by(VideoSubtitle.sequence_number).all()

        if not subtitles:
            raise ValueError("자막이 비어있습니다")

        # Step 3: 출력 파일 경로 생성 (환경에 따라 자동 감지)
        subtitle_dir = Path(settings.upload_dir) / "subtitles"
        subtitle_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{video_file.id}_{language_type}.srt"
        output_path = subtitle_dir / filename

        # Step 4: SubtitleGeneratorAgent로 SRT 파일 생성
        segments_data = []
        for sub in subtitles:
            text = sub.translated_text if language_type == "translated" else sub.original_text
            if text:  # 번역 전이면 translated_text가 None일 수 있음
                segments_data.append({
                    "sequence_number": sub.sequence_number,
                    "start_time_ms": sub.start_time_ms,
                    "end_time_ms": sub.end_time_ms,
                    "text": text
                })

        file_path = await self.subtitle_generator.process(
            segments=segments_data,
            output_path=str(output_path),
            subtitle_type=language_type.upper()
        )

        logger.info(f"✅ SRT 파일 생성 완료: {file_path}")

        # 파일 크기 확인
        file_size = os.path.getsize(file_path)

        return {
            "video_file_id": video_file_id,
            "file_path": file_path,
            "language_type": language_type,
            "file_size_bytes": file_size
        }
