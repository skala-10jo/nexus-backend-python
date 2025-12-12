"""
용어집 추출 서비스 모듈.
문서에서 전문 용어를 추출하는 비즈니스 로직을 처리합니다.
"""
import os
import logging
from datetime import datetime
from typing import Dict, Any
from sqlalchemy.orm import Session

from agent.glossary.glossary_agent import GlossaryAgent
from app.models.glossary import GlossaryTerm, GlossaryTermDocument, GlossaryExtractionJob
from app.core.file_utils import extract_text_from_file
from app.config import settings

logger = logging.getLogger(__name__)


class GlossaryService:
    """
    용어집 추출 비즈니스 로직 서비스.

    담당 역할:
    - AI 처리를 위한 Agent 조율
    - 데이터베이스 작업
    - 작업 상태 관리
    - 에러 처리

    사용 예시:
        >>> service = GlossaryService()
        >>> await service.extract_and_save_terms(
        ...     job_id="job-123",
        ...     file_id="file-456",
        ...     file_path="uploads/document.pdf",
        ...     user_id="user-789",
        ...     project_id="proj-101",
        ...     db=db_session
        ... )
    """

    def __init__(self):
        """GlossaryAgent로 서비스를 초기화합니다."""
        self.agent = GlossaryAgent()

    async def extract_and_save_terms(
        self,
        job_id: str,
        file_id: str,
        file_path: str,
        user_id: str,
        project_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """
        문서에서 용어를 추출하고 데이터베이스에 저장합니다.

        주요 비즈니스 로직 흐름:
        1. 작업 상태를 PROCESSING으로 업데이트
        2. 문서 파일에서 텍스트 추출
        3. Agent를 호출하여 용어 추출
        4. 용어를 데이터베이스에 저장
        5. 작업 상태를 COMPLETED로 업데이트

        매개변수:
            job_id: 추출 작업 ID
            file_id: 파일 ID
            file_path: 문서 파일의 상대 경로
            user_id: 사용자 ID
            project_id: 프로젝트 ID ("None" 또는 None일 수 있음)
            db: 데이터베이스 세션

        반환값:
            다음을 포함하는 딕셔너리:
                - status: "success" 또는 "failed"
                - terms_count: 추출된 용어 수 (성공 시)
                - error: 에러 메시지 (실패 시)

        예외:
            Exception: 추출 실패 시 (에러가 로깅되고 작업이 실패로 표시됨)

        사용 예시:
            >>> service = GlossaryService()
            >>> result = await service.extract_and_save_terms(...)
            >>> print(f"추출된 용어 수: {result['terms_count']}")
        """
        try:
            # 1. 작업 상태를 PROCESSING으로 업데이트
            job = self._update_job_status(db, job_id, "PROCESSING", 10)
            job.started_at = datetime.utcnow()
            db.commit()
            logger.info(f"📝 작업 {job_id}: 처리 시작")

            # 2. 문서에서 텍스트 추출
            full_path = os.path.join(settings.upload_dir, file_path)
            logger.info(f"📄 작업 {job_id}: {full_path}에서 텍스트 추출 중")

            text = extract_text_from_file(full_path)

            if not text or len(text.strip()) < 100:
                raise ValueError("문서 텍스트가 너무 짧거나 비어있습니다")

            job.progress = 30
            db.commit()
            logger.info(f"📝 작업 {job_id}: 텍스트 추출 완료 ({len(text)}자)")

            # 3. Agent를 호출하여 용어 추출 (순수 AI 로직)
            logger.info(f"🤖 작업 {job_id}: GlossaryAgent로 용어 추출 호출")
            terms_data = await self.agent.process(text, max_terms=50)

            job.progress = 70
            db.commit()
            logger.info(f"📝 작업 {job_id}: Agent가 {len(terms_data)}개 용어 반환")

            # 4. 용어를 데이터베이스에 저장
            saved_count = self._save_terms(
                db=db,
                terms_data=terms_data,
                file_id=file_id,
                user_id=user_id,
                project_id=project_id
            )

            # 5. 작업을 완료로 업데이트
            job.status = "COMPLETED"
            job.progress = 100
            job.terms_extracted = saved_count
            job.completed_at = datetime.utcnow()
            db.commit()
            logger.info(f"✅ 작업 {job_id}: 성공적으로 완료")

            return {
                "status": "success",
                "terms_count": saved_count
            }

        except Exception as e:
            logger.error(f"❌ 작업 {job_id}: 에러 발생 - {str(e)}")
            # 실패 표시 전에 롤백
            db.rollback()
            self._mark_job_failed(db, job_id, str(e))
            return {
                "status": "failed",
                "error": str(e)
            }

    def _update_job_status(
        self,
        db: Session,
        job_id: str,
        status: str,
        progress: int
    ) -> GlossaryExtractionJob:
        """
        작업 상태와 진행률을 업데이트합니다 (private 메서드).

        매개변수:
            db: 데이터베이스 세션
            job_id: 작업 ID
            status: 새로운 상태 (PENDING, PROCESSING, COMPLETED, FAILED)
            progress: 진행률 (0-100)

        반환값:
            업데이트된 GlossaryExtractionJob 인스턴스

        예외:
            ValueError: 작업을 찾을 수 없는 경우
        """
        job = db.query(GlossaryExtractionJob).filter(
            GlossaryExtractionJob.id == job_id
        ).first()

        if not job:
            raise ValueError(f"작업 {job_id}을(를) 찾을 수 없습니다")

        job.status = status
        job.progress = progress
        db.commit()

        return job

    def _save_terms(
        self,
        db: Session,
        terms_data: list,
        file_id: str,
        user_id: str,
        project_id: str
    ) -> int:
        """
        추출된 용어를 데이터베이스에 저장합니다 (private 메서드).

        매개변수:
            db: 데이터베이스 세션
            terms_data: Agent에서 반환된 용어 딕셔너리 리스트
            file_id: 파일 ID
            user_id: 사용자 ID
            project_id: 프로젝트 ID ("None" 또는 None일 수 있음)

        반환값:
            성공적으로 저장된 용어 수

        참고:
            project_id의 "None" 문자열과 None 값을 처리합니다.
            에러 격리를 위해 savepoint를 사용합니다 - 하나의 용어가 실패해도
            해당 용어만 롤백되고 전체 배치는 롤백되지 않습니다.
        """
        saved_count = 0
        link_count = 0

        # 문자열 "None"을 실제 None으로 변환
        actual_project_id = None if project_id in ["None", None] else project_id

        for term_data in terms_data:
            # 개별 롤백을 위해 각 용어에 대해 savepoint 생성
            savepoint = db.begin_nested()
            try:
                # 용어가 이미 존재하는지 확인 (중복 체크)
                existing_term = db.query(GlossaryTerm).filter(
                    GlossaryTerm.user_id == user_id,
                    GlossaryTerm.korean_term == term_data['korean']
                ).first()

                if existing_term:
                    # 기존 용어가 있어도 새 문서와의 연결은 추가해야 함
                    existing_link = db.query(GlossaryTermDocument).filter(
                        GlossaryTermDocument.term_id == existing_term.id,
                        GlossaryTermDocument.file_id == file_id
                    ).first()

                    if not existing_link:
                        # 용어-문서 연결 추가 (새 문서에서도 이 용어가 발견됨)
                        term_doc = GlossaryTermDocument(
                            term_id=existing_term.id,
                            file_id=file_id
                        )
                        db.add(term_doc)
                        savepoint.commit()  # savepoint 커밋
                        link_count += 1
                        logger.info(f"기존 용어에 문서 연결 추가: '{term_data['korean']}'")
                    else:
                        savepoint.commit()  # 저장할 것 없지만 savepoint 커밋
                        logger.debug(f"용어와 문서 연결이 이미 존재하여 건너뜀: '{term_data['korean']}'")
                    continue

                # GlossaryTerm 생성
                term = GlossaryTerm(
                    project_id=actual_project_id,
                    user_id=user_id,
                    korean_term=term_data['korean'],
                    english_term=term_data.get('english'),
                    vietnamese_term=term_data.get('vietnamese'),
                    japanese_term=term_data.get('japanese'),
                    chinese_term=term_data.get('chinese'),
                    abbreviation=term_data.get('abbreviation'),
                    definition=term_data['definition'],
                    context=term_data.get('context'),
                    example_sentence=term_data.get('example_sentence'),
                    note=term_data.get('note'),
                    domain=term_data['domain'],
                    confidence_score=term_data['confidence'],
                    status='AUTO_EXTRACTED'
                )
                db.add(term)
                db.flush()  # term ID 획득

                # 용어-파일 연결 생성
                term_doc = GlossaryTermDocument(
                    term_id=term.id,
                    file_id=file_id
                )
                db.add(term_doc)
                savepoint.commit()  # savepoint 커밋
                saved_count += 1

            except Exception as e:
                # 전체 트랜잭션이 아닌 이 savepoint만 롤백
                savepoint.rollback()
                logger.warning(f"용어 저장 실패 '{term_data.get('korean', 'unknown')}': {str(e)}")
                continue

        # 메인 트랜잭션 커밋
        db.commit()
        logger.info(f"💾 작업: 신규 용어 {saved_count}개, 문서 연결 {link_count}개 저장 완료 (추출된 {len(terms_data)}개 중)")

        return saved_count + link_count

    def _mark_job_failed(
        self,
        db: Session,
        job_id: str,
        error_message: str
    ):
        """
        작업을 실패로 표시하고 에러 메시지를 저장합니다 (private 메서드).

        매개변수:
            db: 데이터베이스 세션
            job_id: 작업 ID
            error_message: 저장할 에러 메시지
        """
        job = db.query(GlossaryExtractionJob).filter(
            GlossaryExtractionJob.id == job_id
        ).first()

        if job:
            job.status = "FAILED"
            job.error_message = error_message
            job.completed_at = datetime.utcnow()
            db.commit()
