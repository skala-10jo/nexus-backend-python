"""
Glossary extraction service.
Handles business logic for extracting glossary terms from documents.
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
    Service for glossary term extraction business logic.

    Orchestrates:
    - Agent for AI processing
    - Database operations
    - Job status management
    - Error handling

    Example:
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
        """Initialize service with GlossaryAgent."""
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
        Extract glossary terms from document and save to database.

        This is the main business logic flow:
        1. Update job status to PROCESSING
        2. Extract text from document file
        3. Call Agent to extract terms
        4. Save terms to database
        5. Update job status to COMPLETED

        Args:
            job_id: Extraction job ID
            file_id: File ID
            file_path: Relative path to document file
            user_id: User ID
            project_id: Project ID (can be "None" or None for no project)
            db: Database session

        Returns:
            Dict with:
                - status: "success" or "failed"
                - terms_count: Number of terms extracted (if success)
                - error: Error message (if failed)

        Raises:
            Exception: If extraction fails (error is logged and job marked as failed)

        Example:
            >>> service = GlossaryService()
            >>> result = await service.extract_and_save_terms(...)
            >>> print(f"Extracted {result['terms_count']} terms")
        """
        try:
            # 1. Update job status to PROCESSING
            job = self._update_job_status(db, job_id, "PROCESSING", 10)
            job.started_at = datetime.utcnow()
            db.commit()
            logger.info(f"📝 Job {job_id}: Started processing")

            # 2. Extract text from document
            full_path = os.path.join(settings.upload_dir, file_path)
            logger.info(f"📄 Job {job_id}: Extracting text from {full_path}")

            text = extract_text_from_file(full_path)

            if not text or len(text.strip()) < 100:
                raise ValueError("Document text is too short or empty")

            job.progress = 30
            db.commit()
            logger.info(f"📝 Job {job_id}: Text extracted ({len(text)} characters)")

            # 3. Call Agent to extract terms (pure AI logic)
            logger.info(f"🤖 Job {job_id}: Calling GlossaryAgent for term extraction")
            terms_data = await self.agent.process(text, max_terms=50)

            job.progress = 70
            db.commit()
            logger.info(f"📝 Job {job_id}: Agent returned {len(terms_data)} terms")

            # 4. Save terms to database
            saved_count = self._save_terms(
                db=db,
                terms_data=terms_data,
                file_id=file_id,
                user_id=user_id,
                project_id=project_id
            )

            # 5. Update job as completed
            job.status = "COMPLETED"
            job.progress = 100
            job.terms_extracted = saved_count
            job.completed_at = datetime.utcnow()
            db.commit()
            logger.info(f"✅ Job {job_id}: Completed successfully")

            return {
                "status": "success",
                "terms_count": saved_count
            }

        except Exception as e:
            logger.error(f"❌ Job {job_id}: Failed with error: {str(e)}")
            # Rollback before marking job as failed
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
        Update job status and progress (private method).

        Args:
            db: Database session
            job_id: Job ID
            status: New status (PENDING, PROCESSING, COMPLETED, FAILED)
            progress: Progress percentage (0-100)

        Returns:
            Updated GlossaryExtractionJob instance

        Raises:
            ValueError: If job not found
        """
        job = db.query(GlossaryExtractionJob).filter(
            GlossaryExtractionJob.id == job_id
        ).first()

        if not job:
            raise ValueError(f"Job {job_id} not found")

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
        Save extracted terms to database (private method).

        Args:
            db: Database session
            terms_data: List of term dictionaries from Agent
            file_id: File ID
            user_id: User ID
            project_id: Project ID (can be "None" or None)

        Returns:
            Number of terms successfully saved

        Note:
            Handles "None" string and None values for project_id.
            Uses savepoints for error isolation - if one term fails,
            only that term is rolled back, not the entire batch.
        """
        saved_count = 0
        link_count = 0

        # Convert string "None" to actual None
        actual_project_id = None if project_id in ["None", None] else project_id

        for term_data in terms_data:
            # Create savepoint for each term so we can rollback individually
            savepoint = db.begin_nested()
            try:
                # Check if term already exists (중복 체크)
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
                        savepoint.commit()  # Commit the savepoint
                        link_count += 1
                        logger.info(f"Term already exists, added document link: '{term_data['korean']}'")
                    else:
                        savepoint.commit()  # Nothing to save, but commit savepoint
                        logger.debug(f"Term and document link already exist, skipping: '{term_data['korean']}'")
                    continue

                # Create GlossaryTerm
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
                db.flush()  # Get term ID

                # Create term-file link
                term_doc = GlossaryTermDocument(
                    term_id=term.id,
                    file_id=file_id
                )
                db.add(term_doc)
                savepoint.commit()  # Commit the savepoint
                saved_count += 1

            except Exception as e:
                # Rollback only this savepoint, not the entire transaction
                savepoint.rollback()
                logger.warning(f"Failed to save term '{term_data.get('korean', 'unknown')}': {str(e)}")
                continue

        # Commit the main transaction
        db.commit()
        logger.info(f"💾 Job: Saved {saved_count} new terms, {link_count} document links (from {len(terms_data)} extracted)")

        return saved_count + link_count

    def _mark_job_failed(
        self,
        db: Session,
        job_id: str,
        error_message: str
    ):
        """
        Mark job as failed with error message (private method).

        Args:
            db: Database session
            job_id: Job ID
            error_message: Error message to store
        """
        job = db.query(GlossaryExtractionJob).filter(
            GlossaryExtractionJob.id == job_id
        ).first()

        if job:
            job.status = "FAILED"
            job.error_message = error_message
            job.completed_at = datetime.utcnow()
            db.commit()
