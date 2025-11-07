"""
API endpoints for glossary term extraction.
"""
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.glossary import ExtractionRequest, ExtractionResponse, HealthResponse
from app.models.glossary import GlossaryExtractionJob, GlossaryTerm, GlossaryTermDocument
from app.services.text_extractor import extract_text_from_file
from app.services.gpt_service import extract_terms_with_gpt
from app.config import settings
from datetime import datetime
import logging
import os

router = APIRouter()
logger = logging.getLogger(__name__)


async def process_extraction(
    job_id: str,
    document_id: str,
    file_path: str,
    user_id: str,
    project_id: str,
    db: Session
):
    """
    Background task to process glossary term extraction.

    Args:
        job_id: Extraction job ID
        document_id: Document ID
        file_path: Path to document file
        user_id: User ID
        project_id: Project ID
        db: Database session
    """
    try:
        # Update job status to PROCESSING
        job = db.query(GlossaryExtractionJob).filter(GlossaryExtractionJob.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        job.status = "PROCESSING"
        job.started_at = datetime.utcnow()
        job.progress = 10
        db.commit()
        logger.info(f"📝 Job {job_id}: Started processing")

        # Build full file path
        full_path = os.path.join(settings.UPLOAD_BASE_DIR, file_path)
        logger.info(f"📄 Job {job_id}: Extracting text from {full_path}")

        # Extract text from document
        text = extract_text_from_file(full_path)

        if not text or len(text.strip()) < 100:
            raise ValueError("Document text is too short or empty")

        job.progress = 30
        db.commit()
        logger.info(f"📝 Job {job_id}: Text extracted ({len(text)} characters)")

        # Extract terms using GPT-4o
        logger.info(f"🤖 Job {job_id}: Calling GPT-4o for term extraction")
        terms_data = await extract_terms_with_gpt(text, max_terms=50)

        job.progress = 70
        db.commit()
        logger.info(f"📝 Job {job_id}: GPT-4o returned {len(terms_data)} terms")

        # Save terms to database
        saved_count = 0
        for term_data in terms_data:
            try:
                # Convert string "None" to actual None for project_id
                actual_project_id = None if project_id == "None" or project_id is None else project_id

                # Create term
                term = GlossaryTerm(
                    project_id=actual_project_id,
                    user_id=user_id,
                    korean_term=term_data['korean'],
                    english_term=term_data.get('english'),
                    abbreviation=term_data.get('abbreviation'),
                    definition=term_data['definition'],
                    context=term_data.get('context'),
                    domain=term_data['domain'],
                    confidence_score=term_data['confidence'],
                    status='AUTO_EXTRACTED'
                )
                db.add(term)
                db.flush()  # Get term ID

                # Create term-document link
                term_doc = GlossaryTermDocument(
                    term_id=term.id,
                    document_id=document_id
                )
                db.add(term_doc)
                saved_count += 1

            except Exception as e:
                logger.warning(f"Failed to save term '{term_data.get('korean', 'unknown')}': {str(e)}")
                continue

        db.commit()
        logger.info(f"💾 Job {job_id}: Saved {saved_count}/{len(terms_data)} terms")

        # Update job as completed
        job.status = "COMPLETED"
        job.progress = 100
        job.terms_extracted = saved_count
        job.completed_at = datetime.utcnow()
        db.commit()
        logger.info(f"✅ Job {job_id}: Completed successfully")

    except Exception as e:
        logger.error(f"❌ Job {job_id}: Failed with error: {str(e)}")
        # Update job as failed
        job = db.query(GlossaryExtractionJob).filter(GlossaryExtractionJob.id == job_id).first()
        if job:
            job.status = "FAILED"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            db.commit()


@router.post("/glossary/extract", response_model=ExtractionResponse)
async def extract_glossary_terms(
    request: ExtractionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Start glossary term extraction from a document.

    This endpoint starts the extraction process as a background task
    and immediately returns with the job ID. The client should poll
    the job status endpoint to check progress.

    Args:
        request: Extraction request containing job_id, document_id, file_path, user_id, project_id
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        ExtractionResponse with status and job_id
    """
    logger.info(f"🚀 Starting extraction for document {request.document_id}")

    # Add background task
    background_tasks.add_task(
        process_extraction,
        str(request.job_id),
        str(request.document_id),
        request.file_path,
        str(request.user_id),
        str(request.project_id),
        db
    )

    return ExtractionResponse(
        status="started",
        job_id=request.job_id
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Python AI backend."""
    return HealthResponse(
        status="healthy",
        message="Python AI backend is running"
    )
