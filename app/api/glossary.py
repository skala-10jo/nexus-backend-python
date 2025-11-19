"""
API endpoints for glossary term extraction.
Handles routing and validation only - business logic is in GlossaryService.
"""
from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.glossary import ExtractionRequest, ExtractionResponse, HealthResponse
from app.services.glossary_service import GlossaryService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


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
        request: Extraction request containing job_id, file_id, file_path, user_id, project_id
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        ExtractionResponse with status and job_id

    Example:
        >>> POST /api/ai/glossary/extract
        >>> {
        >>>   "job_id": "uuid",
        >>>   "file_id": "uuid",
        >>>   "file_path": "uploads/documents/file.pdf",
        >>>   "user_id": "uuid",
        >>>   "project_id": "uuid"
        >>> }
        >>> Response: {"status": "started", "job_id": "uuid"}
    """
    logger.info(f"🚀 Starting extraction for file {request.file_id}")

    # Create service instance
    service = GlossaryService()

    # Add background task (delegate all business logic to Service)
    background_tasks.add_task(
        service.extract_and_save_terms,
        job_id=str(request.job_id),
        file_id=str(request.file_id),
        file_path=request.file_path,
        user_id=str(request.user_id),
        project_id=str(request.project_id),
        db=db
    )

    return ExtractionResponse(
        status="started",
        job_id=request.job_id,
        message="Glossary extraction started in background"
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for Python AI backend.

    Returns:
        HealthResponse with status and message
    """
    return HealthResponse(
        status="healthy",
        message="Python AI backend is running"
    )
