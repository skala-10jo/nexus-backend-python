"""
API endpoints for document processing (text extraction and summarization).
Handles routing and validation only - business logic is in DocumentProcessService.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.document_process import (
    DocumentProcessRequest,
    DocumentProcessResponse,
    DocumentSummaryResponse
)
from app.services.document_process_service import DocumentProcessService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/documents/process", response_model=DocumentProcessResponse)
async def process_document(
    request: DocumentProcessRequest,
    db: Session = Depends(get_db)
):
    """
    Process a document: extract text and generate AI summary.

    This endpoint extracts text from the document file and generates
    an AI summary, storing both in the database.

    Args:
        request: Process request containing file_id and file_path
        db: Database session

    Returns:
        DocumentProcessResponse with processing results

    Example:
        >>> POST /api/ai/documents/process
        >>> {
        >>>   "file_id": "uuid",
        >>>   "file_path": "uploads/documents/file.pdf"
        >>> }
        >>> Response: {
        >>>   "file_id": "uuid",
        >>>   "status": "success",
        >>>   "extracted_text_length": 5000,
        >>>   "summary_length": 800,
        >>>   "processed_at": "2025-01-01T00:00:00Z"
        >>> }
    """
    logger.info(f"📄 Processing document: file_id={request.file_id}")

    service = DocumentProcessService()

    try:
        result = await service.process_document(
            file_id=str(request.file_id),
            file_path=request.file_path,
            db=db
        )

        return DocumentProcessResponse(
            file_id=request.file_id,
            status=result["status"],
            extracted_text_length=result["extracted_text_length"],
            summary_length=result["summary_length"],
            processed_at=result["processed_at"]
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@router.get("/documents/{file_id}/summary", response_model=DocumentSummaryResponse)
async def get_document_summary(
    file_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the stored summary for a document.

    Args:
        file_id: The file ID to get summary for
        db: Database session

    Returns:
        DocumentSummaryResponse with extracted text and summary

    Example:
        >>> GET /api/ai/documents/{file_id}/summary
        >>> Response: {
        >>>   "file_id": "uuid",
        >>>   "extracted_text": "...",
        >>>   "summary": "...",
        >>>   "processed_at": "2025-01-01T00:00:00Z",
        >>>   "is_processed": true
        >>> }
    """
    logger.info(f"📖 Getting summary for document: file_id={file_id}")

    service = DocumentProcessService()

    result = await service.get_document_summary(
        file_id=file_id,
        db=db
    )

    if not result:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentSummaryResponse(
        file_id=result["file_id"],
        extracted_text=result["extracted_text"],
        summary=result["summary"],
        processed_at=result["processed_at"],
        is_processed=result["is_processed"]
    )
