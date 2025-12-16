"""
Pydantic schemas for document processing API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID
from datetime import datetime


class DocumentProcessRequest(BaseModel):
    """Request schema for document processing."""
    file_id: UUID = Field(..., description="File ID to process")
    file_path: str = Field(..., description="Path to the document file")


class DocumentProcessResponse(BaseModel):
    """Response schema for document processing."""
    file_id: UUID
    status: str
    extracted_text_length: int = Field(..., description="Length of extracted text")
    summary_length: int = Field(..., description="Length of generated summary")
    processed_at: datetime


class DocumentSummaryResponse(BaseModel):
    """Response schema for getting document summary."""
    file_id: UUID
    extracted_text: Optional[str] = None
    summary: Optional[str] = None
    processed_at: Optional[datetime] = None
    is_processed: bool = Field(..., description="Whether document has been processed")
