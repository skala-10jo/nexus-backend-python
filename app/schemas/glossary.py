"""
Pydantic schemas for glossary API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID
from datetime import datetime
from decimal import Decimal


class ExtractionRequest(BaseModel):
    """용어 추출 요청 스키마."""
    job_id: UUID
    file_id: UUID
    file_path: str
    file_content: Optional[str] = None  # Base64 인코딩된 파일 내용
    file_name: Optional[str] = None  # 원본 파일명 (확장자 포함)
    user_id: UUID
    project_id: Optional[UUID] = None


class ExtractionResponse(BaseModel):
    """Response schema for term extraction start."""
    status: str
    job_id: UUID


class TermData(BaseModel):
    """Schema for term data extracted by GPT-4o."""
    korean: str = Field(..., description="Korean term")
    english: Optional[str] = Field(None, description="English term")
    vietnamese: Optional[str] = Field(None, description="Vietnamese term")
    japanese: Optional[str] = Field(None, description="Japanese term")
    chinese: Optional[str] = Field(None, description="Chinese term")
    abbreviation: Optional[str] = Field(None, description="Abbreviation if exists")
    definition: str = Field(..., description="Clear definition of the term")
    context: Optional[str] = Field(None, description="Context where term was used in document")
    example_sentence: Optional[str] = Field(None, description="Example sentence showing term usage")
    note: Optional[str] = Field(None, description="Additional notes and references")
    domain: str = Field(..., description="Domain (IT, Business, etc.)")
    confidence: Decimal = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0-1.0")


class GPTExtractionResult(BaseModel):
    """Schema for GPT-4o extraction result."""
    terms: List[TermData]


class JobStatusResponse(BaseModel):
    """Response schema for job status."""
    id: UUID
    status: str
    progress: int
    terms_extracted: int
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str
