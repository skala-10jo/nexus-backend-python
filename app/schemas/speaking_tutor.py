"""
Pydantic schemas for Speaking Tutor API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class AnalysisStatus(str, Enum):
    """Analysis status enum."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ===== Request Schemas =====

class UploadRequest(BaseModel):
    """Request schema for audio file upload metadata."""
    language: str = Field(default="en-US", description="Language code (e.g., en-US, ko-KR)")

    class Config:
        populate_by_name = True


class FeedbackRequest(BaseModel):
    """Request schema for requesting feedback on an utterance."""
    utteranceId: str = Field(..., alias="utteranceId", description="UUID of the utterance")
    context: Optional[str] = Field(default=None, description="Optional context (e.g., 'business meeting')")

    class Config:
        populate_by_name = True


class UpdateSpeakerLabelRequest(BaseModel):
    """Request schema for updating a speaker label."""
    label: str = Field(..., min_length=1, max_length=50, description="New speaker label")

    class Config:
        populate_by_name = True


class BatchFeedbackRequest(BaseModel):
    """Request schema for batch feedback generation."""
    utteranceIds: List[str] = Field(..., alias="utteranceIds", description="List of utterance UUIDs")
    context: Optional[str] = Field(default=None, description="Optional context")

    class Config:
        populate_by_name = True


# ===== Response Schemas =====

class ScoreBreakdown(BaseModel):
    """Score breakdown by category."""
    grammar: int = Field(..., ge=0, le=10)
    vocabulary: int = Field(..., ge=0, le=10)
    fluency: int = Field(..., ge=0, le=10)
    clarity: int = Field(..., ge=0, le=10)


class FeedbackData(BaseModel):
    """Feedback data structure."""
    grammarCorrections: List[str] = Field(default=[], alias="grammarCorrections")
    suggestions: List[str] = Field(default=[])
    improvedSentence: str = Field(..., alias="improvedSentence")
    score: int = Field(..., ge=0, le=10)
    scoreBreakdown: Optional[ScoreBreakdown] = Field(default=None, alias="scoreBreakdown")

    class Config:
        populate_by_name = True


class UtteranceResponse(BaseModel):
    """Response schema for a single utterance."""
    id: str
    speakerId: int = Field(..., alias="speakerId")
    speakerLabel: Optional[str] = Field(default=None, alias="speakerLabel")
    text: str
    startTimeMs: int = Field(..., alias="startTimeMs")
    endTimeMs: int = Field(..., alias="endTimeMs")
    confidence: Optional[float] = None
    hasFeedback: bool = Field(default=False, alias="hasFeedback")
    feedback: Optional[FeedbackData] = None
    sequenceNumber: int = Field(..., alias="sequenceNumber")

    class Config:
        populate_by_name = True


class SpeakerInfo(BaseModel):
    """Speaker information with statistics."""
    id: int
    label: str
    utteranceCount: int = Field(..., alias="utteranceCount")

    class Config:
        populate_by_name = True


class UploadResponse(BaseModel):
    """Response schema for upload request."""
    sessionId: str = Field(..., alias="sessionId")
    status: str
    message: str

    class Config:
        populate_by_name = True


class AnalysisProgressResponse(BaseModel):
    """Response schema for analysis in progress."""
    sessionId: str = Field(..., alias="sessionId")
    status: str
    progress: int
    message: str

    class Config:
        populate_by_name = True


class AnalysisCompleteResponse(BaseModel):
    """Response schema for completed analysis."""
    sessionId: str = Field(..., alias="sessionId")
    status: str
    durationSeconds: Optional[float] = Field(default=None, alias="durationSeconds")
    speakerCount: int = Field(..., alias="speakerCount")
    utteranceCount: int = Field(..., alias="utteranceCount")
    speakers: List[SpeakerInfo]
    utterances: List[UtteranceResponse]

    class Config:
        populate_by_name = True


class FeedbackResponse(BaseModel):
    """Response schema for feedback generation."""
    utteranceId: str = Field(..., alias="utteranceId")
    feedback: FeedbackData

    class Config:
        populate_by_name = True


class UpdateSpeakerLabelResponse(BaseModel):
    """Response schema for speaker label update."""
    speakerId: int = Field(..., alias="speakerId")
    label: str
    updated: bool

    class Config:
        populate_by_name = True


class LearningItem(BaseModel):
    """Learning mode item."""
    utteranceId: str = Field(..., alias="utteranceId")
    originalText: str = Field(..., alias="originalText")
    improvedText: str = Field(..., alias="improvedText")
    grammarPoints: List[str] = Field(default=[], alias="grammarPoints")
    practiceCount: int = Field(default=0, alias="practiceCount")

    class Config:
        populate_by_name = True


class LearningModeResponse(BaseModel):
    """Response schema for learning mode data."""
    sessionId: str = Field(..., alias="sessionId")
    learningItems: List[LearningItem] = Field(..., alias="learningItems")

    class Config:
        populate_by_name = True


class SessionListItem(BaseModel):
    """Session list item for history."""
    id: str
    originalFilename: str = Field(..., alias="originalFilename")
    status: str
    speakerCount: int = Field(..., alias="speakerCount")
    utteranceCount: int = Field(..., alias="utteranceCount")
    durationSeconds: Optional[float] = Field(default=None, alias="durationSeconds")
    createdAt: datetime = Field(..., alias="createdAt")

    class Config:
        populate_by_name = True


class SessionListResponse(BaseModel):
    """Response schema for session list."""
    sessions: List[SessionListItem]
    total: int

    class Config:
        populate_by_name = True


class DeleteSessionResponse(BaseModel):
    """Response schema for session deletion."""
    sessionId: str = Field(..., alias="sessionId")
    deleted: bool
    message: str

    class Config:
        populate_by_name = True
