"""
Speaking Tutor models for PostgreSQL database.
Stores speaking analysis sessions and utterances with speaker diarization.
"""
from sqlalchemy import Column, String, Text, Float, Integer, BigInteger, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.database import Base


class SpeakingAnalysisSession(Base):
    """
    Speaking Analysis Session model for storing uploaded audio analysis.

    Each session represents one audio file uploaded by a user for analysis.
    Contains metadata about the file and analysis status.
    """
    __tablename__ = "speaking_analysis_sessions"

    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # File information
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    duration_seconds = Column(Float, nullable=True)

    # Analysis status: PENDING, PROCESSING, COMPLETED, FAILED
    status = Column(String(20), nullable=False, default="PENDING", index=True)
    progress = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)

    # Metadata
    speaker_count = Column(Integer, nullable=False, default=0)
    utterance_count = Column(Integer, nullable=False, default=0)
    language = Column(String(10), nullable=False, default="en-US")

    # Speaker labels: {"1": "나", "2": "상대방1", ...}
    speaker_labels = Column(JSONB, nullable=False, default={})

    # AI-generated summary of meeting content
    summary = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    utterances = relationship("SpeakingUtterance", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<SpeakingAnalysisSession(id={self.id}, filename='{self.original_filename}', status='{self.status}')>"


class SpeakingUtterance(Base):
    """
    Speaking Utterance model for storing speaker-separated utterances.

    Each utterance represents a single speech segment from one speaker,
    with timing information and optional feedback.
    """
    __tablename__ = "speaking_utterances"

    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(PGUUID(as_uuid=True), ForeignKey("speaking_analysis_sessions.id", ondelete="CASCADE"), nullable=False, index=True)

    # Utterance information
    speaker_id = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)

    # Timestamps (milliseconds)
    start_time_ms = Column(BigInteger, nullable=False)
    end_time_ms = Column(BigInteger, nullable=False)

    # Confidence score from Azure STT (0.0 - 1.0)
    confidence = Column(Float, nullable=True)

    # Feedback (JSONB for flexibility)
    # Structure: {
    #   "grammar_corrections": [...],
    #   "suggestions": [...],
    #   "improved_sentence": "...",
    #   "score": 7,
    #   "score_breakdown": {"grammar": 6, "vocabulary": 8, "fluency": 7, "clarity": 7}
    # }
    feedback = Column(JSONB, nullable=True)

    # Sequence number for ordering
    sequence_number = Column(Integer, nullable=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    session = relationship("SpeakingAnalysisSession", back_populates="utterances")

    def __repr__(self):
        preview = self.text[:30] + "..." if len(self.text) > 30 else self.text
        return f"<SpeakingUtterance(id={self.id}, speaker={self.speaker_id}, text='{preview}')>"
