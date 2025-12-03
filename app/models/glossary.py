"""
SQLAlchemy ORM models for glossary feature.
"""
from sqlalchemy import Column, String, Text, Integer, Numeric, Boolean, ForeignKey, UniqueConstraint, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from app.database import Base
import uuid


class GlossaryTerm(Base):
    """Glossary term model."""
    __tablename__ = "glossary_terms"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Term information
    korean_term = Column(String(255), nullable=False)
    english_term = Column(String(255), nullable=True)
    vietnamese_term = Column(String(255), nullable=True)
    japanese_term = Column(String(255), nullable=True)
    chinese_term = Column(String(255), nullable=True)
    abbreviation = Column(String(100), nullable=True)
    definition = Column(Text, nullable=False)
    context = Column(Text, nullable=True)
    example_sentence = Column(Text, nullable=True)
    note = Column(Text, nullable=True)

    # Metadata
    domain = Column(String(100), nullable=True)
    confidence_score = Column(Numeric(3, 2), nullable=True)
    status = Column(String(20), nullable=False, default="AUTO_EXTRACTED")
    is_verified = Column(Boolean, default=False)
    usage_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint('project_id', 'korean_term', name='glossary_terms_project_korean_unique'),
    )


class GlossaryTermDocument(Base):
    """Junction table for many-to-many relationship between terms and files."""
    __tablename__ = "glossary_term_documents"

    term_id = Column(UUID(as_uuid=True), ForeignKey("glossary_terms.id", ondelete="CASCADE"), primary_key=True)
    file_id = Column(UUID(as_uuid=True), primary_key=True)  # No FK constraint - managed by Java backend
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class GlossaryExtractionJob(Base):
    """Glossary extraction job model."""
    __tablename__ = "glossary_extraction_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    file_id = Column(UUID(as_uuid=True), nullable=False)  # No FK constraint - managed by Java backend

    # Job status
    status = Column(String(20), nullable=False, default="PENDING")
    progress = Column(Integer, default=0)
    terms_extracted = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)

    # Timestamps
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint('file_id', name='glossary_extraction_jobs_file_unique'),
    )
