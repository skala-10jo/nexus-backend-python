"""
Scenario model for PostgreSQL database.
Stores conversation scenarios generated from projects/schedules.
"""
from sqlalchemy import Column, String, Text, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB
from sqlalchemy.sql import func
import uuid

from app.models.base import Base


class Scenario(Base):
    """
    Scenario model for storing conversation practice scenarios.

    Scenarios can be auto-generated from projects/schedules or manually created.
    Uses JSONB for flexible storage of roles and terminology.
    """
    __tablename__ = "scenarios"

    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)

    # Basic information
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    scenario_text = Column(Text, nullable=False)

    # Metadata
    language = Column(String(10), nullable=False, default='en', index=True)  # en, zh, ja, ko
    difficulty = Column(String(20), nullable=False, default='intermediate', index=True)  # beginner, intermediate, advanced
    category = Column(String(50), nullable=False)

    # JSONB fields for flexible data
    # roles: {"user": "role description", "ai": "role description"}
    roles = Column(JSONB, nullable=False)
    required_terminology = Column(JSONB, nullable=False, default=[])  # ["term1", "term2", ...]

    # Source tracking
    project_ids = Column(JSONB, nullable=False, default=[])  # ["uuid1", "uuid2", ...]
    schedule_ids = Column(JSONB, nullable=False, default=[])  # ["uuid1", "uuid2", ...]
    document_ids = Column(JSONB, nullable=False, default=[])  # ["uuid1", "uuid2", ...]

    # Generation type
    auto_generated = Column(Boolean, nullable=False, default=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self):
        return f"<Scenario(id={self.id}, title='{self.title}', language='{self.language}', difficulty='{self.difficulty}')>"
