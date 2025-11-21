"""
Conversation models for PostgreSQL database.
Stores conversation practice sessions and their messages.
"""
from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.models.base import Base


class ConversationSession(Base):
    """
    Conversation session model for storing actual conversation practice sessions.

    Links to a Scenario (template) and tracks the user's practice session.
    """
    __tablename__ = "conversation_sessions"

    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    scenario_id = Column(PGUUID(as_uuid=True), ForeignKey('scenarios.id', ondelete='CASCADE'), nullable=False, index=True)

    # Session metadata
    status = Column(String(20), nullable=False, default='active', index=True)  # active, completed, abandoned
    user_role = Column(String(100), nullable=True)
    ai_role = Column(String(100), nullable=True)

    # Session timeline
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)

    # Statistics
    total_messages = Column(Integer, nullable=False, default=0)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationship
    scenario = relationship("Scenario", foreign_keys=[scenario_id])
    messages = relationship("ConversationMessage", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ConversationSession(id={self.id}, scenario_id={self.scenario_id}, status='{self.status}')>"


class ConversationMessage(Base):
    """
    Conversation message model for storing individual messages in a session.

    Each message can have translation, detected terms, and feedback.
    """
    __tablename__ = "conversation_messages"

    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(PGUUID(as_uuid=True), ForeignKey('conversation_sessions.id', ondelete='CASCADE'), nullable=False, index=True)

    # Message content
    sender = Column(String(10), nullable=False)  # 'user' or 'ai'
    message_text = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=True)

    # AI analysis
    detected_terms = Column(JSONB, nullable=True)  # ["term1", "term2", ...]
    feedback = Column(Text, nullable=True)

    # Ordering
    sequence_number = Column(Integer, nullable=False, index=True)

    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    # Relationship
    session = relationship("ConversationSession", back_populates="messages")

    def __repr__(self):
        return f"<ConversationMessage(id={self.id}, session_id={self.session_id}, sender='{self.sender}', seq={self.sequence_number})>"
