"""
SQLAlchemy ORM models for expression feature.
"""
from sqlalchemy import Column, String, Boolean, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.database import Base
import uuid


class Expression(Base):
    """Expression model."""
    __tablename__ = "expressions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    expression = Column(String(500), nullable=False, unique=True)
    meaning = Column(String(500), nullable=False)
    examples = Column(JSONB, nullable=False)
    unit = Column(String(200), nullable=False)
    chapter = Column(String(200), nullable=False)
    source_section = Column(String(200), nullable=True)


class UserExpression(Base):
    """User expression learning status model."""
    __tablename__ = "user_expressions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    expression_id = Column(UUID(as_uuid=True), ForeignKey("expressions.id", ondelete="CASCADE"), nullable=False)
    is_learned = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint('user_id', 'expression_id', name='uk_user_expressions'),
    )
