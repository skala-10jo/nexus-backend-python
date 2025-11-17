"""
Email model for RAG-based email search.

Author: NEXUS Team
Date: 2025-01-12
Updated: 2025-01-17 (Qdrant 마이그레이션 - EmailEmbedding 제거)
"""
from sqlalchemy import Column, String, Integer, Text, ForeignKey, TIMESTAMP, Boolean, BigInteger
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from app.models.base import Base
import uuid
from datetime import datetime


class Email(Base):
    """
    Email 테이블 모델 (Java Backend와 공유)

    Java Backend의 EmailSyncService에서 Outlook 메일을 동기화하여 저장합니다.
    Python Backend는 이 테이블을 읽어서 임베딩을 생성합니다.
    """
    __tablename__ = "emails"

    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Keys
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)

    # Email 기본 정보
    subject = Column(String(500), nullable=True)
    from_name = Column(String(255), nullable=True)
    from_address = Column(String(255), nullable=True)
    to_recipients = Column(Text, nullable=True)  # 세미콜론으로 구분된 수신자 목록
    cc_recipients = Column(Text, nullable=True)
    body = Column(Text, nullable=True)  # 전체 본문 (임베딩 대상)
    body_preview = Column(String(500), nullable=True)

    # Metadata
    folder = Column(String(100), nullable=True)  # "Inbox", "SentItems", etc.
    has_attachments = Column(Boolean, default=False)

    # Outlook IDs
    message_id = Column(String(255), nullable=False)
    conversation_id = Column(String(255), nullable=True)

    # Additional fields from actual DB
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id', ondelete='SET NULL'), nullable=True)
    bcc_recipients = Column(Text, nullable=True)
    body_type = Column(String(20), nullable=True)
    is_read = Column(Boolean, default=False, nullable=False)

    # Timestamps
    received_date_time = Column(TIMESTAMP(timezone=False), nullable=True)
    sent_date_time = Column(TIMESTAMP(timezone=False), nullable=True)
    synced_at = Column(TIMESTAMP(timezone=False), default=datetime.utcnow, nullable=False)

    # Relationships
    project = relationship("Project", foreign_keys=[project_id], lazy="joined")

    def __repr__(self):
        return f"<Email(id={self.id}, subject='{self.subject[:30]}...', folder={self.folder})>"
