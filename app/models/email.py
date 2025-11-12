"""
Email and EmailEmbedding models for RAG-based email search.

Author: NEXUS Team
Date: 2025-01-12
"""
from sqlalchemy import Column, String, Integer, Text, ForeignKey, TIMESTAMP, Boolean, BigInteger
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
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

    def __repr__(self):
        return f"<Email(id={self.id}, subject='{self.subject[:30]}...', folder={self.folder})>"


class EmailEmbedding(Base):
    """
    Email 임베딩 테이블 (RAG 검색용)

    메일 본문을 청킹하여 각 청크를 OpenAI text-embedding-ada-002로 임베딩합니다.
    pgvector를 사용하여 코사인 유사도 기반 검색을 수행합니다.

    Attributes:
        id: Primary key
        email_id: emails 테이블 참조 (CASCADE DELETE)
        chunk_index: 메일 내에서 청크 순서 (0, 1, 2, ...)
        chunk_text: 청킹된 텍스트 (메타데이터 포함)
        embedding: OpenAI 임베딩 벡터 (1536차원)
        metadata: JSONB 메타데이터 {subject, from_name, to_recipients, date, folder}
        created_at: 생성 시각

    Example:
        >>> # 메일 본문 2500자 → 3개 청크 (1000자 단위, 200자 오버랩)
        >>> email = db.query(Email).filter(Email.id == email_id).first()
        >>> embeddings = db.query(EmailEmbedding).filter(
        ...     EmailEmbedding.email_id == email_id
        ... ).all()
        >>> len(embeddings)  # 3
    """
    __tablename__ = "email_embeddings"

    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Key
    email_id = Column(
        UUID(as_uuid=True),
        ForeignKey('emails.id', ondelete='CASCADE'),
        nullable=False
    )

    # Chunk 정보
    chunk_index = Column(Integer, nullable=False)  # 청크 순서 (0부터 시작)
    chunk_text = Column(Text, nullable=False)      # 메타데이터 포함된 청크 텍스트

    # Vector
    embedding = Column(Vector(1536), nullable=False)  # OpenAI text-embedding-ada-002

    # Metadata (SQL 필터링용)
    email_metadata = Column('metadata', JSONB, nullable=True)  # {subject, from_name, to_recipients, date, folder, has_attachments}

    # Timestamp
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<EmailEmbedding(email_id={self.email_id}, chunk_index={self.chunk_index})>"
