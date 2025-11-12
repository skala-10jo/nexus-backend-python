"""
Document 관련 SQLAlchemy ORM 모델

Java의 Document, DocumentContent, DocumentMetadata 엔티티와 매핑됩니다.
"""
from sqlalchemy import Column, String, Text, Integer, BigInteger, Boolean, ForeignKey, DateTime, Enum, Table
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import uuid
import enum


class DocumentStatus(enum.Enum):
    """문서 상태 열거형 (Java DocumentStatus와 동일)"""
    UPLOADED = "UPLOADED"       # 업로드 완료
    PROCESSING = "PROCESSING"   # 분석 중
    READY = "READY"             # 사용 가능
    ERROR = "ERROR"             # 오류 발생


# Project-Document 다대다 관계 테이블
project_documents = Table(
    'project_documents',
    Base.metadata,
    Column('project_id', UUID(as_uuid=True), ForeignKey('projects.id', ondelete='CASCADE'), primary_key=True),
    Column('document_id', UUID(as_uuid=True), ForeignKey('documents.id', ondelete='CASCADE'), primary_key=True),
    Column('created_at', DateTime(timezone=True), server_default=func.now(), nullable=False)
)


class Document(Base):
    """
    문서 모델

    사용자가 업로드한 문서 정보를 저장합니다.
    """
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # 파일 정보
    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False, unique=True)
    file_path = Column(String(500), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    file_type = Column(String(50), nullable=False)
    mime_type = Column(String(100), nullable=False)
    upload_date = Column(DateTime(timezone=True), nullable=False)

    # 상태
    status = Column(Enum(DocumentStatus), nullable=False, default=DocumentStatus.READY)
    is_analyzed = Column(Boolean, nullable=False, default=False)

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # 관계
    projects = relationship("Project", secondary=project_documents, back_populates="documents")
    contents = relationship("DocumentContent", back_populates="document", cascade="all, delete-orphan")
    doc_metadata = relationship("DocumentMetadata", back_populates="document", uselist=False, cascade="all, delete-orphan")


class DocumentContent(Base):
    """
    문서 콘텐츠 모델

    문서의 각 페이지별 텍스트 내용을 저장합니다.
    """
    __tablename__ = "document_content"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)

    # 페이지 정보
    page_number = Column(Integer, nullable=True)
    content_text = Column(Text, nullable=False)

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # 관계
    document = relationship("Document", back_populates="contents")


class DocumentMetadata(Base):
    """
    문서 메타데이터 모델

    문서의 메타데이터 정보를 저장합니다.
    """
    __tablename__ = "document_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, unique=True)

    # 메타데이터
    language = Column(String(10), nullable=True)
    page_count = Column(Integer, nullable=True)
    word_count = Column(Integer, nullable=True)
    character_count = Column(Integer, nullable=True)
    encoding = Column(String(20), nullable=True)

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # 관계
    document = relationship("Document", back_populates="doc_metadata")
