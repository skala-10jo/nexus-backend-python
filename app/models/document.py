"""
Document 관련 SQLAlchemy ORM 모델

Java의 Document, DocumentContent, DocumentMetadata 엔티티와 매핑됩니다.
"""
from sqlalchemy import Column, String, Text, Integer, BigInteger, Boolean, ForeignKey, DateTime, Enum, Table
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, foreign
from sqlalchemy.sql import func
from app.database import Base
import uuid
import enum


class DocumentStatus(enum.Enum):
    """문서 상태 열거형 (Java DocumentStatus와 동일)"""
    UPLOADED = "UPLOADED"       # 업로드 완료
    PROCESSING = "PROCESSING"   # 분석 중
    PROCESSED = "PROCESSED"     # 분석 완료 (사용 가능)
    FAILED = "FAILED"           # 오류 발생


# Project-File 다대다 관계 테이블 (project_files 테이블 사용)
project_files = Table(
    'project_files',
    Base.metadata,
    Column('project_id', UUID(as_uuid=True), ForeignKey('projects.id', ondelete='CASCADE'), primary_key=True),
    Column('file_id', UUID(as_uuid=True), ForeignKey('files.id', ondelete='CASCADE'), primary_key=True),
    Column('created_at', DateTime(timezone=True), server_default=func.now(), nullable=False)
)


class Document(Base):
    """
    문서 모델

    사용자가 업로드한 문서 정보를 저장합니다.
    NOTE: files 테이블을 사용합니다 (documents → files 마이그레이션 완료)
    """
    __tablename__ = "files"

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

    # 상태 (files 테이블에서는 String 타입 사용)
    status = Column(String(20), nullable=False, default="PROCESSED")

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # 관계 (project_files 테이블 사용)
    projects = relationship("Project", secondary=project_files, back_populates="documents")
    contents = relationship(
        "DocumentContent",
        primaryjoin="Document.id == foreign(DocumentContent.file_id)",
        back_populates="document",
        cascade="all, delete-orphan"
    )
    doc_metadata = relationship(
        "DocumentMetadata",
        primaryjoin="Document.id == foreign(DocumentMetadata.file_id)",
        back_populates="document",
        uselist=False,
        cascade="all, delete-orphan"
    )
    # video_document relationship removed (migrated to files/video_files system)


class DocumentContent(Base):
    """
    문서 콘텐츠 모델

    문서의 각 페이지별 텍스트 내용을 저장합니다.
    NOTE: file_id 컬럼을 사용합니다 (document_id → file_id 마이그레이션 완료)
    """
    __tablename__ = "document_content"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_id = Column(UUID(as_uuid=True), ForeignKey("files.id", ondelete="CASCADE"), nullable=False, name="file_id")

    # 페이지 정보
    page_number = Column(Integer, nullable=True)
    content_text = Column(Text, nullable=False)

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # 관계
    document = relationship("Document", foreign_keys=[file_id], back_populates="contents")


class DocumentMetadata(Base):
    """
    문서 메타데이터 모델

    문서의 메타데이터 정보를 저장합니다.
    NOTE: file_id 컬럼을 사용합니다 (document_id → file_id 마이그레이션 완료)
    """
    __tablename__ = "document_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_id = Column(UUID(as_uuid=True), ForeignKey("files.id", ondelete="CASCADE"), nullable=False, unique=True, name="file_id")

    # 메타데이터
    language = Column(String(10), nullable=True)
    page_count = Column(Integer, nullable=True)
    word_count = Column(Integer, nullable=True)
    character_count = Column(Integer, nullable=True)
    encoding = Column(String(20), nullable=True)

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # 관계
    document = relationship("Document", foreign_keys=[file_id], back_populates="doc_metadata")
