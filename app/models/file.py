"""
File 관련 SQLAlchemy ORM 모델

파일 공통 메타데이터를 저장합니다.
Java Backend의 V14 마이그레이션 스키마와 호환되도록 설계되었습니다.
"""
from sqlalchemy import Column, String, BigInteger, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import uuid
import enum


class FileType(str, enum.Enum):
    """파일 타입 Enum"""
    DOCUMENT = "DOCUMENT"
    VIDEO = "VIDEO"
    AUDIO = "AUDIO"


class File(Base):
    """
    파일 모델 (모든 파일 타입의 공통 메타데이터)

    Java Backend V14 스키마와 호환:
    - documents, videos, audio 파일의 공통 메타데이터 저장
    - file_type으로 구분
    """
    __tablename__ = "files"
    __table_args__ = {'extend_existing': True}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # 파일 타입 (DOCUMENT, VIDEO, AUDIO)
    file_type = Column(String(20), nullable=False)

    # 파일 정보
    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False, unique=True)
    file_path = Column(String(500), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    mime_type = Column(String(100), nullable=False)

    # 업로드 및 처리 상태
    upload_date = Column(DateTime(timezone=True), nullable=False)
    status = Column(String(20), nullable=False)

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # 관계
    # user = relationship("User", back_populates="files")  # User 모델에서 정의되어야 함
    video_file = relationship("VideoFile", back_populates="file", uselist=False, cascade="all, delete-orphan")

    # Document 관련 relationship (DocumentContent, DocumentMetadata)
    contents = relationship("DocumentContent", back_populates="file", cascade="all, delete-orphan")
    doc_metadata = relationship("DocumentMetadata", back_populates="file", uselist=False, cascade="all, delete-orphan")
