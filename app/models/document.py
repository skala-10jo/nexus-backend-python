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


# DocumentStatus는 File 모델로 이동됨
# project_files 테이블은 File 모델에서 사용
# Document 클래스는 제거됨 - File 모델을 사용하세요


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
    file = relationship("File", back_populates="contents")


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
    file = relationship("File", back_populates="doc_metadata")
