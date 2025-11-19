"""
VideoDocument 관련 SQLAlchemy ORM 모델

영상 파일 메타데이터를 저장합니다.
Java Backend의 V13 마이그레이션 스키마와 호환되도록 설계되었습니다.
"""
from sqlalchemy import Column, String, Text, Integer, Boolean, ForeignKey, DateTime, DECIMAL
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import uuid


class VideoDocument(Base):
    """
    영상 문서 모델

    Java Backend V13 스키마와 호환:
    - document_id: Document와 1:1 관계
    - 영상 메타데이터 및 처리 상태 저장
    """
    __tablename__ = "video_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, unique=True)

    # 영상 메타데이터
    duration_seconds = Column(Integer, nullable=True)
    video_codec = Column(String(50), nullable=True)
    audio_codec = Column(String(50), nullable=True)
    resolution = Column(String(20), nullable=True)
    frame_rate = Column(DECIMAL(5, 2), nullable=True)
    has_audio = Column(Boolean, default=True, nullable=False)

    # 처리 상태
    stt_status = Column(String(20), default="pending", nullable=False)
    # pending, processing, completed, failed
    translation_status = Column(String(20), default="pending", nullable=False)
    # pending, processing, completed, failed

    # 언어 설정
    source_language = Column(String(10), nullable=True)
    target_language = Column(String(10), nullable=True)

    # 결과 파일 경로
    original_subtitle_path = Column(String(500), nullable=True)
    translated_subtitle_path = Column(String(500), nullable=True)

    # 에러 정보
    error_message = Column(Text, nullable=True)

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # 관계
    document = relationship("Document", back_populates="video_document")
    subtitles = relationship("VideoSubtitle", back_populates="video_document", cascade="all, delete-orphan")
