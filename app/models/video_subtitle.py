"""
Video Subtitle 관련 SQLAlchemy ORM 모델

영상 자막 데이터를 저장합니다.
Java Backend의 V22 마이그레이션 스키마와 호환되도록 설계되었습니다.
"""
from sqlalchemy import Column, String, Text, Integer, BigInteger, ForeignKey, DateTime, DECIMAL
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import uuid


class VideoSubtitle(Base):
    """
    영상 자막 세그먼트 모델

    Java Backend V22 스키마와 호환:
    - video_file_id: VideoFile 참조 (video_files 테이블)
    - original_text, translated_text: 원본 및 번역 텍스트를 하나의 row에 저장
    - translations JSONB: 다국어 번역 지원
    """
    __tablename__ = "video_subtitles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_file_id = Column(UUID(as_uuid=True), ForeignKey("video_files.id", ondelete="CASCADE"), nullable=False)

    # 시퀀스 및 타이밍
    sequence_number = Column(Integer, nullable=False)
    start_time_ms = Column(BigInteger, nullable=False)  # 밀리초 단위
    end_time_ms = Column(BigInteger, nullable=False)

    # 텍스트
    original_text = Column(Text, nullable=False)
    original_language = Column(String(10), nullable=False, server_default='ko')  # 원본 언어 코드

    # 다국어 번역 (JSONB)
    # 예: {"en": "Hello", "ja": "こんにちは", "vi": "Xin chào"}
    translations = Column(JSONB, nullable=False, server_default='{}')

    # 레거시 필드 (하위 호환성 유지, deprecated)
    translated_text = Column(Text, nullable=True)

    # 화자 정보 (선택사항)
    speaker_id = Column(Integer, nullable=True)

    # 신뢰도 점수 (STT 결과)
    confidence_score = Column(DECIMAL(3, 2), nullable=True)

    # 탐지된 전문용어 (JSON 배열)
    detected_terms = Column(JSONB, nullable=True)

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # 관계
    video_file = relationship("VideoFile", back_populates="subtitles")
