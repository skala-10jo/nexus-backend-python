"""
번역 SQLAlchemy 모델

Translation 및 TranslationTerm 엔티티
"""

import uuid
from sqlalchemy import Column, String, Text, Boolean, Integer, ForeignKey, DateTime, UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class Translation(Base):
    """
    번역 정보

    사용자가 수행한 번역의 기록을 저장합니다.
    프로젝트와 연결되어 컨텍스트 기반 번역 여부를 추적합니다.
    """
    __tablename__ = "translations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True)

    # 번역 내용
    original_text = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=False)
    source_language = Column(String(10), nullable=False)  # ko, en, ja, vi
    target_language = Column(String(10), nullable=False)

    # 컨텍스트 정보
    context_used = Column(Boolean, default=False)
    context_summary = Column(Text, nullable=True)  # 사용된 컨텍스트 요약

    # 메타데이터
    terms_detected = Column(Integer, default=0)  # 탐지된 용어 수

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now(), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    translation_terms = relationship("TranslationTerm", back_populates="translation", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Translation(id={self.id}, {self.source_language}→{self.target_language})>"


class TranslationTerm(Base):
    """
    번역에서 탐지된 용어 매핑

    번역 시 탐지된 전문용어와 용어집의 매핑 정보를 저장합니다.
    원문에서의 위치 정보도 함께 저장하여 하이라이트에 활용합니다.
    """
    __tablename__ = "translation_terms"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    translation_id = Column(UUID(as_uuid=True), ForeignKey("translations.id", ondelete="CASCADE"), nullable=False)
    glossary_term_id = Column(UUID(as_uuid=True), ForeignKey("glossary_terms.id", ondelete="CASCADE"), nullable=False)

    # 탐지 정보
    position_start = Column(Integer, nullable=False)  # 원문에서의 시작 위치
    position_end = Column(Integer, nullable=False)    # 원문에서의 종료 위치
    matched_text = Column(String(255), nullable=False)  # 탐지된 실제 텍스트

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now(), default=func.now(), nullable=False)

    # Relationships
    translation = relationship("Translation", back_populates="translation_terms")

    def __repr__(self):
        return f"<TranslationTerm(matched_text={self.matched_text})>"
