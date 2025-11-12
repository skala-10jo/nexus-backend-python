"""
User 관련 SQLAlchemy ORM 모델

Java의 User 엔티티와 매핑됩니다.
"""
from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from app.database import Base
import uuid


class User(Base):
    """
    사용자 모델

    시스템의 사용자 정보를 저장합니다.
    """
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # 기본 사용자 정보
    username = Column(String(30), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    avatar_url = Column(String(500), nullable=True)

    # Outlook 연동 정보
    outlook_email = Column(String(255), nullable=True)
    outlook_access_token = Column(String, nullable=True)
    outlook_refresh_token = Column(String, nullable=True)
    outlook_token_expires_at = Column(DateTime(timezone=True), nullable=True)
    outlook_delta_link = Column(String, nullable=True)  # 증분 동기화용

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
