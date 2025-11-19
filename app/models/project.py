"""
Project 관련 SQLAlchemy ORM 모델

Java의 Project 엔티티와 매핑됩니다.
"""
from sqlalchemy import Column, String, Text, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import uuid


class Project(Base):
    """
    프로젝트 모델

    사용자의 프로젝트 정보를 저장합니다.
    프로젝트는 문서와 용어집을 가질 수 있습니다.
    """
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # 프로젝트 정보
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="ACTIVE")  # ACTIVE, ARCHIVED, DELETED

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # 관계
    # documents는 project_files 테이블을 통한 다대다 관계
    # document.py에서 정의된 project_files 테이블 사용
    documents = relationship("Document", secondary="project_files", back_populates="projects")

    # glossary_terms는 일대다 관계 (GlossaryTerm의 project_id가 외래키)
    # Note: glossary.py에서 relationship이 이미 정의되어 있으므로 여기서는 back_populates만 설정
