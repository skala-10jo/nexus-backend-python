"""
문서 처리 서비스 모듈.
문서 업로드 시 텍스트 추출 및 AI 요약을 처리합니다.
"""
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.file_utils import extract_text_from_file
from app.config import settings
from agent.summarization.document_summarizer_agent import DocumentSummarizerAgent

logger = logging.getLogger(__name__)


class DocumentProcessService:
    """
    문서 처리 비즈니스 로직 서비스.

    담당 역할:
    - 문서에서 텍스트 추출
    - AI를 통한 요약 생성
    - 데이터베이스에 결과 저장

    사용 예시:
        >>> service = DocumentProcessService()
        >>> result = await service.process_document(
        ...     file_id="file-123",
        ...     file_path="uploads/document.pdf",
        ...     db=db_session
        ... )
    """

    def __init__(self):
        """DocumentSummarizerAgent로 서비스를 초기화합니다."""
        self.agent = DocumentSummarizerAgent()

    async def process_document(
        self,
        file_id: str,
        file_path: str,
        db: Session
    ) -> Dict[str, Any]:
        """
        문서에서 텍스트를 추출하고 요약을 생성합니다.

        주요 비즈니스 로직 흐름:
        1. 문서 파일에서 텍스트 추출
        2. Agent를 호출하여 요약 생성
        3. 결과를 데이터베이스에 저장

        매개변수:
            file_id: 파일 ID
            file_path: 문서 파일의 상대 경로
            db: 데이터베이스 세션

        반환값:
            다음을 포함하는 딕셔너리:
                - status: "success" 또는 "failed"
                - file_id: 처리된 파일 ID
                - extracted_text_length: 추출된 텍스트 길이
                - summary_length: 요약 길이
                - processed_at: 처리 완료 시간

        예외:
            Exception: 처리 실패 시
        """
        try:
            logger.info(f"📄 문서 처리 시작: file_id={file_id}")

            # 1. 문서에서 텍스트 추출
            full_path = os.path.join(settings.upload_dir, file_path)
            logger.info(f"📂 텍스트 추출 중: {full_path}")

            extracted_text = extract_text_from_file(full_path)

            if not extracted_text or len(extracted_text.strip()) < 100:
                raise ValueError("문서 텍스트가 너무 짧거나 비어있습니다")

            logger.info(f"✅ 텍스트 추출 완료: {len(extracted_text)}자")

            # 2. Agent를 호출하여 요약 생성
            logger.info(f"🤖 요약 생성 중...")
            summary = await self.agent.process([extracted_text], max_length=1000)
            logger.info(f"✅ 요약 생성 완료: {len(summary)}자")

            # 3. 데이터베이스에 저장
            processed_at = datetime.utcnow()
            self._save_to_db(
                db=db,
                file_id=file_id,
                extracted_text=extracted_text,
                summary=summary,
                processed_at=processed_at
            )

            logger.info(f"💾 DB 저장 완료: file_id={file_id}")

            return {
                "status": "success",
                "file_id": file_id,
                "extracted_text_length": len(extracted_text),
                "summary_length": len(summary),
                "processed_at": processed_at
            }

        except Exception as e:
            logger.error(f"❌ 문서 처리 실패: file_id={file_id}, error={str(e)}")
            raise

    def _save_to_db(
        self,
        db: Session,
        file_id: str,
        extracted_text: str,
        summary: str,
        processed_at: datetime
    ) -> None:
        """
        추출된 텍스트와 요약을 document_files 테이블에 저장합니다.

        매개변수:
            db: 데이터베이스 세션
            file_id: 파일 ID
            extracted_text: 추출된 전체 텍스트
            summary: AI 생성 요약
            processed_at: 처리 완료 시간
        """
        query = text("""
            UPDATE document_files
            SET extracted_text = :extracted_text,
                summary = :summary,
                processed_at = :processed_at,
                updated_at = :processed_at
            WHERE id = :file_id
        """)

        db.execute(query, {
            "file_id": file_id,
            "extracted_text": extracted_text,
            "summary": summary,
            "processed_at": processed_at
        })
        db.commit()

    async def get_document_summary(
        self,
        file_id: str,
        db: Session
    ) -> Optional[Dict[str, Any]]:
        """
        저장된 문서 요약을 조회합니다.

        매개변수:
            file_id: 파일 ID
            db: 데이터베이스 세션

        반환값:
            다음을 포함하는 딕셔너리 또는 None:
                - file_id: 파일 ID
                - extracted_text: 추출된 텍스트
                - summary: 요약
                - processed_at: 처리 시간
                - is_processed: 처리 여부
        """
        query = text("""
            SELECT id, extracted_text, summary, processed_at
            FROM document_files
            WHERE id = :file_id
        """)

        result = db.execute(query, {"file_id": file_id}).fetchone()

        if not result:
            return None

        return {
            "file_id": str(result[0]),
            "extracted_text": result[1],
            "summary": result[2],
            "processed_at": result[3],
            "is_processed": result[3] is not None
        }
