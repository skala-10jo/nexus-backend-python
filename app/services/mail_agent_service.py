"""
MailAgentService: 메일 임베딩 및 검색 오케스트레이션.

Agent를 조율하는 비즈니스 로직을 담당합니다.

Author: NEXUS Team
Date: 2025-01-12
Updated: 2025-01-17 (Qdrant 연동)
"""
from agent.mail.embedding_agent import EmbeddingAgent
from agent.mail.search_agent import SearchAgent
from app.models.email import Email
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MailAgentService:
    """
    메일 임베딩 및 검색 오케스트레이션.

    계층 구조:
        - API: 라우팅만
        - Service: Agent 조율
        - Agent: AI 로직 (임베딩 생성 + Qdrant 저장)

    Example:
        >>> service = MailAgentService()
        >>> result = await service.generate_embeddings_for_email('email_id', db)
        >>> result['status']  # 'success'
    """

    def __init__(self):
        """
        Agent 인스턴스화.

        BaseAgent 패턴에 따라 각 Agent는 싱글톤 OpenAI 클라이언트를 공유합니다.
        """
        self.embedding_agent = EmbeddingAgent()
        self.search_agent = SearchAgent()

    async def generate_embeddings_for_email(
        self,
        email_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """
        단일 메일에 대한 임베딩 생성 및 Qdrant 저장.

        Args:
            email_id: 이메일 ID
            db: DB 세션

        Returns:
            {
                'status': 'success' | 'skipped' | 'failed',
                'chunks_created': int (성공 시),
                'email_id': str,
                'reason': str (스킵 시),
                'error': str (실패 시)
            }

        Example:
            >>> result = await service.generate_embeddings_for_email('uuid', db)
            >>> result
            {'status': 'success', 'chunks_created': 3, 'email_id': 'uuid'}
        """
        try:
            # 1. 메일 데이터 조회
            email = db.query(Email).filter(Email.id == email_id).first()
            if not email:
                logger.error(f"Email {email_id}: Not found")
                return {
                    'status': 'failed',
                    'error': f"Email {email_id} not found"
                }

            # 2. 중복 체크 (Qdrant에 이미 있는지 확인)
            from app.core.qdrant_client import get_qdrant_client
            from app.config import settings
            from qdrant_client.http import models

            qdrant_client = get_qdrant_client()

            # email_id로 검색
            existing = qdrant_client.scroll(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="email_id",
                            match=models.MatchValue(value=str(email_id))
                        )
                    ]
                ),
                limit=1
            )

            if existing[0]:  # 이미 임베딩이 있음
                logger.info(f"Email {email_id}: Already has embeddings, skipping")
                return {
                    'status': 'skipped',
                    'reason': 'Already has embeddings',
                    'email_id': str(email_id)
                }

            # 2. Agent로 임베딩 생성 + Qdrant 저장
            email_data = {
                'email_id': email.id,
                'user_id': email.user_id,  # user_id 추가
                'subject': email.subject,
                'body': email.body,
                'folder': email.folder,
                'from_name': email.from_name,
                'to_recipients': email.to_recipients,
                'date': email.received_date_time or email.sent_date_time,
                'has_attachments': email.has_attachments
            }

            result = await self.embedding_agent.process(email_data)

            logger.info(
                f"Email {email_id}: Successfully created {result['chunks_created']} embeddings"
            )

            return result

        except ValueError as e:
            # 본문이 너무 짧거나 비어있음
            logger.warning(f"Email {email_id}: {str(e)}")
            return {
                'status': 'skipped',
                'reason': str(e),
                'email_id': str(email_id)
            }

        except Exception as e:
            logger.error(f"Email {email_id}: Failed to generate embeddings: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'email_id': str(email_id)
            }

    async def batch_generate_embeddings(
        self,
        user_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """
        사용자의 모든 메일에 대해 임베딩 일괄 생성.

        Args:
            user_id: 사용자 ID
            db: DB 세션

        Returns:
            {
                'status': 'success',
                'total': int,
                'processed': int,
                'skipped': int,
                'failed': int
            }

        Example:
            >>> result = await service.batch_generate_embeddings('user_id', db)
            >>> result
            {'status': 'success', 'total': 100, 'processed': 95, 'skipped': 3, 'failed': 2}
        """
        # 사용자의 모든 메일 조회
        emails = db.query(Email).filter(Email.user_id == user_id).all()

        logger.info(f"User {user_id}: Found {len(emails)} emails for batch processing")

        processed = 0
        skipped = 0
        failed = 0

        for email in emails:
            result = await self.generate_embeddings_for_email(str(email.id), db)

            if result['status'] == 'success':
                processed += 1
            elif result['status'] == 'skipped':
                skipped += 1
            else:
                failed += 1

        logger.info(
            f"User {user_id}: Batch complete - "
            f"processed={processed}, skipped={skipped}, failed={failed}"
        )

        return {
            'status': 'success',
            'total': len(emails),
            'processed': processed,
            'skipped': skipped,
            'failed': failed
        }

    async def search_emails(
        self,
        query: str,
        user_id: str,
        db: Session,
        top_k: int = 10,
        folder: str = None,
        date_from: str = None,
        date_to: str = None
    ) -> List[Dict[str, Any]]:
        """
        자연어 쿼리로 메일 검색.

        Args:
            query: 검색 쿼리
            user_id: 사용자 ID
            db: DB 세션
            top_k: 최대 결과 개수
            folder: 폴더 필터 (선택)
            date_from: 시작 날짜 (선택)
            date_to: 종료 날짜 (선택)

        Returns:
            검색 결과 리스트

        Example:
            >>> results = await service.search_emails(
            ...     query="프로젝트 일정",
            ...     user_id="uuid",
            ...     db=db,
            ...     folder="Inbox"
            ... )
            >>> len(results)  # 최대 top_k개
        """
        return await self.search_agent.process(
            query=query,
            user_id=user_id,
            db=db,
            top_k=top_k,
            folder=folder,
            date_from=date_from,
            date_to=date_to
        )
