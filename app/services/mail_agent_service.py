"""
MailAgentService: 메일 임베딩 및 검색 오케스트레이션.


Updated: 2025-01-17 (Qdrant 연동)
"""
from agent.mail.embedding_agent import EmbeddingAgent
from agent.mail.search_agent import SearchAgent

from app.models.email import Email
from sqlalchemy.orm import Session

from typing import List, Dict, Any, Optional
import logging

from app.core.qdrant_client import get_qdrant_client
from app.config import settings
from qdrant_client.http import models


logger = logging.getLogger(__name__)


class MailAgentService:
    """
    메일 임베딩 및 검색 오케스트레이션.
    """

    def __init__(self):
        self.embedding_agent = EmbeddingAgent()
        self.search_agent = SearchAgent()

    async def generate_embeddings_for_email(
        self,
        email_id: str,
        db: Session,
        force_regenerate: bool = False
    ) -> Dict[str, Any]:

        try:
            # 1. 메일 조회
            email = db.query(Email).filter(Email.id == email_id).first()
            if not email:
                logger.error(f"Email {email_id}: Not found")
                return {
                    'status': 'failed',
                    'error': f"Email {email_id} not found"
                }

            # 2. Qdrant 중복 체크
            qdrant_client = get_qdrant_client()

            existing = qdrant_client.scroll(
                collection_name=settings.QDRANT_EMAIL_COLLECTION,
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

            # 이미 임베딩 있으면 스킵 or 재생성
            if existing[0]:
                if not force_regenerate:
                    return {
                        'status': 'skipped',
                        'reason': 'Already has embeddings',
                        'email_id': str(email_id)
                    }
                else:
                    qdrant_client.delete(
                        collection_name=settings.QDRANT_EMAIL_COLLECTION,
                        points_selector=models.FilterSelector(
                            filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="email_id",
                                        match=models.MatchValue(value=str(email_id))
                                    )
                                ]
                            )
                        )
                    )

            # 3. Agent로 임베딩 생성
            email_data = {
                'email_id': email.id,
                'user_id': email.user_id,
                'subject': email.subject,
                'body': email.body,
                'folder': email.folder,
                'from_name': email.from_name,
                'to_recipients': email.to_recipients,
                'date': email.received_date_time or email.sent_date_time,
                'has_attachments': email.has_attachments,
                'project_id': str(email.project_id) if email.project_id else None,
                'project_name': (
                    email.project.name
                    if email.project_id and hasattr(email, "project")
                    else None
                ),
            }

            result = await self.embedding_agent.process(email_data)
            return result

        except ValueError as e:
            # 본문이 짧아 스킵
            return {
                'status': 'skipped',
                'reason': str(e),
                'email_id': str(email_id)
            }

        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'email_id': str(email_id)
            }

    async def batch_generate_embeddings(
        self,
        user_id: str,
        db: Session,
        force_regenerate: bool = False
    ) -> Dict[str, Any]:

        emails = db.query(Email).filter(Email.user_id == user_id).all()

        processed = 0
        skipped = 0
        failed = 0

        for email in emails:
            result = await self.generate_embeddings_for_email(
                str(email.id),
                db,
                force_regenerate
            )

            if result['status'] == 'success':
                processed += 1
            elif result['status'] == 'skipped':
                skipped += 1
            else:
                failed += 1

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
        folder: Optional[str] = None,         
        date_from: Optional[str] = None,      
        date_to: Optional[str] = None,        
        project_name: Optional[str] = None    
    ) -> List[Dict[str, Any]]:

        return await self.search_agent.process(
            query=query,
            user_id=user_id,
            db=db,
            top_k=top_k,
            folder=folder,
            date_from=date_from,
            date_to=date_to,
            project_name=project_name
        )