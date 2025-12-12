"""
MailAgentService: 메일 임베딩, 검색, 챗봇 오케스트레이션.

Updated: 2025-01-17 (Qdrant 연동)
Updated: 2025-12-12 (chat() 메서드 추가 - API 계층 리팩토링)
"""
from agent.mail.embedding_agent import EmbeddingAgent
from agent.mail.search_agent import SearchAgent
from agent.mail.query_agent import QueryAgent
from agent.mail.answer_agent import AnswerAgent

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

    async def chat(
        self,
        message: str,
        user_id: str,
        db: Session,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        메일 검색/작성/번역 통합 챗봇

        QueryAgent가 쿼리 타입을 분석하고:
        - search: 메일 검색
        - draft: 메일 초안 작성 (RAG 통합)
        - translate: 메일 번역 (RAG 통합)
        - general: 일반 대화

        Args:
            message: 사용자 메시지
            user_id: 사용자 ID
            db: DB 세션
            conversation_history: 대화 히스토리 (선택)

        Returns:
            챗봇 응답 데이터
        """
        # Lazy import to avoid circular dependency
        from app.services.email_draft_service import EmailDraftService

        logger.info(f"Chat request: message='{message[:50]}...', user={user_id}")

        # 1. QueryAgent로 쿼리 분석
        query_agent = QueryAgent()
        history = conversation_history or []

        query_result = await query_agent.process(
            user_message=message,
            conversation_history=history
        )

        logger.info(f"Query extraction result: {query_result}")

        query_type = query_result.get("query_type", "general")
        answer = query_result.get("response", "")

        # 응답 초기화
        response_data: Dict[str, Any] = {
            "query_type": query_type,
            "answer": answer
        }

        # 2. query_type별 처리
        if query_type == "search":
            # 메일 검색
            results = await self.search_emails(
                query=query_result.get("query", ""),
                user_id=user_id,
                db=db,
                top_k=5,
                folder=query_result.get("folder"),
                date_from=query_result.get("date_from"),
                date_to=query_result.get("date_to"),
                project_name=query_result.get("project_name")
            )

            # AnswerAgent로 자연어 답변 생성
            answer_agent = AnswerAgent()
            answer = await answer_agent.process(
                user_query=message,
                search_results=results,
                conversation_history=history
            )

            response_data.update({
                "query": query_result.get("query"),
                "folder": query_result.get("folder"),
                "date_from": query_result.get("date_from"),
                "date_to": query_result.get("date_to"),
                "project_name": query_result.get("project_name"),
                "needs_search": True,
                "answer": answer,
                "search_results": results  # raw dict list (API에서 변환)
            })

        elif query_type == "draft":
            # 메일 초안 작성 (RAG 통합)
            logger.info(f"Creating email draft with keywords: {query_result.get('keywords')}")

            draft_service = EmailDraftService()
            draft_result = await draft_service.create_draft(
                original_message=query_result.get("original_message", message),
                keywords=query_result.get("keywords"),
                target_language=query_result.get("target_language", "ko"),
                conversation_history=history
            )

            response_data.update({
                "email_draft": draft_result.get("email_draft"),
                "subject": draft_result.get("subject"),
                "rag_sections": draft_result.get("rag_sections", []),
                "answer": f"{answer}\n\n**제목:** {draft_result.get('subject')}\n\n**본문:**\n{draft_result.get('email_draft')}"
            })

        elif query_type == "translate":
            # 메일 번역 (RAG 통합)
            logger.info(f"Translating email with keywords: {query_result.get('keywords')}")

            draft_service = EmailDraftService()
            translate_result = await draft_service.translate_email(
                email_text=query_result.get("original_message", ""),
                keywords=query_result.get("keywords"),
                target_language=query_result.get("target_language", "en"),
                conversation_history=history
            )

            response_data.update({
                "translated_email": translate_result.get("translated_email"),
                "rag_sections": translate_result.get("rag_sections", []),
                "answer": f"{answer}\n\n**번역 결과:**\n{translate_result.get('translated_email')}"
            })

        # 3. general은 기본 answer만 반환

        return response_data