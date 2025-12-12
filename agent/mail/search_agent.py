"""
SearchAgent: 사용자 쿼리를 임베딩으로 변환하고 Qdrant로 검색하는 Agent.


Updated: 2025-01-17 (Qdrant 연동)
"""
from agent.base_agent import BaseAgent
from app.core.qdrant_client import get_qdrant_client
from app.core.text_utils import strip_html_tags
from app.models.email import Email
from app.config import settings
from qdrant_client.http import models
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SearchAgent(BaseAgent):
    """
    사용자 쿼리를 임베딩으로 변환하고 Qdrant로 검색하는 Agent.

    하이브리드 검색 전략:
        1. Qdrant 필터로 범위 축소 (user_id, folder, date)
        2. Qdrant로 의미 기반 검색 (RAG)
        3. 유사도 높은 순으로 정렬

    Example:
        >>> agent = SearchAgent()
        >>> results = await agent.process(
        ...     query="프로젝트 일정 회의",
        ...     user_id="uuid",
        ...     db=db_session,
        ...     folder="Inbox",
        ...     top_k=10
        ... )
        >>> len(results)  # 최대 10개
    """

    def __init__(self):
        super().__init__()
        self.qdrant_client = get_qdrant_client()

    async def process(
        self,
        query: str,
        user_id: str,
        db: Session,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        folder: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        project_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        자연어 쿼리로 메일 검색 (Qdrant 필터 + RAG).

        Args:
            query: 사용자 검색 쿼리 ("~내용에 관한 메일 있었나?")
            user_id: 현재 사용자 ID
            db: DB 세션 (메일 메타데이터 조회용)
            top_k: 최대 결과 개수 (기본 10)
            similarity_threshold: 최소 유사도 (0~1, 기본 0.7)
            folder: 폴더 필터 (선택, 'Inbox' or 'SentItems')
            date_from: 시작 날짜 (선택, 'YYYY-MM-DD')
            date_to: 종료 날짜 (선택, 'YYYY-MM-DD')
            project_name: 프로젝트명 필터 (선택, 'prototype-dev' 등)

        Returns:
            List of {
                'email_id': UUID,
                'subject': str,
                'from_name': str,
                'to_recipients': str,
                'folder': str,
                'date': datetime,
                'similarity': float,
                'matched_chunk': str  # 매칭된 청크 텍스트 미리보기
            }

        Raises:
            ValueError: 쿼리가 비어있을 때
        """
        if not query or len(query.strip()) < 2:
            raise ValueError("Query is too short (min 2 characters)")

        # 1. 쿼리를 임베딩으로 변환
        query_embedding = await self._generate_embedding(query)
        logger.info(f"Generated embedding for query: '{query[:50]}...'")

        # 2. Qdrant 필터 동적 생성
        filter_conditions = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id)
            )
        ]

        if folder:
            filter_conditions.append(
                models.FieldCondition(
                    key="folder",
                    match=models.MatchValue(value=folder)
                )
            )

        if date_from:
            filter_conditions.append(
                models.FieldCondition(
                    key="date",
                    range=models.Range(gte=date_from)
                )
            )

        if date_to:
            filter_conditions.append(
                models.FieldCondition(
                    key="date",
                    range=models.Range(lte=date_to)
                )
            )

        if project_name:
            filter_conditions.append(
                models.FieldCondition(
                    key="project_name",
                    match=models.MatchValue(value=project_name)
                )
            )

        # 3. Qdrant 벡터 검색
        search_results = self.qdrant_client.search(
            collection_name=settings.QDRANT_EMAIL_COLLECTION,
            query_vector=query_embedding,
            query_filter=models.Filter(must=filter_conditions) if filter_conditions else None,
            limit=top_k * 2,  # 중복 제거 위해 넉넉하게 검색
            score_threshold=similarity_threshold
        )

        logger.info(f"Qdrant search returned {len(search_results)} results")

        # 4. 결과 포맷팅 (중복 메일 제거 - 가장 유사도 높은 청크만)
        formatted_results = []
        seen_emails = set()

        for hit in search_results:
            email_id = hit.payload.get('email_id')

            # 같은 메일의 여러 청크가 매칭될 수 있으므로, 가장 유사도 높은 것만 반환
            if email_id in seen_emails:
                continue
            seen_emails.add(email_id)

            # DB에서 이메일 전체 정보 조회
            email = db.query(Email).filter(Email.id == email_id).first()
            if not email:
                logger.warning(f"Email {email_id} not found in DB, skipping")
                continue

            # HTML 제거 후 반환 (AnswerAgent에 깔끔한 텍스트 제공)
            clean_body = strip_html_tags(email.body) if email.body else ''

            formatted_results.append({
                'email_id': str(email_id),
                'subject': email.subject or '(제목 없음)',
                'from_name': email.from_name,
                'to_recipients': email.to_recipients,
                'folder': email.folder,
                'date': email.received_date_time or email.sent_date_time,
                'similarity': float(hit.score),
                'matched_chunk': hit.payload.get('chunk_text', '')[:200] + '...',
                'full_body': clean_body,  # HTML 제거된 전체 본문
                'project_name': email.project.name if email.project else None  # 프로젝트명 추가
            })

            # top_k개만 반환
            if len(formatted_results) >= top_k:
                break

        logger.info(
            f"Found {len(formatted_results)} matching emails "
            f"(filters: folder={folder}, date_from={date_from}, date_to={date_to}, project_name={project_name})"
        )
        return formatted_results

    async def _generate_embedding(self, text: str) -> List[float]:
        """
        OpenAI API로 쿼리 임베딩 생성.

        사용자 쿼리와 메일 청크를 같은 모델로 임베딩해야
        코사인 유사도 계산이 의미있습니다.

        Args:
            text: 검색 쿼리

        Returns:
            임베딩 벡터 (1536 dimensions)

        Raises:
            OpenAIError: OpenAI API 호출 실패 시
        """
        response = await self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
