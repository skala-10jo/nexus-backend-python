"""
SearchAgent: 사용자 쿼리를 임베딩으로 변환하고 pgvector로 검색하는 Agent.

Author: NEXUS Team
Date: 2025-01-12
"""
from agent.base_agent import BaseAgent
from app.models.email import EmailEmbedding, Email
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SearchAgent(BaseAgent):
    """
    사용자 쿼리를 임베딩으로 변환하고 pgvector로 검색하는 Agent.

    하이브리드 검색 전략:
        1. SQL 필터로 범위 축소 (user_id, folder, date)
        2. pgvector로 의미 기반 검색 (RAG)
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

    async def process(
        self,
        query: str,
        user_id: str,
        db: Session,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        folder: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        자연어 쿼리로 메일 검색 (SQL 필터 + RAG).

        Args:
            query: 사용자 검색 쿼리 ("~내용에 관한 메일 있었나?")
            user_id: 현재 사용자 ID
            db: DB 세션
            top_k: 최대 결과 개수 (기본 10)
            similarity_threshold: 최소 유사도 (0~1, 기본 0.7)
            folder: 폴더 필터 (선택, 'Inbox' or 'SentItems')
            date_from: 시작 날짜 (선택, 'YYYY-MM-DD')
            date_to: 종료 날짜 (선택, 'YYYY-MM-DD')

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
        logger.info(f"🔍 Generated embedding for query: '{query[:50]}...'")

        # 2. SQL 필터 동적 생성
        filters = ["e.user_id = :user_id"]
        # pgvector는 리스트를 문자열로 변환해서 전달
        query_embedding_str = str(query_embedding)
        params = {
            'query_embedding': query_embedding_str,
            'user_id': user_id,
            'threshold': similarity_threshold,
            'top_k': top_k
        }

        if folder:
            filters.append("metadata->>'folder' = :folder")
            params['folder'] = folder

        if date_from:
            filters.append("metadata->>'date' >= :date_from")
            params['date_from'] = date_from

        if date_to:
            filters.append("metadata->>'date' <= :date_to")
            params['date_to'] = date_to

        where_clause = " AND ".join(filters)

        # 3. pgvector cosine similarity 검색
        sql_query = text(f"""
            SELECT
                ee.email_id,
                ee.chunk_text,
                ee.metadata,
                1 - (ee.embedding <=> CAST(:query_embedding AS vector)) AS similarity,
                e.subject,
                e.from_name,
                e.to_recipients,
                e.folder,
                e.received_date_time,
                e.sent_date_time
            FROM email_embeddings ee
            JOIN emails e ON ee.email_id = e.id
            WHERE {where_clause}
              AND 1 - (ee.embedding <=> CAST(:query_embedding AS vector)) > :threshold
            ORDER BY similarity DESC
            LIMIT :top_k
        """)

        results = db.execute(sql_query, params).fetchall()

        # 4. 결과 포맷팅 (중복 메일 제거 - 가장 유사도 높은 청크만)
        search_results = []
        seen_emails = set()

        for row in results:
            email_id = row.email_id

            # 같은 메일의 여러 청크가 매칭될 수 있으므로, 가장 유사도 높은 것만 반환
            if email_id in seen_emails:
                continue
            seen_emails.add(email_id)

            search_results.append({
                'email_id': str(email_id),
                'subject': row.subject or '(제목 없음)',
                'from_name': row.from_name,
                'to_recipients': row.to_recipients,
                'folder': row.folder,
                'date': row.received_date_time or row.sent_date_time,
                'similarity': float(row.similarity),
                'matched_chunk': row.chunk_text[:200] + '...' if len(row.chunk_text) > 200 else row.chunk_text
            })

        logger.info(
            f"✅ Found {len(search_results)} matching emails "
            f"(filters: folder={folder}, date_from={date_from}, date_to={date_to})"
        )
        return search_results

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
