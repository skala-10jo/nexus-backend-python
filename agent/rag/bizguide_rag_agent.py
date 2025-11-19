"""
BizGuideRAGAgent: BizGuide 문서에서 관련 비즈니스 표현을 검색하는 RAG Agent.

메일 번역/초안 작성 시 비즈매너 표현을 제공하기 위한 Agent입니다.

Author: NEXUS Team
Date: 2025-01-18
"""
from agent.base_agent import BaseAgent
from app.core.qdrant_client import get_qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchAny
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BizGuideRAGAgent(BaseAgent):
    """
    BizGuide RAG Agent

    keywords 기반으로 Qdrant에서 관련 비즈니스 표현을 검색합니다.

    Example:
        >>> agent = BizGuideRAGAgent()
        >>> results = await agent.process(
        ...     query="meeting opening greeting",
        ...     keywords=["meeting", "greeting"],
        ...     top_k=3
        ... )
        >>> for chunk in results:
        ...     print(chunk["text"])
    """

    def __init__(self, collection_name: str = "bizguide"):
        super().__init__()
        self.qdrant = get_qdrant_client()
        self.collection_name = collection_name

    async def process(
        self,
        query: str,
        keywords: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        BizGuide에서 관련 표현을 검색합니다.

        Args:
            query: 검색 쿼리 (자연어)
            keywords: topic 필터링용 키워드 배열 (선택)
            top_k: 반환할 최대 결과 개수

        Returns:
            검색 결과 리스트 [
                {
                    "text": "표현 내용",
                    "chapter": "Chapter 1. ...",
                    "section": "Media Talk",
                    "topic": ["meeting", "greeting"],
                    "score": 0.85
                },
                ...
            ]

        Raises:
            Exception: 검색 실패 시
        """
        try:
            # 1. 쿼리 임베딩 생성
            logger.info(f"Generating embedding for query: {query[:50]}...")
            embedding_response = await self.client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_vector = embedding_response.data[0].embedding

            # 2. topic 필터 생성 (keywords가 있으면)
            query_filter = None
            if keywords:
                logger.info(f"Applying topic filter: {keywords}")
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="topic",
                            match=MatchAny(any=keywords)
                        )
                    ]
                )

            # 3. Qdrant 검색 (필터 적용)
            search_results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            )

            # 4. Fallback: 필터 결과가 0개면 유사도 검색만 수행
            if not search_results and query_filter:
                logger.warning(f"No results with topic filter. Falling back to similarity search only...")
                search_results = self.qdrant.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=None,  # 필터 제거 → 유사도 검색만
                    limit=top_k,
                    with_payload=True
                )
                logger.info(f"Fallback search (similarity only) returned {len(search_results)} results")

            # 5. 결과 포맷팅
            results = []
            for hit in search_results:
                results.append({
                    "text": hit.payload.get("text", ""),
                    "chapter": hit.payload.get("chapter", ""),
                    "section": hit.payload.get("section", ""),
                    "topic": hit.payload.get("topic", []),
                    "score": hit.score
                })

            logger.info(f"Found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"BizGuide RAG search failed: {str(e)}")
            # 실패해도 빈 배열 반환 (메일 작성은 계속 진행)
            return []

    async def search_for_email(
        self,
        original_message: str,
        keywords: Optional[List[str]] = None,
        target_language: str = "ko"
    ) -> List[Dict[str, Any]]:
        """
        메일 작성/번역을 위한 비즈니스 표현 검색

        Args:
            original_message: 사용자의 원본 메시지
            keywords: topic 키워드
            target_language: 목표 언어 ("ko" or "en")

        Returns:
            관련 비즈니스 표현 리스트
        """
        # 쿼리 구성: 원본 메시지 + 언어 힌트
        query = f"{original_message} business email {target_language}"

        return await self.process(
            query=query,
            keywords=keywords,
            top_k=3  # 메일용은 상위 3개만
        )
