"""
EmbeddingAgent: 메일 본문을 청킹하고 각 청크를 임베딩으로 변환하는 Agent.

Author: NEXUS Team
Date: 2025-01-12
"""
from agent.base_agent import BaseAgent
from app.core.text_utils import split_text_into_chunks
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class EmbeddingAgent(BaseAgent):
    """
    메일 본문을 청킹하고 각 청크를 임베딩으로 변환하는 Agent.

    계층 구조:
        - API: 라우팅만
        - Service: Agent 조율 + DB 저장
        - Agent: 순수 AI 로직 (청킹 + 임베딩 생성)

    Example:
        >>> agent = EmbeddingAgent()
        >>> email_data = {
        ...     'email_id': 'uuid',
        ...     'subject': '프로젝트 일정',
        ...     'body': '메일 본문...',
        ...     'folder': 'Inbox',
        ...     'from_name': '홍길동',
        ...     'date': datetime.now()
        ... }
        >>> results = await agent.process(email_data)
        >>> len(results)  # 청크 개수
        3
    """

    async def process(
        self,
        email_data: Dict[str, Any],
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        메일 데이터를 받아 청킹 + 임베딩 생성.

        Args:
            email_data: {
                'email_id': UUID,
                'subject': str,
                'body': str,
                'folder': 'Inbox' or 'SentItems',
                'from_name': str (Inbox용),
                'to_recipients': str (SentItems용),
                'date': datetime
            }
            chunk_size: 청크 크기 (기본 1000자)
            overlap: 오버랩 크기 (기본 200자, 20%)

        Returns:
            List of {
                'chunk_index': int,
                'chunk_text': str (메타데이터 포함),
                'embedding': List[float],
                'metadata': dict
            }

        Raises:
            ValueError: 본문이 너무 짧거나 비어있을 때
        """
        body = email_data.get('body', '')
        if not body or len(body.strip()) < 50:
            logger.warning(f"Email {email_data.get('email_id')}: Body too short, skipping")
            raise ValueError("Email body is too short or empty (min 50 characters)")

        # 1. 청킹 (오버랩 포함)
        chunks = split_text_into_chunks(body, chunk_size, overlap)
        logger.info(
            f"📧 Email {email_data.get('email_id')}: "
            f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={overlap})"
        )

        # 2. 각 청크마다 임베딩 생성
        results = []
        for idx, chunk in enumerate(chunks):
            # 메타데이터 포함한 텍스트 생성
            formatted_text = self._format_chunk_text(email_data, chunk)

            # OpenAI 임베딩 API 호출
            embedding = await self._generate_embedding(formatted_text)

            results.append({
                'chunk_index': idx,
                'chunk_text': formatted_text,
                'embedding': embedding,
                'metadata': self._build_metadata(email_data)
            })

            logger.debug(f"  ✅ Chunk {idx}/{len(chunks)-1}: Embedded {len(chunk)} chars")

        logger.info(f"🎯 Email {email_data.get('email_id')}: Generated {len(results)} embeddings")
        return results

    def _format_chunk_text(self, email_data: Dict, chunk: str) -> str:
        """
        청크에 메타데이터 추가.

        메타데이터를 포함함으로써:
        - "홍길동이 보낸 메일" 같은 쿼리에서 발신자 이름도 검색 가능
        - 제목, 날짜 정보가 벡터에 반영되어 컨텍스트 풍부화

        Args:
            email_data: 메일 정보
            chunk: 청크 텍스트

        Returns:
            메타데이터가 포함된 포맷된 텍스트
        """
        folder = email_data.get('folder')
        subject = email_data.get('subject', '(제목 없음)')
        date = email_data.get('date')
        date_str = date.strftime('%Y-%m-%d %H:%M') if date else '(날짜 없음)'

        if folder == 'SentItems':
            to_recipients = email_data.get('to_recipients', '(수신자 없음)')
            return f"제목: {subject}\n수신자: {to_recipients}\n날짜: {date_str}\n내용:\n{chunk}"
        else:  # Inbox or other
            from_name = email_data.get('from_name', '(발신자 없음)')
            return f"제목: {subject}\n발신자: {from_name}\n날짜: {date_str}\n내용:\n{chunk}"

    def _build_metadata(self, email_data: Dict) -> Dict:
        """
        JSONB 메타데이터 생성 (SQL 필터링용).

        chunk_text에 이미 메타데이터가 포함되어 있지만,
        JSONB 메타데이터는 SQL 필터링/정렬에 사용됩니다.

        Args:
            email_data: 메일 정보

        Returns:
            JSONB 메타데이터 딕셔너리
        """
        date = email_data.get('date')
        return {
            'subject': email_data.get('subject'),
            'from_name': email_data.get('from_name'),
            'to_recipients': email_data.get('to_recipients'),
            'date': date.isoformat() if date else None,
            'folder': email_data.get('folder'),
            'has_attachments': email_data.get('has_attachments', False)
        }

    async def _generate_embedding(self, text: str) -> List[float]:
        """
        OpenAI API로 임베딩 생성.

        Model: text-embedding-ada-002
        - 1536 dimensions
        - ~8191 tokens max
        - 1000자 ≈ 250-330 토큰

        Args:
            text: 임베딩할 텍스트 (메타데이터 포함)

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
