"""
문서 요약 Agent (Document Summarizer Agent)

여러 문서를 요약하여 번역 컨텍스트를 생성하는 Micro Agent.
단일 책임: 문서 리스트 → 핵심 요약

재사용 시나리오:
- 번역 컨텍스트 생성: 프로젝트 문서 요약
- 문서 미리보기: 긴 문서의 핵심 내용
- 검색 결과 요약: 검색된 문서들의 요약
- 프로젝트 개요 생성: 프로젝트 설명 자동 생성
"""

import logging
from typing import List
from agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class DocumentSummarizerAgent(BaseAgent):
    """
    문서 요약 Agent.

    책임: 문서 리스트 → 핵심 요약

    여러 문서의 내용을 분석하여 핵심 주제와 맥락을 요약합니다.
    번역 시 컨텍스트로 활용하거나 문서 미리보기에 사용됩니다.

    예시:
        >>> agent = DocumentSummarizerAgent()
        >>> documents = [
        ...     "클라우드 컴퓨팅 아키텍처 문서...",
        ...     "마이크로서비스 설계 가이드..."
        ... ]
        >>> summary = await agent.process(documents, max_length=500)
        >>> print(summary)
        "이 프로젝트는 클라우드 기반 마이크로서비스..."
    """

    def _create_system_prompt(self) -> str:
        """
        시스템 프롬프트를 생성합니다.

        Returns:
            시스템 프롬프트 문자열
        """
        return """당신은 문서 요약 전문가입니다.
여러 문서의 내용을 분석하여 핵심 주제와 맥락을 간결하게 요약하세요.

**요약 원칙**:
1. 문서들의 공통 주제와 핵심 내용 파악
2. 전문 용어와 기술적 맥락 유지
3. 프로젝트의 도메인과 목적 명확히 표현
4. 불필요한 세부사항은 생략

**출력 형식**:
- 2-3개 문단으로 구성
- 각 문단은 2-3문장
- 명확하고 간결한 문장 사용
- 전문 용어는 그대로 유지

**중요**:
- 요약문만 출력하고 추가 설명은 포함하지 마세요
- 문서에 없는 내용을 추가하지 마세요
- 객관적이고 사실적인 요약을 제공하세요
"""

    async def process(
        self,
        documents: List[str],
        max_length: int = 500
    ) -> str:
        """
        여러 문서를 요약하여 컨텍스트를 생성합니다.

        Args:
            documents: 문서 텍스트 리스트
            max_length: 최대 요약 길이 (문자 수, 기본값: 500)

        Returns:
            요약된 컨텍스트 텍스트

        Raises:
            ValueError: 문서가 비어있거나 유효하지 않은 경우
            Exception: GPT API 호출 실패 시
        """
        if not documents:
            raise ValueError("요약할 문서가 없습니다")

        # 빈 문서 제거
        valid_documents = [doc.strip() for doc in documents if doc and doc.strip()]

        if not valid_documents:
            raise ValueError("유효한 문서가 없습니다")

        logger.info(f"📝 문서 요약 시작: {len(valid_documents)}개 문서")

        try:
            # 문서들을 하나로 합치기
            combined_text = "\n\n---\n\n".join(valid_documents)
            
            # 50,000자 이하: 한 번에 처리
            # 50,000자 초과: 청크 분할 → 개별 요약 → 최종 통합
            MAX_CHARS = 50000
            
            if len(combined_text) <= MAX_CHARS:
                # 단일 요약
                logger.info(f"📄 단일 요약 모드: {len(combined_text)}자")
                summary = await self._summarize_text(combined_text, max_length)
            else:
                # 청크 분할 요약 (Map-Reduce)
                logger.info(f"📚 청크 분할 모드: {len(combined_text)}자 → 청크 분할")
                summary = await self._summarize_with_chunks(combined_text, max_length, MAX_CHARS)

            logger.info(f"✅ 요약 완료: {len(combined_text)}자 → {len(summary)}자")
            return summary

        except Exception as e:
            logger.error(f"❌ 요약 실패: {str(e)}")
            raise Exception(f"문서 요약 중 오류 발생: {str(e)}")

    async def _summarize_text(self, text: str, max_length: int) -> str:
        """
        단일 텍스트를 요약합니다.

        Args:
            text: 요약할 텍스트
            max_length: 최대 요약 길이

        Returns:
            요약된 텍스트
        """
        system_prompt = self._create_system_prompt()
        user_prompt = f"""다음 문서들을 {max_length}자 이내로 요약하세요:

{text}
"""

        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        return response.choices[0].message.content.strip()

    async def _summarize_with_chunks(
        self,
        text: str,
        max_length: int,
        chunk_size: int
    ) -> str:
        """
        긴 텍스트를 청크로 분할하여 요약합니다 (Map-Reduce 패턴).

        1단계 (Map): 각 청크를 개별 요약
        2단계 (Reduce): 개별 요약들을 최종 통합 요약

        Args:
            text: 전체 텍스트
            max_length: 최종 요약 최대 길이
            chunk_size: 청크 크기

        Returns:
            최종 통합 요약
        """
        # 1. 텍스트를 청크로 분할
        chunks = self._split_into_chunks(text, chunk_size)
        logger.info(f"📦 {len(chunks)}개 청크로 분할")

        # 2. 각 청크 개별 요약 (Map 단계)
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"🔄 청크 {i+1}/{len(chunks)} 요약 중...")
            chunk_summary = await self._summarize_text(chunk, max_length=500)
            chunk_summaries.append(chunk_summary)

        # 3. 개별 요약들을 통합 (Reduce 단계)
        logger.info(f"🔗 {len(chunk_summaries)}개 요약 통합 중...")
        combined_summaries = "\n\n---\n\n".join(chunk_summaries)

        # 통합된 요약이 여전히 길면 재귀적으로 처리
        if len(combined_summaries) > chunk_size:
            return await self._summarize_with_chunks(combined_summaries, max_length, chunk_size)

        # 최종 요약 생성
        final_summary = await self._summarize_text(combined_summaries, max_length)
        return final_summary

    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """
        텍스트를 지정된 크기의 청크로 분할합니다.
        문단 경계를 고려하여 자연스럽게 분할합니다.

        Args:
            text: 분할할 텍스트
            chunk_size: 청크 크기

        Returns:
            청크 리스트
        """
        chunks = []
        current_pos = 0
        text_length = len(text)

        while current_pos < text_length:
            # 청크 끝 위치 계산
            end_pos = min(current_pos + chunk_size, text_length)

            # 마지막 청크가 아니면 문단 경계 찾기
            if end_pos < text_length:
                # 가까운 문단 경계 찾기 (역순으로)
                boundary_pos = text.rfind("\n\n", current_pos, end_pos)
                if boundary_pos > current_pos:
                    end_pos = boundary_pos

            chunk = text[current_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)

            current_pos = end_pos
            # 문단 경계에서 분할했으면 공백 건너뛰기
            while current_pos < text_length and text[current_pos] in "\n\r\t ":
                current_pos += 1

        return chunks
