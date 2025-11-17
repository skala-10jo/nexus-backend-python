"""
AnswerAgent: 검색 결과를 바탕으로 자연어 답변을 생성하는 Agent.

Author: NEXUS Team
Date: 2025-01-17
"""
from agent.base_agent import BaseAgent
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AnswerAgent(BaseAgent):
    """
    검색 결과를 분석하여 사용자 질문에 대한 자연어 답변을 생성하는 Agent.

    대화형 답변 생성:
        1. 검색 결과의 matched_chunk를 컨텍스트로 활용
        2. GPT-4o로 자연어 답변 생성
        3. 출처 정보 포함 (보낸이, 날짜)
        4. 대화 히스토리 지원 (연속 질문 가능)

    Example:
        >>> agent = AnswerAgent()
        >>> answer = await agent.process(
        ...     user_query="회식 장소 어디더라?",
        ...     search_results=[...],
        ...     conversation_history=[]
        ... )
        >>> print(answer)
        "홍길동님이 1월 15일에 보낸 메일에 따르면, 회식 장소는 등촌칼국수입니다."
    """

    def __init__(self):
        super().__init__()
        self.system_prompt = """당신은 메일 검색 결과를 바탕으로 사용자 질문에 답변하는 AI 어시스턴트입니다.

검색된 메일의 내용을 분석하여 명확하고 간결한 답변을 제공하세요.

답변 작성 가이드:
1. 반드시 출처를 포함하세요 (보낸이, 날짜)
2. 구체적인 정보를 우선하세요 (날짜, 시간, 장소 등)
3. 자연스러운 대화체로 작성하세요
4. 검색 결과가 여러 개면 가장 관련성 높은 것을 우선하세요
5. 불확실한 경우 명확하게 표현하세요
6. **중요**: 마크다운 문법(**, *, #, - 등)을 사용하지 말고 일반 텍스트로만 작성하세요

예시:
- 질문: "회식 장소 어디더라?"
- 검색 결과: [제목: 회식 안내, 보낸이: 홍길동, 날짜: 2025-01-15,
            내용: "장소는 등촌칼국수, 시간은 오후 2시입니다."]
- 답변: "홍길동님이 1월 15일에 보낸 메일에 따르면, 회식 장소는 등촌칼국수이고 시간은 오후 2시입니다."

검색 결과가 없는 경우:
"죄송합니다. 관련된 메일을 찾지 못했습니다."
"""

    async def process(
        self,
        user_query: str,
        search_results: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        검색 결과를 바탕으로 자연어 답변을 생성합니다.

        Args:
            user_query: 사용자의 원래 질문
            search_results: SearchAgent로부터 받은 검색 결과
            conversation_history: 이전 대화 내역 (연속 질문 지원)

        Returns:
            자연어 답변 문자열

        Example:
            >>> results = [
            ...     {
            ...         "subject": "회식 안내",
            ...         "from_name": "홍길동",
            ...         "date": "2025-01-15T10:00:00Z",
            ...         "matched_chunk": "장소는 등촌칼국수입니다.",
            ...         "similarity": 0.92
            ...     }
            ... ]
            >>> answer = await agent.process(
            ...     user_query="회식 장소 어디더라?",
            ...     search_results=results
            ... )
        """
        # 검색 결과가 없는 경우
        if not search_results or len(search_results) == 0:
            logger.warning(f"No search results for query: {user_query}")
            return "죄송합니다. 관련된 메일을 찾지 못했습니다."

        # 상위 5개 결과만 사용 (토큰 절약)
        top_results = search_results[:5]

        # 컨텍스트 구성
        context = self._build_context(top_results)

        # 대화 히스토리 구성
        messages = [{"role": "system", "content": self.system_prompt}]

        if conversation_history:
            # 이전 대화 내역 추가 (연속 질문 지원)
            messages.extend(conversation_history)

        # 사용자 질문 + 검색 결과 컨텍스트
        user_message = f"질문: {user_query}\n\n검색된 메일 정보:\n{context}"
        messages.append({"role": "user", "content": user_message})

        # GPT-4o 호출
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,  # 일관된 답변을 위해 낮은 temperature
                max_tokens=300     # 간결한 답변
            )

            answer = response.choices[0].message.content

            logger.info(f"Generated answer for query: '{user_query[:50]}...'")
            return answer

        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            # 에러 시 기본 응답
            return "죄송합니다. 답변 생성 중 오류가 발생했습니다. 다시 시도해주세요."

    def _build_context(self, search_results: List[Dict[str, Any]]) -> str:
        """
        검색 결과를 GPT가 이해할 수 있는 컨텍스트로 변환합니다.

        하이브리드 전략:
        - 1등 결과: full_body 전체 사용 (가장 관련성 높음)
        - 2-5등 결과: full_body의 처음 300자 사용 (컨텍스트 제공)

        Args:
            search_results: 검색 결과 리스트

        Returns:
            포맷팅된 컨텍스트 문자열
        """
        context_parts = []

        for i, result in enumerate(search_results, 1):
            # 날짜 포맷팅
            date_str = result.get('date', '')
            if date_str:
                # datetime 객체를 문자열로 변환
                if hasattr(date_str, 'strftime'):
                    date_str = date_str.strftime('%Y년 %m월 %d일')
                else:
                    # 이미 문자열인 경우 처리
                    date_str = str(date_str)[:10]  # YYYY-MM-DD 부분만

            # 하이브리드 컨텍스트 선택
            if i == 1:
                # 1등 결과: 전체 본문 사용 (가장 관련성 높음)
                content = result.get('full_body', result.get('matched_chunk', ''))
            else:
                # 2-5등 결과: 300자로 제한 (토큰 절약)
                full_body = result.get('full_body', '')
                if full_body:
                    content = full_body[:300]
                    if len(full_body) > 300:
                        content += '...'
                else:
                    # full_body가 없으면 matched_chunk 사용
                    content = result.get('matched_chunk', '')

            # 프로젝트명 포함 (있는 경우)
            project_info = ""
            if result.get('project_name'):
                project_info = f"\n프로젝트: {result.get('project_name')}"

            context_part = f"""[메일 {i}]
제목: {result.get('subject', '제목 없음')}
보낸이: {result.get('from_name', '알 수 없음')}
날짜: {date_str}
유사도: {result.get('similarity', 0):.2f}{project_info}
내용: {content}
"""
            context_parts.append(context_part)

        return "\n".join(context_parts)
