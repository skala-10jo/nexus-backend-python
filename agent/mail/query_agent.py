"""
QueryAgent: 사용자의 자연어 대화에서 메일 검색 쿼리를 추출하는 Agent.

Author: NEXUS Team
Date: 2025-01-17
"""
from agent.base_agent import BaseAgent
from typing import Dict, Any, Optional
import json
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class QueryAgent(BaseAgent):
    """
    사용자의 자연어 대화를 분석하여 메일 검색 쿼리를 추출하는 Agent.

    대화형 인터페이스:
        1. 사용자 메시지 분석
        2. 검색 의도 파악
        3. 검색 파라미터 추출 (query, folder, date_from, date_to, project_name)

    Example:
        >>> agent = QueryAgent()
        >>> result = await agent.process(
        ...     user_message="prototype-dev 프로젝트 관련 메일 가져와줘",
        ...     conversation_history=[]
        ... )
        >>> result
        {
            "query": "메일",
            "project_name": "prototype-dev",
            "needs_search": True,
            "response": "prototype-dev 프로젝트 관련 메일을 검색하겠습니다."
        }
    """

    def __init__(self):
        super().__init__()
        self.system_prompt = """당신은 메일 검색 전문 AI 어시스턴트입니다.

사용자의 자연어 요청을 분석하여 메일 검색에 필요한 정보를 추출하세요.

추출해야 할 정보:
1. query (필수): 검색할 키워드 (예: "프로젝트", "일정", "회의", "회식")
2. folder (선택): "Inbox" (받은편지함) 또는 "SentItems" (보낸편지함)
3. date_from (선택): 시작 날짜 (YYYY-MM-DD)
4. date_to (선택): 종료 날짜 (YYYY-MM-DD)
5. project_name (선택): 프로젝트명 (예: "prototype-dev", "nexus", "backend-migration")
6. needs_search (필수): 검색이 필요한지 여부 (true/false)

검색이 필요한 경우 (needs_search: true):
- 메일 내용을 물어보는 경우 ("~어디더라?", "~언제더라?", "~누가 보냈지?")
- 메일 검색을 요청하는 경우 ("~찾아줘", "~있어?", "~알려줘")
- 메일에서 정보를 확인하려는 경우 ("~뭐라고 했어?", "~어떻게 되었어?")

검색이 불필요한 경우 (needs_search: false):
- 단순 인사 ("안녕", "hi")
- 메일과 무관한 질문

날짜 키워드 해석:
- "오늘": {today}
- "어제": {yesterday}
- "이번 주": date_from={this_week_start}
- "지난주": date_from={last_week_start}, date_to={last_week_end}
- "이번 달": date_from={this_month_start}

응답 형식 (JSON):
{{
    "query": "검색 키워드",
    "folder": "Inbox" or "SentItems" or null,
    "date_from": "YYYY-MM-DD" or null,
    "date_to": "YYYY-MM-DD" or null,
    "project_name": "프로젝트명" or null,
    "needs_search": true or false,
    "response": "사용자에게 보여줄 응답 메시지"
}}

예시:
- 사용자: "어제 받은 프로젝트 관련 메일 찾아줘"
  응답: {{"query": "프로젝트", "folder": "Inbox", "date_from": "{yesterday}", "needs_search": true, "response": "어제 받은 프로젝트 관련 메일을 검색하겠습니다."}}

- 사용자: "회식 장소 어디더라?"
  응답: {{"query": "회식 장소", "needs_search": true, "response": "회식 장소 관련 메일을 검색하겠습니다."}}

- 사용자: "prototype-dev 프로젝트 관련 메일 가져와줘"
  응답: {{"query": "메일", "project_name": "prototype-dev", "needs_search": true, "response": "prototype-dev 프로젝트 관련 메일을 검색하겠습니다."}}

- 사용자: "안녕"
  응답: {{"needs_search": false, "response": "안녕하세요! 메일 검색을 도와드리겠습니다. 어떤 메일을 찾으시나요?"}}
"""

    async def process(
        self,
        user_message: str,
        conversation_history: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        사용자 메시지를 분석하여 검색 쿼리를 추출합니다.

        Args:
            user_message: 사용자의 자연어 메시지
            conversation_history: 이전 대화 내역 (선택)

        Returns:
            {
                "query": str (검색 키워드),
                "folder": str or None,
                "date_from": str or None,
                "date_to": str or None,
                "project_name": str or None,
                "needs_search": bool,
                "response": str (사용자에게 보여줄 메시지)
            }

        Raises:
            ValueError: 메시지가 비어있을 때
        """
        if not user_message or len(user_message.strip()) < 1:
            raise ValueError("Message is empty")

        # 날짜 정보 계산
        now_utc = datetime.now(timezone.utc)

        today = now_utc.strftime('%Y-%m-%d')
        yesterday = (now_utc - timedelta(days=1)).strftime('%Y-%m-%d')

        # 이번 주 시작 (UTC 기준, 월요일)
        this_week_start = (now_utc - timedelta(days=now_utc.weekday())).strftime('%Y-%m-%d')

        # 지난주
        last_week_end = this_week_start
        last_week_start = (now_utc - timedelta(days=now_utc.weekday() + 7)).strftime('%Y-%m-%d')

        # 이번 달 시작
        this_month_start = now_utc.replace(day=1).strftime('%Y-%m-%d')

        # 시스템 프롬프트에 날짜 정보 삽입
        system_prompt = self.system_prompt.format(
            today=today,
            yesterday=yesterday,
            this_week_start=this_week_start,
            last_week_start=last_week_start,
            last_week_end=last_week_end,
            this_month_start=this_month_start
        )

        # 대화 메시지 구성
        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_message})

        # GPT 호출
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content
            result = json.loads(result_text)

            logger.info(f"Query extraction result: {result}")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GPT response: {e}")
            return {
                "needs_search": False,
                "response": "죄송합니다. 요청을 이해하지 못했습니다. 다시 말씀해주시겠어요?"
            }

        except Exception as e:
            logger.error(f"Query extraction failed: {str(e)}")
            return {
                "needs_search": False,
                "response": "오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
            }