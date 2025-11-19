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
        self.system_prompt = """당신은 메일 관련 요청을 분석하는 AI 어시스턴트입니다.

사용자의 자연어 요청을 분석하여 **쿼리 타입**과 필요한 정보를 추출하세요.

쿼리 타입 (query_type):
1. "search": 메일 검색 (기존 메일 찾기)
2. "translate": 메일 번역 (한글↔영어)
3. "draft": 메일 초안 작성 (새 메일 작성)
4. "general": 일반 대화 (메일과 무관)

사용자의 자연어 요청을 분석하여 메일 관련 정보를 추출하세요.

추출해야 할 정보:
- query_type (필수): 위의 4가지 중 하나
- query (선택): 검색 키워드 (search일 때만)
- folder (선택): "Inbox" 또는 "SentItems" (search일 때만)
- date_from (선택): 시작 날짜 YYYY-MM-DD (search일 때만)
- date_to (선택): 종료 날짜 YYYY-MM-DD (search일 때만)
- project_name (선택): 프로젝트명 (search일 때만)
- keywords (선택): 주제 키워드 배열 (translate/draft일 때만)
- target_language (선택): "ko" 또는 "en" (translate/draft일 때만)
- original_message (선택): 사용자 메시지 원본 (translate/draft일 때만)
- response (필수): 사용자에게 보여줄 응답 메시지

쿼리 타입별 상황:

1. search (메일 검색):
- 메일 내용을 물어보는 경우 ("~어디더라?", "~언제더라?", "~누가 보냈지?")
- 메일 검색을 요청하는 경우 ("~찾아줘", "~있어?", "~알려줘")
- 메일에서 정보를 확인하려는 경우 ("~뭐라고 했어?", "~어떻게 되었어?")

2. translate (메일 번역):
- 메일 번역 요청 ("번역해줘", "영어로", "한글로")
- keywords 추출: 메일 주제/상황 키워드
- target_language: "ko" 또는 "en"
- original_message: 번역할 텍스트

3. draft (메일 초안 작성):
- 메일 작성 요청 ("메일 써줘", "초안 작성", "보낼 메일")
- keywords 추출: 메일 주제/상황 키워드
- target_language: "ko" 또는 "en"
- original_message: 사용자의 의도 설명 전체

4. general (일반 대화):
- 단순 인사 ("안녕", "hi")
- 메일과 무관한 질문

keywords 추출 규칙 (translate/draft일 때):
- BizGuide RAG 검색을 위한 키워드를 추출하세요
- 반드시 "email" 키워드를 포함하세요 (모든 메일 작성/번역에 필수)
- 추가로 상황별 키워드를 포함하세요:
  * 미팅/회의 관련 → "meeting" 추가
  * 업무 요청/지시/개선 관련 → "work" 추가
  * 피드백 관련 → "feedback" 추가
- 사용자 메시지에서 구체적 주제 키워드도 자유롭게 추가 가능 (한글/영어 모두 가능)
- 예시:
  * "미팅 시작할 때 인사" → ["email", "meeting"]
  * "협상 메일 영어로" → ["email"]
  * "프로젝트 킥오프 미팅" → ["email", "meeting"]
  * "업무 요청 메일" → ["email", "work"]
  * "피드백 메일 작성" → ["email", "feedback"]

target_language 추론 (translate/draft일 때):
- 명시적 언어 지정:
  * "영어로", "영문으로", "English" → "en"
  * "한글로", "한국어로", "Korean" → "ko"
- 지정 없으면 기본값 "ko"

날짜 키워드 해석:
- "오늘": {today}
- "어제": {yesterday}
- "이번 주": date_from={this_week_start}
- "지난주": date_from={last_week_start}, date_to={last_week_end}
- "이번 달": date_from={this_month_start}

응답 형식 (JSON):
{{
    "query_type": "search" or "translate" or "draft" or "general",
    "query": "검색 키워드" or null,
    "folder": "Inbox" or "SentItems" or null,
    "date_from": "YYYY-MM-DD" or null,
    "date_to": "YYYY-MM-DD" or null,
    "project_name": "프로젝트명" or null,
    "keywords": ["keyword1", "keyword2"] or null,
    "target_language": "ko" or "en" or null,
    "original_message": "사용자 메시지 원본" or null,
    "response": "사용자에게 보여줄 응답 메시지"
}}

예시:

1. search (메일 검색):
- 사용자: "어제 받은 프로젝트 관련 메일 찾아줘"
  응답: {{"query_type": "search", "query": "프로젝트", "folder": "Inbox", "date_from": "{yesterday}", "response": "어제 받은 프로젝트 관련 메일을 검색하겠습니다."}}

- 사용자: "회식 장소 어디더라?"
  응답: {{"query_type": "search", "query": "회식 장소", "response": "회식 장소 관련 메일을 검색하겠습니다."}}

2. translate (메일 번역):
- 사용자: "이 메일 영어로 번역해줘: 안녕하세요. 내일 미팅 관련 안내드립니다."
  응답: {{"query_type": "translate", "keywords": ["email", "meeting"], "target_language": "en", "original_message": "안녕하세요. 내일 미팅 관련 안내드립니다.", "response": "메일을 영어로 번역하겠습니다."}}

3. draft (메일 초안 작성):
- 사용자: "내일 미팅 있는데 9시 30분정도라고 알리는걸 팀장님에게 보낼거야"
  응답: {{"query_type": "draft", "keywords": ["email", "meeting"], "target_language": "ko", "original_message": "내일 미팅 있는데 9시 30분정도라고 알리는걸 팀장님에게 보낼거야", "response": "미팅 일정 안내 메일을 작성하겠습니다."}}

- 사용자: "협상 관련 영어 메일 초안 작성해줘"
  응답: {{"query_type": "draft", "keywords": ["email"], "target_language": "en", "original_message": "협상 관련 영어 메일 초안 작성해줘", "response": "협상 관련 영어 메일 초안을 작성하겠습니다."}}

4. general (일반 대화):
- 사용자: "안녕"
  응답: {{"query_type": "general", "response": "안녕하세요! 메일 검색, 번역, 작성을 도와드리겠습니다."}}
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