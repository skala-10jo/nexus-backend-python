"""
메일 작성/번역 RAG 통합 예시 (올바른 아키텍처)

전체 플로우:
1. QueryAgent: 쿼리 분석 + keyword 추출
2. EmailDraftService: Agent 조율 (RAG Agent + Draft Agent)
3. 결과 반환


"""
import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.mail.query_agent import QueryAgent
from app.services.email_draft_service import EmailDraftService


async def example_draft_email():
    """메일 초안 작성 예시 (Service 계층 사용)"""
    print("=" * 60)
    print("📧 메일 초안 작성 예시 (올바른 아키텍처)")
    print("=" * 60)

    # 사용자 쿼리
    user_message = "내일 미팅 있는데 9시 30분정도라고 알리는걸 팀장님에게 보낼거야"

    # 1단계: QueryAgent로 쿼리 분석
    print("\n[1단계] QueryAgent - 쿼리 분석")
    print(f"사용자 입력: {user_message}")

    query_agent = QueryAgent()
    query_result = await query_agent.process(user_message)

    print(f"쿼리 타입: {query_result.get('query_type')}")
    print(f"Keywords: {query_result.get('keywords')}")
    print(f"Target Language: {query_result.get('target_language')}")

    # 2단계: EmailDraftService로 메일 작성
    if query_result.get("query_type") == "draft":
        print("\n[2단계] EmailDraftService - Agent 조율")
        print("  ├─ BizGuideRAGAgent: RAG 검색")
        print("  └─ EmailDraftAgent: 메일 작성")

        service = EmailDraftService()
        result = await service.create_draft(
            original_message=query_result.get("original_message", user_message),
            keywords=query_result.get("keywords"),
            target_language=query_result.get("target_language", "ko")
        )

        print(f"\n✅ 작성 완료!")
        print(f"\n제목: {result['subject']}")
        print(f"\n본문:\n{result['email_draft']}")
        print(f"\n사용된 BizGuide 섹션: {', '.join(result['rag_sections'])}")


async def example_translate_email():
    """메일 번역 예시 (Service 계층 사용)"""
    print("\n" + "=" * 60)
    print("🌐 메일 번역 예시 (올바른 아키텍처)")
    print("=" * 60)

    # 사용자 쿼리
    user_message = "이 메일 영어로 번역해줘: 안녕하세요. 내일 오전 9시 30분에 미팅이 예정되어 있어 안내드립니다."

    # 1단계: QueryAgent
    print("\n[1단계] QueryAgent - 쿼리 분석")
    print(f"사용자 입력: {user_message}")

    query_agent = QueryAgent()
    query_result = await query_agent.process(user_message)

    print(f"쿼리 타입: {query_result.get('query_type')}")
    print(f"Keywords: {query_result.get('keywords')}")
    print(f"Target Language: {query_result.get('target_language')}")

    # 2단계: EmailDraftService로 번역
    if query_result.get("query_type") == "translate":
        print("\n[2단계] EmailDraftService - Agent 조율")
        print("  ├─ BizGuideRAGAgent: RAG 검색")
        print("  └─ EmailDraftAgent: 번역")

        service = EmailDraftService()
        result = await service.translate_email(
            email_text=query_result.get("original_message", ""),
            keywords=query_result.get("keywords"),
            target_language=query_result.get("target_language", "en")
        )

        print(f"\n✅ 번역 완료!")
        print(f"\n번역된 메일:\n{result['translated_email']}")
        print(f"\n사용된 BizGuide 섹션: {', '.join(result['rag_sections'])}")


async def example_search_only():
    """메일 검색 예시 (기존 기능)"""
    print("\n" + "=" * 60)
    print("🔍 메일 검색 예시 (기존 기능)")
    print("=" * 60)

    user_message = "어제 받은 프로젝트 관련 메일 찾아줘"

    query_agent = QueryAgent()
    query_result = await query_agent.process(user_message)

    print(f"사용자 입력: {user_message}")
    print(f"\n쿼리 타입: {query_result.get('query_type')}")
    print(f"검색 키워드: {query_result.get('query')}")
    print(f"폴더: {query_result.get('folder')}")
    print(f"날짜: {query_result.get('date_from')}")

    if query_result.get("query_type") == "search":
        print("\n➡️ 메일 검색 API 호출 (기존 로직)")


async def main():
    """전체 예시 실행"""
    print("\n" + "🚀 BizGuide RAG 통합 - 올바른 아키텍처 예시\n")
    print("계층 구조:")
    print("  API → Service → Agent (RAG, Draft)")
    print("")

    # 1. 메일 초안 작성 (한글)
    await example_draft_email()

    # 2. 메일 번역 (한글 → 영어)
    await example_translate_email()

    # 3. 메일 검색 (기존 기능)
    await example_search_only()

    print("\n" + "=" * 60)
    print("✅ 모든 예시 완료!")
    print("=" * 60)
    print("\n아키텍처 정리:")
    print("  ✅ Agent: 순수 AI 로직만 (DB 접근 X, Agent 간 호출 X)")
    print("  ✅ Service: Agent 조율 + 비즈니스 로직")
    print("  ✅ API: HTTP 요청/응답 (Service 호출)")


if __name__ == "__main__":
    asyncio.run(main())
