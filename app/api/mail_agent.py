"""
Mail Agent API endpoints for email embedding and search.

Author: NEXUS Team
Date: 2025-01-12
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.mail_agent_service import MailAgentService
from app.schemas.mail_agent import (
    GenerateEmbeddingsRequest,
    GenerateEmbeddingsResponse,
    BatchGenerateRequest,
    BatchGenerateResponse,
    SearchRequest,
    SearchResponse,
    EmailSearchResult,
    ChatRequest,
    ChatResponse
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai/mail", tags=["Mail AI"])

# Service 인스턴스 (싱글톤)
service = MailAgentService()


@router.post("/embeddings/generate", response_model=GenerateEmbeddingsResponse)
async def generate_embeddings(
    request: GenerateEmbeddingsRequest,
    db: Session = Depends(get_db)
):
    """
    단일 메일 임베딩 생성.

    메일 본문을 청킹하여 각 청크를 OpenAI text-embedding-ada-002로 임베딩합니다.

    Args:
        email_id: 임베딩 생성할 메일 ID

    Returns:
        임베딩 생성 결과 {status, chunks_created}

    Example:
        Request:
            POST /api/ai/mail/embeddings/generate
            {"email_id": "uuid"}

        Response:
            {"status": "success", "chunks_created": 3}
    """
    logger.info(f"📧 Generating embeddings for email: {request.email_id}")

    result = await service.generate_embeddings_for_email(request.email_id, db)
    return GenerateEmbeddingsResponse(**result)


@router.post("/embeddings/batch", response_model=BatchGenerateResponse)
async def batch_generate_embeddings(
    request: BatchGenerateRequest,
    db: Session = Depends(get_db)
):
    """
    사용자의 모든 메일 임베딩 일괄 생성.

    임베딩이 없는 메일들만 자동으로 선택하여 처리합니다.

    Args:
        user_id: 사용자 ID

    Returns:
        일괄 생성 결과 {status, total, processed, skipped, failed}

    Example:
        Request:
            POST /api/ai/mail/embeddings/batch
            {"user_id": "uuid"}

        Response:
            {
                "status": "success",
                "total": 100,
                "processed": 95,
                "skipped": 3,
                "failed": 2
            }

    Notes:
        - 시간이 오래 걸릴 수 있으므로 타임아웃 주의
        - 향후 백그라운드 작업으로 전환 고려
    """
    logger.info(f"🚀 Batch generating embeddings for user: {request.user_id}")

    result = await service.batch_generate_embeddings(request.user_id, db)
    return BatchGenerateResponse(**result)


@router.post("/search", response_model=SearchResponse)
async def search_emails(
    request: SearchRequest,
    db: Session = Depends(get_db)
):
    """
    자연어로 메일 검색 (RAG + SQL 필터).

    하이브리드 검색 전략:
        1. Qdrant 필터로 범위 축소 (user_id, folder, date)
        2. Qdrant로 의미 기반 검색 (벡터 유사도)
        3. 유사도 높은 순으로 정렬

    Args:
        query: 검색 쿼리 (예: "프로젝트 일정 관련 메일")
        user_id: 사용자 ID
        top_k: 최대 결과 개수 (1-50, 기본 10)
        folder: 폴더 필터 (선택, 'Inbox' or 'SentItems')
        date_from/date_to: 날짜 범위 필터 (선택, 'YYYY-MM-DD')

    Returns:
        검색 결과 목록 (유사도 높은 순)

    Example:
        Request:
            POST /api/ai/mail/search
            {
                "query": "프로젝트 일정 회의",
                "user_id": "uuid",
                "top_k": 10,
                "folder": "Inbox",
                "date_from": "2025-01-01"
            }

        Response:
            {
                "success": true,
                "data": [
                    {
                        "email_id": "uuid",
                        "subject": "프로젝트 일정 회의 요청",
                        "from_name": "홍길동",
                        "similarity": 0.92,
                        "matched_chunk": "제목: 프로젝트 일정 회의 요청...",
                        ...
                    }
                ],
                "count": 5
            }
    """
    logger.info(
        f"🔍 Searching emails: query='{request.query[:50]}...', "
        f"user={request.user_id}, folder={request.folder}"
    )

    try:
        results = await service.search_emails(
            query=request.query,
            user_id=request.user_id,
            db=db,
            top_k=request.top_k,
            folder=request.folder,
            date_from=request.date_from,
            date_to=request.date_to
        )

        # Dict를 Pydantic 모델로 변환
        search_results = [EmailSearchResult(**r) for r in results]

        return SearchResponse(
            success=True,
            data=search_results,
            count=len(search_results)
        )

    except ValueError as e:
        logger.error(f"❌ Invalid search query: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/chat", response_model=ChatResponse)
async def chat_with_mail_search(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    대화형 메일 검색 (챗봇 인터페이스).

    사용자의 자연어 메시지를 분석하여 검색 쿼리를 추출하고,
    필요한 경우 자동으로 메일을 검색합니다.

    Args:
        message: 사용자 메시지 (자연어)
        user_id: 사용자 ID
        conversation_history: 대화 히스토리 (선택)

    Returns:
        {
            "query": 추출된 검색 쿼리,
            "folder": 폴더 필터,
            "date_from": 시작 날짜,
            "date_to": 종료 날짜,
            "needs_search": 검색 수행 여부,
            "response": 사용자 응답 메시지,
            "search_results": 검색 결과 (검색 수행 시)
        }

    Example:
        Request:
            POST /api/ai/mail/chat
            {
                "message": "어제 받은 프로젝트 관련 메일 찾아줘",
                "user_id": "uuid",
                "conversation_history": []
            }

        Response:
            {
                "query": "프로젝트",
                "folder": "Inbox",
                "date_from": "2025-01-16",
                "needs_search": true,
                "response": "어제 받은 프로젝트 관련 메일을 검색하겠습니다.",
                "search_results": [...]
            }
    """
    logger.info(f"Chat request: message='{request.message[:50]}...', user={request.user_id}")

    try:
        # QueryAgent로 쿼리 추출
        from agent.mail.query_agent import QueryAgent

        query_agent = QueryAgent()

        # 대화 히스토리를 딕셔너리 형태로 변환
        conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.conversation_history
        ] if request.conversation_history else []

        query_result = await query_agent.process(
            user_message=request.message,
            conversation_history=conversation_history
        )

        logger.info(f"Query extraction result: {query_result}")

        # 검색이 필요한 경우 자동으로 검색 수행
        search_results = None
        if query_result.get("needs_search") and query_result.get("query"):
            logger.info(f"Performing search with query: {query_result.get('query')}")

            results = await service.search_emails(
                query=query_result.get("query"),
                user_id=request.user_id,
                db=db,
                top_k=5,  # 챗봇은 최대 5개만 표시
                folder=query_result.get("folder"),
                date_from=query_result.get("date_from"),
                date_to=query_result.get("date_to")
            )

            search_results = [EmailSearchResult(**r) for r in results]
            logger.info(f"Search completed: {len(search_results)} results")

        return ChatResponse(
            query=query_result.get("query"),
            folder=query_result.get("folder"),
            date_from=query_result.get("date_from"),
            date_to=query_result.get("date_to"),
            needs_search=query_result.get("needs_search", False),
            response=query_result.get("response", ""),
            search_results=search_results
        )

    except ValueError as e:
        logger.error(f"Invalid chat request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
