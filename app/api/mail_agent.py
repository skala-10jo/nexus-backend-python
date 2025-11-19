"""
Mail Agent API endpoints for email embedding and search.

Author: NEXUS Team
Date: 2025-01-12
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
import logging

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

from app.core.qdrant_client import get_qdrant_client
from app.config import settings
from qdrant_client.http import models
from app.models.email import Email
from app.models.project import Project
from agent.mail.query_agent import QueryAgent
from agent.mail.answer_agent import AnswerAgent
from app.services.email_draft_service import EmailDraftService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai/mail", tags=["Mail AI"])

# Service 인스턴스 (싱글톤)
service = MailAgentService()


@router.post("/embeddings/generate", response_model=GenerateEmbeddingsResponse)
async def generate_embeddings(
    request: GenerateEmbeddingsRequest,
    db: Session = Depends(get_db)
):
    logger.info(f"📧 Generating embeddings for email: {request.email_id}")

    result = await service.generate_embeddings_for_email(request.email_id, db)
    return GenerateEmbeddingsResponse(**result)


@router.post("/embeddings/batch", response_model=BatchGenerateResponse)
async def batch_generate_embeddings(
    request: BatchGenerateRequest,
    db: Session = Depends(get_db)
):
    logger.info(
        f"🚀 Batch generating embeddings for user: {request.user_id} "
        f"(force_regenerate={request.force_regenerate})"
    )

    result = await service.batch_generate_embeddings(
        request.user_id,
        db,
        force_regenerate=request.force_regenerate
    )
    return BatchGenerateResponse(**result)


@router.post("/search", response_model=SearchResponse)
async def search_emails(
    request: SearchRequest,
    db: Session = Depends(get_db)
):
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
            date_to=request.date_to,
            project_name=request.project_name
        )

        search_results = [EmailSearchResult(**r) for r in results]

        return SearchResponse(
            success=True,
            data=search_results,
            count=len(search_results)
        )

    except ValueError as e:
        logger.error(f"Invalid search query: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch("/emails/{email_id}/project")
async def update_email_project(
    email_id: str,
    project_id: Optional[str] = None,    
    db: Session = Depends(get_db)
):
    """
    메일의 프로젝트 할당/해제.
    """

    logger.info(f"Updating project for email: {email_id}, project_id={project_id}")

    try:
        # 1. 이메일 존재 확인
        email = db.query(Email).filter(Email.id == email_id).first()
        if not email:
            raise HTTPException(status_code=404, detail=f"Email {email_id} not found")

        # 2. 프로젝트 정보 조회
        project_name = None
        if project_id:
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
            project_name = project.name

        # 3. Qdrant Payload 업데이트
        qdrant_client = get_qdrant_client()

        payload_update = {
            "project_id": str(project_id) if project_id else None,
            "project_name": project_name
        }

        qdrant_client.set_payload(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            payload=payload_update,
            points=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="email_id",
                            match=models.MatchValue(value=str(email_id))
                        )
                    ]
                )
            )
        )

        logger.info(
            f"Updated Qdrant payload for email {email_id}: "
            f"project_id={project_id}, project_name={project_name}"
        )

        return {
            "success": True,
            "email_id": str(email_id),
            "project_id": str(project_id) if project_id else None,
            "project_name": project_name,
            "message": "프로젝트 정보가 업데이트되었습니다 (임베딩 재생성 없음)"
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Failed to update email project: {str(e)}")
        # (4) 에러 처리 개선
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/chat", response_model=ChatResponse)
async def chat_with_mail_search(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    메일 검색/작성/번역 통합 챗봇 API

    QueryAgent가 쿼리 타입을 분석하고:
    - search: 메일 검색
    - draft: 메일 초안 작성 (RAG 통합)
    - translate: 메일 번역 (RAG 통합)
    - general: 일반 대화
    """
    logger.info(f"Chat request: message='{request.message[:50]}...', user={request.user_id}")

    try:
        # 1. QueryAgent로 쿼리 분석
        query_agent = QueryAgent()

        conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.conversation_history
        ] if request.conversation_history else []

        query_result = await query_agent.process(
            user_message=request.message,
            conversation_history=conversation_history
        )

        logger.info(f"Query extraction result: {query_result}")

        query_type = query_result.get("query_type", "general")
        answer = query_result.get("response", "")

        # 응답 초기화
        response_data = {
            "query_type": query_type,
            "answer": answer
        }

        # 2. query_type별 처리
        if query_type == "search":
            # 메일 검색
            results = await service.search_emails(
                query=query_result.get("query"),
                user_id=request.user_id,
                db=db,
                top_k=5,
                folder=query_result.get("folder"),
                date_from=query_result.get("date_from"),
                date_to=query_result.get("date_to"),
                project_name=query_result.get("project_name")
            )

            search_results = [EmailSearchResult(**r) for r in results]

            # AnswerAgent로 자연어 답변 생성
            answer_agent = AnswerAgent()
            answer = await answer_agent.process(
                user_query=request.message,
                search_results=results,
                conversation_history=conversation_history
            )

            response_data.update({
                "query": query_result.get("query"),
                "folder": query_result.get("folder"),
                "date_from": query_result.get("date_from"),
                "date_to": query_result.get("date_to"),
                "project_name": query_result.get("project_name"),
                "needs_search": True,
                "answer": answer,
                "search_results": search_results
            })

        elif query_type == "draft":
            # 메일 초안 작성 (RAG 통합)
            logger.info(f"Creating email draft with keywords: {query_result.get('keywords')}")

            draft_service = EmailDraftService()
            draft_result = await draft_service.create_draft(
                original_message=query_result.get("original_message", request.message),
                keywords=query_result.get("keywords"),
                target_language=query_result.get("target_language", "ko")
            )

            response_data.update({
                "email_draft": draft_result.get("email_draft"),
                "subject": draft_result.get("subject"),
                "rag_sections": draft_result.get("rag_sections", []),
                "answer": f"{answer}\n\n**제목:** {draft_result.get('subject')}\n\n**본문:**\n{draft_result.get('email_draft')}"
            })

        elif query_type == "translate":
            # 메일 번역 (RAG 통합)
            logger.info(f"Translating email with keywords: {query_result.get('keywords')}")

            draft_service = EmailDraftService()
            translate_result = await draft_service.translate_email(
                email_text=query_result.get("original_message", ""),
                keywords=query_result.get("keywords"),
                target_language=query_result.get("target_language", "en")
            )

            response_data.update({
                "translated_email": translate_result.get("translated_email"),
                "rag_sections": translate_result.get("rag_sections", []),
                "answer": f"{answer}\n\n**번역 결과:**\n{translate_result.get('translated_email')}"
            })

        # 3. general은 기본 answer만 반환

        return ChatResponse(**response_data)

    except ValueError as e:
        logger.error(f"Invalid chat request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")