"""
EmailDraftService: 메일 초안 작성/번역 비즈니스 로직

Agent들을 조율하여 BizGuide RAG 기반 메일 작성/번역을 제공합니다.

Author: NEXUS Team
Date: 2025-01-18
"""
from agent.mail.draft_agent import EmailDraftAgent
from agent.rag.bizguide_rag_agent import BizGuideRAGAgent
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class EmailDraftService:
    """
    메일 초안 작성/번역 Service

    Agent 조율 계층:
    - BizGuideRAGAgent: RAG 검색
    - EmailDraftAgent: 메일 작성/번역

    Example:
        >>> service = EmailDraftService()
        >>> result = await service.create_draft(
        ...     original_message="내일 미팅 9시 30분 팀장님께 알려야해",
        ...     keywords=["meeting", "scheduling"],
        ...     target_language="ko"
        ... )
    """

    def __init__(self):
        self.rag_agent = BizGuideRAGAgent()
        self.draft_agent = EmailDraftAgent()

    async def create_draft(
        self,
        original_message: str,
        keywords: Optional[List[str]] = None,
        target_language: str = "ko",
        recipient: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        메일 초안을 작성합니다 (RAG 통합).

        플로우:
        1. BizGuideRAGAgent로 관련 비즈니스 표현 검색
        2. EmailDraftAgent로 메일 작성

        Args:
            original_message: 사용자의 의도/내용 설명
            keywords: 주제 키워드 (RAG 필터링용)
            target_language: "ko" (한글) 또는 "en" (영어)
            recipient: 수신자 정보 (선택)

        Returns:
            {
                "email_draft": "작성된 메일 초안",
                "subject": "메일 제목",
                "rag_sections": ["사용된 BizGuide 섹션들"],
                "status": "success"
            }
        """
        try:
            logger.info(f"Creating email draft: lang={target_language}, keywords={keywords}")

            # 1. BizGuide RAG 검색
            rag_results = await self.rag_agent.search_for_email(
                original_message=original_message,
                keywords=keywords,
                target_language=target_language
            )

            # 2. RAG 컨텍스트 추출
            rag_context = []
            rag_sections = []

            if rag_results:
                for result in rag_results:
                    # 컨텍스트: chunk 전체 텍스트
                    rag_context.append(
                        f"[{result['section']}]\n{result['text']}"
                    )
                    # 섹션명 저장 (사용자에게 표시)
                    rag_sections.append(result['section'])

                logger.info(f"Found {len(rag_results)} relevant BizGuide sections")
            else:
                logger.warning("No RAG results, proceeding without BizGuide context")

            # 3. EmailDraftAgent로 메일 작성
            draft_result = await self.draft_agent.process(
                original_message=original_message,
                rag_context=rag_context,
                target_language=target_language,
                recipient=recipient
            )

            # 4. 응답 구성
            return {
                "email_draft": draft_result.get("email_draft", ""),
                "subject": draft_result.get("subject", ""),
                "rag_sections": rag_sections,
                "status": draft_result.get("status", "success")
            }

        except Exception as e:
            logger.error(f"Failed to create email draft: {str(e)}")
            return {
                "email_draft": "",
                "subject": "",
                "rag_sections": [],
                "status": "error",
                "error_message": str(e)
            }

    async def translate_email(
        self,
        email_text: str,
        keywords: Optional[List[str]] = None,
        target_language: str = "en"
    ) -> Dict[str, Any]:
        """
        메일을 번역합니다 (RAG 통합).

        플로우:
        1. BizGuideRAGAgent로 관련 비즈니스 표현 검색
        2. EmailDraftAgent로 번역

        Args:
            email_text: 번역할 메일 텍스트
            keywords: 주제 키워드 (RAG 필터링용)
            target_language: 목표 언어 ("ko" or "en")

        Returns:
            {
                "translated_email": "번역된 메일",
                "rag_sections": ["사용된 BizGuide 섹션들"],
                "status": "success"
            }
        """
        try:
            logger.info(f"Translating email: lang={target_language}, keywords={keywords}")

            # 1. BizGuide RAG 검색
            rag_results = await self.rag_agent.search_for_email(
                original_message=email_text,
                keywords=keywords,
                target_language=target_language
            )

            # 2. RAG 컨텍스트 추출
            rag_context = []
            rag_sections = []

            if rag_results:
                for result in rag_results:
                    # chunk 전체 텍스트
                    rag_context.append(
                        f"[{result['section']}]\n{result['text']}"
                    )
                    rag_sections.append(result['section'])

            # 3. EmailDraftAgent로 번역
            translate_result = await self.draft_agent.translate(
                email_text=email_text,
                rag_context=rag_context,
                target_language=target_language
            )

            # 4. 응답 구성
            return {
                "translated_email": translate_result.get("translated_email", ""),
                "rag_sections": rag_sections,
                "status": translate_result.get("status", "success")
            }

        except Exception as e:
            logger.error(f"Failed to translate email: {str(e)}")
            return {
                "translated_email": "",
                "rag_sections": [],
                "status": "error",
                "error_message": str(e)
            }
