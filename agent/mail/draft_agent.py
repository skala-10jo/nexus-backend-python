"""
EmailDraftAgent: 비즈니스 메일 초안 작성 Agent (순수 AI 로직)

RAG 컨텍스트를 활용하여 비즈니스 매너가 적용된 한글/영어 메일을 작성합니다.

Author: NEXUS Team
Date: 2025-01-18
"""
from agent.base_agent import BaseAgent
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class EmailDraftAgent(BaseAgent):
    """
    메일 초안 작성 Agent (순수 AI 로직만)

    RAG 컨텍스트를 받아서 GPT로 메일을 작성합니다.
    Service 계층에서 RAG 검색 후 결과를 전달받아야 합니다.

    Example:
        >>> agent = EmailDraftAgent()
        >>> result = await agent.process(
        ...     original_message="내일 미팅 9시 30분",
        ...     rag_context=["Thank you for arranging..."],
        ...     target_language="ko"
        ... )
        >>> print(result["email_draft"])
    """

    async def process(
        self,
        original_message: str,
        rag_context: Optional[List[str]] = None,
        target_language: str = "ko",
        recipient: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        메일 초안을 작성합니다 (순수 AI 로직).

        Args:
            original_message: 사용자의 의도/내용 설명
            rag_context: BizGuide RAG 검색 결과 (Service에서 전달)
            target_language: "ko" (한글) 또는 "en" (영어)
            recipient: 수신자 정보 (선택)

        Returns:
            {
                "email_draft": "작성된 메일 초안",
                "subject": "메일 제목",
                "status": "success"
            }

        Raises:
            ValueError: original_message가 비어있을 때
        """
        if not original_message or len(original_message.strip()) < 1:
            raise ValueError("original_message is required")

        try:
            # RAG 컨텍스트 구성
            rag_text = ""
            if rag_context and len(rag_context) > 0:
                rag_text = "\n".join(rag_context)
                logger.info(f"Using {len(rag_context)} RAG contexts")
            else:
                logger.warning("No RAG context provided")

            # GPT 프롬프트 구성
            language_instruction = {
                "ko": "한글로 작성하세요. 비즈니스 매너를 지키되, 자연스러운 한국어 표현을 사용하세요.",
                "en": "Write in English with proper business etiquette and professional tone."
            }

            system_prompt = f"""당신은 비즈니스 메일 작성 전문가입니다.

사용자의 의도를 파악하여 격식 있고 공손한 비즈니스 메일을 작성하세요.

언어: {target_language}
{language_instruction.get(target_language, language_instruction["ko"])}

메일 구성:
1. 인사말 (적절한 격식으로)
2. 본문 (명확하고 간결하게)
3. 마무리 인사

참고할 비즈니스 표현 (영어 표현의 톤과 구조를 활용):
{rag_text if rag_text else "없음"}

주의사항:
- 받는 사람: {recipient if recipient else "팀장님/상사"}
- 너무 길지 않게 (10줄 이내)
- 존댓말 사용 (한글인 경우)
- 제목도 함께 제안
"""

            user_prompt = f"""다음 상황을 비즈니스 메일로 작성해주세요:

{original_message}

응답 형식:
제목: [메일 제목]

[메일 본문]
"""

            # GPT 호출
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            draft_text = response.choices[0].message.content

            # 제목과 본문 분리
            subject = ""
            email_body = draft_text

            if "제목:" in draft_text or "Subject:" in draft_text:
                lines = draft_text.split("\n", 2)
                if len(lines) >= 2:
                    subject = lines[0].replace("제목:", "").replace("Subject:", "").strip()
                    email_body = lines[2].strip() if len(lines) > 2 else lines[1].strip()

            logger.info("Email draft created successfully")

            return {
                "email_draft": email_body,
                "subject": subject,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Failed to create email draft: {str(e)}")
            return {
                "email_draft": "",
                "subject": "",
                "status": "error",
                "error_message": str(e)
            }

    async def translate(
        self,
        email_text: str,
        rag_context: Optional[List[str]] = None,
        target_language: str = "en"
    ) -> Dict[str, Any]:
        """
        메일을 번역합니다 (순수 AI 로직).

        Args:
            email_text: 번역할 메일 텍스트
            rag_context: BizGuide RAG 검색 결과 (Service에서 전달)
            target_language: 목표 언어 ("ko" or "en")

        Returns:
            {
                "translated_email": "번역된 메일",
                "status": "success"
            }
        """
        try:
            # RAG 컨텍스트 구성
            rag_text = ""
            if rag_context and len(rag_context) > 0:
                rag_text = "\n".join(rag_context)

            # GPT 번역
            system_prompt = f"""당신은 비즈니스 메일 번역 전문가입니다.

원본 메일을 {target_language}로 번역하되, 비즈니스 매너와 격식을 유지하세요.

참고할 비즈니스 표현:
{rag_text if rag_text else "없음"}

번역 시 주의사항:
- 원문의 의도와 톤을 정확히 전달
- 비즈니스 문화에 맞는 표현 사용
- 자연스러운 표현 사용
"""

            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"다음 메일을 {target_language}로 번역해주세요:\n\n{email_text}"}
                ],
                temperature=0.5,
                max_tokens=500
            )

            translated_text = response.choices[0].message.content

            return {
                "translated_email": translated_text,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Failed to translate email: {str(e)}")
            return {
                "translated_email": "",
                "status": "error",
                "error_message": str(e)
            }
