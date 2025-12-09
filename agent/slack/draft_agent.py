"""
SlackDraftAgent: 비즈니스 슬랙 메시지 초안 작성 Agent (순수 AI 로직)

RAG 컨텍스트를 활용하여 비즈니스 매너가 적용된 슬랙 메시지를 작성합니다.
EmailDraftAgent와 동일한 구조를 유지하되, 수신자(recipient)와 제목(subject)만 제외합니다.

Author: NEXUS Team
Date: 2025-01-18
"""
from agent.base_agent import BaseAgent
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class SlackDraftAgent(BaseAgent):
    """
    슬랙 메시지 초안 작성 Agent (순수 AI 로직만)

    RAG 컨텍스트를 받아서 GPT로 슬랙 메시지를 작성합니다.
    EmailDraftAgent와 동일한 BizGuide 활용 방식을 유지합니다.

    EmailDraftAgent와의 차이점 (최소한의 변경):
    - 수신자(recipient) 파라미터 없음: 슬랙은 채널/DM으로 수신자가 이미 특정됨
    - 제목(subject) 없음: 슬랙 메시지는 제목이 없음

    Example:
        >>> agent = SlackDraftAgent()
        >>> result = await agent.process(
        ...     original_message="내일 미팅 9시 30분으로 변경 요청",
        ...     rag_context=["Thank you for your flexibility..."],
        ...     target_language="ko"
        ... )
        >>> print(result["draft"])
    """

    async def process(
        self,
        original_message: str,
        rag_context: Optional[List[str]] = None,
        target_language: str = "ko",
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        슬랙 메시지 초안을 작성합니다 (순수 AI 로직).

        Args:
            original_message: 사용자의 의도/내용 설명 (EmailDraftAgent와 동일)
            rag_context: BizGuide RAG 검색 결과 (Service에서 전달)
            target_language: "ko" (한글) 또는 "en" (영어)
            conversation_history: 이전 대화 내역 (연속 대화 지원)

        Returns:
            {
                "draft": "작성된 슬랙 메시지 초안",
                "status": "success"
            }

        Raises:
            ValueError: original_message가 비어있을 때
        """
        if not original_message or len(original_message.strip()) < 1:
            raise ValueError("original_message is required")

        try:
            # RAG 컨텍스트 구성 (EmailDraftAgent와 동일)
            rag_text = ""
            if rag_context and len(rag_context) > 0:
                rag_text = "\n".join(rag_context)
                logger.info(f"Using {len(rag_context)} RAG contexts")
            else:
                logger.warning("No RAG context provided")

            # GPT 프롬프트 구성 (EmailDraftAgent와 동일한 구조)
            language_instruction = {
                "ko": "한글로 작성하세요. 비즈니스 매너를 지키되, 자연스러운 한국어 표현을 사용하세요.",
                "en": "Write in English with proper business etiquette and professional tone."
            }

            # EmailDraftAgent 프롬프트를 기반으로, recipient/subject만 제거
            system_prompt = f"""당신은 비즈니스 메시지 작성 전문가입니다.

사용자의 의도를 파악하여 격식 있고 공손한 비즈니스 메시지를 작성하세요.

언어: {target_language}
{language_instruction.get(target_language, language_instruction["ko"])}

메시지 구성:
1. 인사말 (적절한 격식으로)
2. 본문 (명확하고 간결하게)
3. 마무리 인사

참고할 비즈니스 표현 (영어 표현의 톤과 구조를 활용):
{rag_text if rag_text else "없음"}

주의사항:
- 슬랙 메시지이므로 "OO님께" 같은 수신자 호칭은 사용하지 마세요 (채널/DM으로 이미 특정됨)
- 너무 길지 않게 (10줄 이내)
- 존댓말 사용 (한글인 경우)
"""

            user_prompt = f"""다음 상황을 비즈니스 메시지로 작성해주세요:

{original_message}

메시지 본문만 작성하세요 (제목, 수신자 호칭 없이):
"""

            # 대화 히스토리 구성
            messages = [{"role": "system", "content": system_prompt}]

            if conversation_history:
                messages.extend(conversation_history)

            messages.append({"role": "user", "content": user_prompt})

            # GPT 호출
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )

            draft_text = response.choices[0].message.content.strip()

            # 불필요한 형식 제거 (GPT가 가끔 추가하는 경우)
            draft_text = self._clean_slack_draft(draft_text)

            logger.info("Slack draft created successfully")

            return {
                "draft": draft_text,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Failed to create Slack draft: {str(e)}")
            return {
                "draft": "",
                "status": "error",
                "error_message": str(e)
            }

    def _clean_slack_draft(self, text: str) -> str:
        """
        GPT 응답에서 불필요한 형식을 제거합니다.

        Args:
            text: 원본 GPT 응답

        Returns:
            정리된 슬랙 메시지
        """
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            # "제목:", "Subject:", "---" 등 제거
            if line.strip().startswith(("제목:", "Subject:", "---", "===", "슬랙 메시지:", "메시지:")):
                continue
            # 빈 줄이 연속되지 않도록
            if line.strip() == "" and cleaned_lines and cleaned_lines[-1].strip() == "":
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    async def translate(
        self,
        message_text: str,
        rag_context: Optional[List[str]] = None,
        target_language: str = "en",
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        메시지를 번역합니다 (EmailDraftAgent.translate와 동일한 구조).

        Args:
            message_text: 번역할 메시지 텍스트
            rag_context: BizGuide RAG 검색 결과 (Service에서 전달)
            target_language: 목표 언어 ("ko" or "en")
            conversation_history: 이전 대화 내역

        Returns:
            {
                "translated_message": "번역된 메시지",
                "status": "success"
            }
        """
        try:
            # RAG 컨텍스트 구성
            rag_text = ""
            if rag_context and len(rag_context) > 0:
                rag_text = "\n".join(rag_context)

            # GPT 번역 (EmailDraftAgent와 동일한 구조)
            system_prompt = f"""당신은 비즈니스 메시지 번역 전문가입니다.

원본 메시지를 {target_language}로 번역하되, 비즈니스 매너와 격식을 유지하세요.

참고할 비즈니스 표현:
{rag_text if rag_text else "없음"}

번역 시 주의사항:
- 원문의 의도와 톤을 정확히 전달
- 비즈니스 문화에 맞는 표현 사용
- 자연스러운 표현 사용
"""

            messages = [{"role": "system", "content": system_prompt}]

            if conversation_history:
                messages.extend(conversation_history)

            messages.append({"role": "user", "content": f"다음 메시지를 {target_language}로 번역해주세요:\n\n{message_text}"})

            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.5,
                max_tokens=500
            )

            translated_text = response.choices[0].message.content

            return {
                "translated_message": translated_text,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Failed to translate message: {str(e)}")
            return {
                "translated_message": "",
                "status": "error",
                "error_message": str(e)
            }
