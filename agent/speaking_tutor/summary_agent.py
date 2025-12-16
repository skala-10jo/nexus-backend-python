"""
회의 요약 Agent (Meeting Summary Agent)

회의/대화 내용을 간결하게 요약하는 Micro Agent.
단일 책임: 발화 목록 -> 간결한 요약 (카드 표시용)
"""

import logging
from typing import List, Dict
from agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class MeetingSummaryAgent(BaseAgent):
    """
    회의 내용 요약 Agent.

    책임: 발화 목록 -> 간결한 요약 (80자 이내)

    회의나 대화의 발화 목록을 분석하여 핵심 주제를 1-2문장으로 요약합니다.
    UI 카드 표시용으로 간결한 요약을 생성합니다.

    예시:
        >>> agent = MeetingSummaryAgent()
        >>> utterances = [
        ...     {"speaker_id": 1, "text": "프로젝트 일정을 논의해봅시다"},
        ...     {"speaker_id": 2, "text": "네, 다음 주까지 완료해야 합니다"}
        ... ]
        >>> summary = await agent.process(utterances)
        >>> print(summary)
        "프로젝트 일정 조정에 대해 논의함"
    """

    # 요약에 사용할 최대 발화 수
    MAX_UTTERANCES = 30
    # 최대 요약 길이
    MAX_SUMMARY_LENGTH = 100

    async def process(
        self,
        utterances: List[Dict],
        language: str = "ko"
    ) -> str:
        """
        발화 목록을 간결하게 요약합니다.

        Args:
            utterances: 발화 딕셔너리 리스트
                각 딕셔너리는 'speaker_id'와 'text' 키를 포함
            language: 언어 코드 (기본값: "ko")
                요약은 항상 한국어로 생성됨

        Returns:
            간결한 요약 문자열 (80자 이내)

        Raises:
            ValueError: 발화가 비어있는 경우
        """
        if not utterances:
            return "내용 없음"

        # 발화를 대화 텍스트로 변환 (최대 30개)
        conversation_lines = []
        for utt in utterances[:self.MAX_UTTERANCES]:
            speaker = f"화자{utt.get('speaker_id', 1)}"
            text = utt.get('text', '')
            if text.strip():
                conversation_lines.append(f"{speaker}: {text}")

        if not conversation_lines:
            return "대화 내용 없음"

        conversation_text = "\n".join(conversation_lines)

        logger.info(f"Meeting summary generation: {len(conversation_lines)} utterances")

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": f"다음 대화 내용의 핵심 주제를 한국어로 간결하게 요약해주세요:\n\n{conversation_text}"}
                ],
                max_tokens=100,
                temperature=0.3
            )

            summary = response.choices[0].message.content.strip()
            summary = self._clean_summary(summary)

            logger.info(f"Summary generated: {summary[:50]}...")
            return summary

        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            # Fallback: 첫 발화 사용
            if conversation_lines:
                first_text = utterances[0].get("text", "")
                return first_text[:self.MAX_SUMMARY_LENGTH] + "..." if len(first_text) > self.MAX_SUMMARY_LENGTH else first_text
            return "요약 생성 실패"

    def _get_system_prompt(self) -> str:
        """시스템 프롬프트 반환"""
        return """당신은 회의/대화 내용을 간결하게 요약하는 전문가입니다.
규칙:
1. 반드시 한국어로 요약하세요
2. 1-2문장, 80자 이내로 작성하세요
3. 핵심 주제와 논의 내용을 포함하세요
4. "~에 대해 논의함", "~를 다룸" 형식으로 작성하세요

예시:
- "프로젝트 일정 조정과 팀 역할 분담에 대해 논의함"
- "신제품 마케팅 전략과 예산 배분을 다룸"
- "고객 피드백 분석 및 개선 방안을 검토함" """

    def _clean_summary(self, summary: str) -> str:
        """요약 텍스트 정리"""
        # 따옴표 제거
        if summary.startswith('"') and summary.endswith('"'):
            summary = summary[1:-1]
        if summary.startswith("'") and summary.endswith("'"):
            summary = summary[1:-1]

        # 길이 제한
        if len(summary) > self.MAX_SUMMARY_LENGTH:
            summary = summary[:self.MAX_SUMMARY_LENGTH - 3] + "..."

        return summary
