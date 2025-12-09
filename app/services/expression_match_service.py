"""
Expression Match Service

비즈니스 표현-예문 매칭 서비스
정규식 매칭을 먼저 시도하고, 실패 시 AI Agent로 Fallback

Author: NEXUS Team
Date: 2025-01-18
"""
import re
import logging
from typing import Dict, Any, List, Optional
from agent.expression.expression_match_agent import ExpressionMatchAgent

logger = logging.getLogger(__name__)


class ExpressionMatchService:
    """
    Expression-Sentence 매칭 서비스

    비즈니스 로직:
    1. 정규식 매칭 시도 (빠름, API 비용 없음)
    2. 실패 시 AI Agent 호출 (Fallback)

    Features:
    - 특수문자 이스케이프 처리
    - 변수 패턴 자동 변환 ((someone) → 정규식)
    - AI Fallback으로 복잡한 케이스 처리
    """

    def __init__(self):
        """서비스 초기화"""
        self.agent = ExpressionMatchAgent()
        logger.info("ExpressionMatchService initialized")

    def _escape_special_chars(self, text: str) -> str:
        """
        정규식 특수문자 이스케이프

        Args:
            text: 원본 텍스트

        Returns:
            이스케이프된 텍스트
        """
        # 정규식 특수문자들
        special_chars = r'\.^$*+?{}[]|()'
        result = ""
        for char in text:
            if char in special_chars:
                result += "\\" + char
            else:
                result += char
        return result

    def _build_flexible_pattern(self, expression: str) -> str:
        """
        expression을 유연한 정규식 패턴으로 변환

        변환 규칙:
        - 끝의 ~ → 제거 (해당 부분까지만 매칭)
        - 끝의 ... → 제거 (해당 부분까지만 매칭)
        - (someone) → (\\w+) - 단어 하나
        - (something) → (.+?) - 비탐욕적 매칭
        - 중간의 ~ → (.+?) - 비탐욕적 매칭
        - 중간의 ... → (.+?) - 비탐욕적 매칭
        - one's → (my|your|his|her|their|its|our|one's)

        Args:
            expression: 비즈니스 표현

        Returns:
            정규식 패턴 문자열
        """
        pattern = expression

        # 1. 끝에 있는 ~ 제거 (예: "I am writing to~" → "I am writing to")
        pattern = re.sub(r'\s*~\s*$', '', pattern)

        # 2. 끝에 있는 ... 제거 (예: "Thank you for..." → "Thank you for")
        pattern = re.sub(r'\s*\.\.\.\s*$', '', pattern)

        # 3. 변수 패턴 임시 치환 (특수문자 이스케이프 전에)
        placeholders = {}

        # (someone), (somebody) → 단어 매칭
        pattern = re.sub(
            r'\(someone\)|\(somebody\)',
            '{{PERSON}}',
            pattern,
            flags=re.IGNORECASE
        )
        placeholders['{{PERSON}}'] = r'(\w+)'

        # (something) → 비탐욕적 매칭
        pattern = re.sub(
            r'\(something\)',
            '{{THING}}',
            pattern,
            flags=re.IGNORECASE
        )
        placeholders['{{THING}}'] = r'(.+?)'

        # 4. one's → 소유격 대명사
        pattern = re.sub(
            r"\bone's\b",
            '{{POSSESSIVE}}',
            pattern,
            flags=re.IGNORECASE
        )
        placeholders['{{POSSESSIVE}}'] = r"(my|your|his|her|their|its|our|one's)"

        # 5. 중간에 있는 특수 기호 처리 (비탐욕적 매칭)
        # ~ → 비탐욕적 매칭
        pattern = pattern.replace('~', '{{TILDE}}')
        placeholders['{{TILDE}}'] = r'(.+?)'

        # ... → 비탐욕적 매칭
        pattern = pattern.replace('...', '{{ELLIPSIS}}')
        placeholders['{{ELLIPSIS}}'] = r'(.+?)'

        # 6. 나머지 특수문자 이스케이프
        for placeholder in placeholders.keys():
            pattern = pattern.replace(placeholder, f'<<<{placeholder}>>>')

        pattern = self._escape_special_chars(pattern)

        for placeholder in placeholders.keys():
            pattern = pattern.replace(f'<<<{placeholder}>>>', placeholders[placeholder])

        return pattern

    def try_regex_match(
        self,
        expression: str,
        sentence: str
    ) -> Optional[Dict[str, Any]]:
        """
        정규식으로 매칭 시도

        Args:
            expression: 비즈니스 표현
            sentence: 예문

        Returns:
            매칭 결과 dict 또는 None (실패 시)
        """
        try:
            pattern = self._build_flexible_pattern(expression)
            logger.debug(f"Regex pattern: {pattern}")

            match = re.search(pattern, sentence, re.IGNORECASE)

            if match:
                matched_text = match.group(0)
                return {
                    "matched": True,
                    "start_index": match.start(),
                    "end_index": match.end(),
                    "matched_text": matched_text,
                    "method": "regex"
                }

            return None

        except re.error as e:
            logger.warning(f"Regex error: {str(e)}")
            return None

    async def find_match(
        self,
        expression: str,
        sentence: str,
        use_ai_fallback: bool = True
    ) -> Dict[str, Any]:
        """
        expression이 sentence 어디에 해당하는지 찾기

        1. 정규식 매칭 시도
        2. 실패 시 AI Agent 호출 (use_ai_fallback=True인 경우)

        Args:
            expression: 비즈니스 표현 (예: "take (someone) through")
            sentence: 예문 (예: "Can you take me through the budget?")
            use_ai_fallback: AI Fallback 사용 여부 (기본값: True)

        Returns:
            Dict[str, Any]: {
                "matched": bool,
                "start_index": int,
                "end_index": int,
                "matched_text": str,
                "method": "regex" | "ai"
            }
        """
        logger.info(f"Finding match: '{expression}' in '{sentence[:50]}...'")

        # 1. 정규식 매칭 시도
        regex_result = self.try_regex_match(expression, sentence)

        if regex_result:
            logger.info(f"Regex match found: '{regex_result['matched_text']}'")
            return regex_result

        logger.info("Regex match failed, trying AI fallback...")

        # 2. AI Fallback
        if use_ai_fallback:
            ai_result = await self.agent.process(expression, sentence)
            ai_result["method"] = "ai"
            return ai_result

        # Fallback 비활성화 시
        return {
            "matched": False,
            "start_index": 0,
            "end_index": 0,
            "matched_text": "",
            "method": "none"
        }

    async def find_matches_batch(
        self,
        items: List[Dict[str, str]],
        use_ai_fallback: bool = True
    ) -> List[Dict[str, Any]]:
        """
        여러 expression-sentence 쌍에 대해 배치 매칭

        Args:
            items: [{"expression": "...", "sentence": "..."}] 형태의 리스트
            use_ai_fallback: AI Fallback 사용 여부

        Returns:
            각 item에 대한 매칭 결과 리스트
        """
        results = []

        for item in items:
            expression = item.get("expression", "")
            sentence = item.get("sentence", "")

            if not expression or not sentence:
                results.append({
                    "matched": False,
                    "start_index": 0,
                    "end_index": 0,
                    "matched_text": "",
                    "method": "none",
                    "error": "Missing expression or sentence"
                })
                continue

            result = await self.find_match(expression, sentence, use_ai_fallback)
            results.append(result)

        return results


# 편의 함수
def get_expression_match_service() -> ExpressionMatchService:
    """
    ExpressionMatchService 인스턴스 반환

    Returns:
        ExpressionMatchService 인스턴스
    """
    return ExpressionMatchService()
