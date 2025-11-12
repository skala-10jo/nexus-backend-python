"""
Glossary extraction agent using GPT-4o.
Extracts technical terms from text with definitions, context, and confidence scores.
"""
import json
import math
import logging
from typing import List, Dict, Any
from agent.base_agent import BaseAgent
from app.core.text_utils import split_text_into_chunks, deduplicate_terms

logger = logging.getLogger(__name__)


class GlossaryAgent(BaseAgent):
    """
    AI agent for extracting technical glossary terms from documents.

    Uses GPT-4o to analyze text and extract IT/project management related terms
    with Korean/English translations, definitions, context, and confidence scores.

    Example:
        >>> agent = GlossaryAgent()
        >>> text = "클라우드 컴퓨팅은 인터넷을 통해 IT 리소스를 제공하는 서비스입니다..."
        >>> terms = await agent.process(text, max_terms=50)
        >>> print(f"Extracted {len(terms)} terms")
        >>> print(terms[0])
        {
            "korean": "클라우드 컴퓨팅",
            "english": "Cloud Computing",
            "abbreviation": "CC",
            "definition": "인터넷을 통해 IT 리소스를 제공하는 서비스",
            "context": "클라우드 컴퓨팅은 인터넷을 통해...",
            "domain": "IT",
            "confidence": 0.95
        }
    """

    def __init__(self):
        """Initialize GlossaryAgent with system prompt."""
        super().__init__()
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """
        Create system prompt for GPT-4o.

        Returns:
            System prompt string
        """
        return """
당신은 전문 용어 추출 및 다국어 번역 전문가입니다.
주어진 텍스트에서 IT 및 프로젝트 관리 관련 전문 용어를 추출하고, 한국어/영어/베트남어로 번역하며 분석합니다.

**출력 형식 (JSON)**:
{
  "terms": [
    {
      "korean": "한글 용어",
      "english": "English term",
      "vietnamese": "Thuật ngữ tiếng Việt",
      "abbreviation": "약어 (있는 경우만, 없으면 null)",
      "definition": "용어의 명확한 정의 (1-2문장)",
      "context": "문서 내에서 사용된 구체적인 맥락 (원문 인용)",
      "example_sentence": "용어 사용 예문 (문서에서 발췌하거나 생성)",
      "note": "추가 설명 및 참고사항 (있는 경우만, 없으면 null)",
      "domain": "분야 (IT, Project Management, Business, Development 등)",
      "confidence": 0.95
    }
  ]
}

**추출 기준**:
1. IT, 프로젝트 관리, 비즈니스 관련 전문 용어만 추출
2. 일반적인 단어는 제외 (예: "컴퓨터" 제외, "클라우드 컴퓨팅" 포함)
3. 약어는 문서에서 명확히 정의된 경우만 포함
4. 정의는 명확하고 간결하게 (1-2문장)
5. 맥락은 실제 문서에서 사용된 문장을 그대로 인용
6. 예문은 문서에서 발췌하거나, 자연스러운 예문 생성
7. 추가 설명은 용어 이해에 도움이 되는 참고사항 제공
8. 베트남어 번역은 정확하고 자연스럽게 (Tiếng Việt)
9. 신뢰도는 해당 용어의 전문성 정도 (0.0-1.0)
10. 도메인은 가장 적합한 분야 하나만 선택

**주의사항**:
- 중복된 용어는 하나만 포함
- 유사 용어는 별도로 추출 (예: "데이터베이스"와 "관계형 데이터베이스"는 별개)
- 외래어는 한글/영어/베트남어를 모두 포함
- 베트남어 번역이 없는 경우 영어를 베트남어로 번역
- 반드시 JSON 형식으로만 응답
"""

    async def process(
        self,
        text: str,
        max_terms: int = 50,
        chunk_size: int = 5000
    ) -> List[Dict[str, Any]]:
        """
        Extract glossary terms from text.

        Process:
        1. Split text into chunks
        2. Extract terms from each chunk using GPT-4o
        3. Deduplicate terms across chunks
        4. Sort by confidence score
        5. Return top N terms

        Args:
            text: Full text to extract terms from
            max_terms: Maximum number of terms to return (default: 50)
            chunk_size: Size of text chunks in characters (default: 5000)

        Returns:
            List of extracted terms, each with:
                - korean: Korean term
                - english: English term (optional)
                - abbreviation: Abbreviation (optional)
                - definition: Term definition
                - context: Usage context from document
                - domain: Domain category (IT, Project Management, etc.)
                - confidence: Confidence score (0.0-1.0)

        Raises:
            Exception: If extraction fails

        Example:
            >>> agent = GlossaryAgent()
            >>> text = load_document("document.pdf")
            >>> terms = await agent.process(text, max_terms=30)
            >>> for term in terms[:5]:
            ...     print(f"{term['korean']}: {term['definition']}")
        """
        logger.info(f"🚀 Starting term extraction (text length: {len(text)} chars, max terms: {max_terms})")

        # 1. Split text into chunks
        chunks = split_text_into_chunks(text, chunk_size)

        # 2. Calculate terms per chunk
        terms_per_chunk = math.ceil(max_terms / len(chunks)) + 5  # +5 buffer for deduplication

        # 3. Extract terms from each chunk
        all_terms = []
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"📝 Processing chunk {i}/{len(chunks)}")
            try:
                terms = await self._extract_chunk(chunk, terms_per_chunk)
                all_terms.extend(terms)
            except Exception as e:
                logger.warning(f"Failed to extract from chunk {i}: {str(e)}")
                continue

        if not all_terms:
            logger.warning("No terms extracted from any chunk")
            return []

        logger.info(f"📊 Total terms extracted: {len(all_terms)}")

        # 4. Deduplicate terms
        unique_terms = deduplicate_terms(all_terms)

        # 5. Sort by confidence (descending)
        unique_terms.sort(key=lambda x: float(x.get('confidence', 0)), reverse=True)

        # 6. Return top N terms
        result_terms = unique_terms[:max_terms]

        avg_confidence = sum(float(t.get('confidence', 0)) for t in result_terms) / len(result_terms)
        logger.info(f"✅ Term extraction complete: {len(result_terms)} terms (avg confidence: {avg_confidence:.2f})")

        return result_terms

    async def _extract_chunk(
        self,
        text_chunk: str,
        terms_per_chunk: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Extract terms from a single text chunk using GPT-4o.

        This is a private method called by process() for each chunk.

        Args:
            text_chunk: Text chunk to process
            terms_per_chunk: Maximum number of terms to extract

        Returns:
            List of extracted terms

        Raises:
            Exception: If API call fails
        """
        try:
            user_prompt = f"""
다음 텍스트에서 전문 용어를 추출하세요.
최대 {terms_per_chunk}개의 용어를 추출하며, 신뢰도가 높은 순서로 정렬하세요.

텍스트:
{text_chunk}
"""

            logger.info(f"🤖 Calling GPT-4o API (chunk size: {len(text_chunk)} chars)")

            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent results
                response_format={"type": "json_object"}
            )

            # Parse response
            content = response.choices[0].message.content
            result = json.loads(content)
            terms = result.get('terms', [])

            logger.info(f"✅ GPT-4o returned {len(terms)} terms")
            return terms

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GPT-4o response as JSON: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"GPT-4o API call failed: {str(e)}")
            raise Exception(f"GPT-4o API error: {str(e)}")
