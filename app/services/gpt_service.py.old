"""
GPT-4o service for glossary term extraction.
Uses OpenAI API to extract technical terms from text.
"""
import logging
import json
import math
from typing import List, Dict, Any
from openai import AsyncOpenAI
from app.config import settings

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# System prompt for GPT-4o
SYSTEM_PROMPT = """
당신은 전문 용어 추출 전문가입니다.
주어진 텍스트에서 IT 및 프로젝트 관리 관련 전문 용어를 추출하고 분석합니다.

**출력 형식 (JSON)**:
{
  "terms": [
    {
      "korean": "한글 용어",
      "english": "English term",
      "abbreviation": "약어 (있는 경우만, 없으면 null)",
      "definition": "용어의 명확한 정의 (1-2문장)",
      "context": "문서 내에서 사용된 구체적인 맥락 (원문 인용)",
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
6. 신뢰도는 해당 용어의 전문성 정도 (0.0-1.0)
7. 도메인은 가장 적합한 분야 하나만 선택

**주의사항**:
- 중복된 용어는 하나만 포함
- 유사 용어는 별도로 추출 (예: "데이터베이스"와 "관계형 데이터베이스"는 별개)
- 외래어는 가능한 한글과 영어를 모두 포함
- 반드시 JSON 형식으로만 응답
"""


def split_text_into_chunks(text: str, chunk_size: int = 5000) -> List[str]:
    """
    Split text into chunks of specified size.

    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk (default: 5000 characters)

    Returns:
        List of text chunks
    """
    chunks = []
    words = text.split()
    current_chunk = []
    current_size = 0

    for word in words:
        word_size = len(word) + 1  # +1 for space
        if current_size + word_size > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logger.info(f"📊 Text split into {len(chunks)} chunks (avg size: {sum(len(c) for c in chunks) // len(chunks)} chars)")
    return chunks


def deduplicate_terms(terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate terms based on korean term.
    If duplicates exist, keep the one with highest confidence.

    Args:
        terms: List of term dictionaries

    Returns:
        List of unique terms
    """
    unique_terms = {}

    for term in terms:
        korean = term.get('korean', '').strip()
        if not korean:
            continue

        if korean not in unique_terms:
            unique_terms[korean] = term
        else:
            # Keep term with higher confidence
            existing_confidence = float(unique_terms[korean].get('confidence', 0))
            new_confidence = float(term.get('confidence', 0))
            if new_confidence > existing_confidence:
                unique_terms[korean] = term

    logger.info(f"🔍 Deduplicated: {len(terms)} → {len(unique_terms)} unique terms")
    return list(unique_terms.values())


async def extract_terms_from_chunk(
    text_chunk: str,
    terms_per_chunk: int = 20
) -> List[Dict[str, Any]]:
    """
    Extract terms from a single text chunk using GPT-4o.

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

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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


async def extract_terms_with_gpt(
    text: str,
    max_terms: int = 50,
    chunk_size: int = 5000
) -> List[Dict[str, Any]]:
    """
    Extract glossary terms from text using GPT-4o.

    Process:
    1. Split text into chunks
    2. Extract terms from each chunk
    3. Deduplicate terms
    4. Sort by confidence
    5. Return top N terms

    Args:
        text: Full text to extract terms from
        max_terms: Maximum number of terms to return (default: 50)
        chunk_size: Size of text chunks (default: 5000 chars)

    Returns:
        List of extracted terms (dict with korean, english, abbreviation, definition, context, domain, confidence)

    Raises:
        Exception: If extraction fails
    """
    logger.info(f"🚀 Starting term extraction (text length: {len(text)} chars, max terms: {max_terms})")

    # Split text into chunks
    chunks = split_text_into_chunks(text, chunk_size)

    # Calculate terms per chunk
    terms_per_chunk = math.ceil(max_terms / len(chunks)) + 5  # +5 buffer for deduplication

    # Extract terms from each chunk
    all_terms = []
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"📝 Processing chunk {i}/{len(chunks)}")
        try:
            terms = await extract_terms_from_chunk(chunk, terms_per_chunk)
            all_terms.extend(terms)
        except Exception as e:
            logger.warning(f"Failed to extract from chunk {i}: {str(e)}")
            continue

    if not all_terms:
        logger.warning("No terms extracted from any chunk")
        return []

    logger.info(f"📊 Total terms extracted: {len(all_terms)}")

    # Deduplicate terms
    unique_terms = deduplicate_terms(all_terms)

    # Sort by confidence (descending)
    unique_terms.sort(key=lambda x: float(x.get('confidence', 0)), reverse=True)

    # Return top N terms
    result_terms = unique_terms[:max_terms]

    logger.info(f"✅ Term extraction complete: {len(result_terms)} terms (avg confidence: {sum(float(t.get('confidence', 0)) for t in result_terms) / len(result_terms):.2f})")

    return result_terms
