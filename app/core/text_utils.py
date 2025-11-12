"""
Text processing utilities for AI agents.
Includes text chunking and term deduplication.
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[str]:
    """
    텍스트를 오버랩이 있는 청크로 분할.

    오버랩을 사용하여 청크 경계에서 정보 손실을 방지하고,
    단어/문장 경계에서 자연스럽게 분할합니다.

    Args:
        text: 원본 텍스트
        chunk_size: 청크 크기 (자, 기본값: 1000)
        overlap: 오버랩 크기 (자, 일반적으로 chunk_size의 10-20%, 기본값: 200)

    Returns:
        청크 리스트

    Example:
        >>> text = "가나다라마바사아자차카타파하" * 100  # 1400자
        >>> chunks = split_text_into_chunks(text, chunk_size=1000, overlap=200)
        >>> len(chunks)  # 2개 청크 (0-1000, 800-1400)
        2
        >>> # 오버랩 확인
        >>> chunks[0][-200:] == chunks[1][:200]
        True

    Notes:
        - 1000자 ≈ 250-330 토큰 (OpenAI 최대 8191 토큰)
        - 오버랩으로 문맥 보존 및 검색 recall 향상
        - 단어/문장 경계에서 자연스럽게 분할
    """
    if not text or len(text) == 0:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        # 단어/문장 경계에서 자르기 (마지막 청크가 아닐 때만)
        if end < len(text):
            # 공백, 마침표, 줄바꿈 등에서 자르기
            last_boundary = max(
                chunk.rfind(' '),
                chunk.rfind('.'),
                chunk.rfind('\n'),
                chunk.rfind('。'),  # 한글 마침표
                chunk.rfind('!')
            )

            # 경계가 청크의 80% 이상 위치에 있으면 사용
            if last_boundary > chunk_size * 0.8:
                chunk = chunk[:last_boundary + 1]
                end = start + last_boundary + 1

        chunks.append(chunk.strip())

        # 다음 시작점: 현재 끝 - 오버랩
        start = end - overlap

        # 오버랩 때문에 무한루프 방지
        if start >= len(text) - overlap:
            break

    avg_size = sum(len(c) for c in chunks) // len(chunks) if chunks else 0
    logger.info(f"📊 Text split into {len(chunks)} chunks (size={chunk_size}, overlap={overlap}, avg={avg_size} chars)")
    return chunks


def deduplicate_terms(terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate terms based on korean term.
    If duplicates exist, keep the one with highest confidence.

    Args:
        terms: List of term dictionaries with 'korean' and 'confidence' keys

    Returns:
        List of unique terms

    Example:
        >>> terms = [
        ...     {"korean": "용어1", "confidence": 0.8},
        ...     {"korean": "용어1", "confidence": 0.9},  # duplicate with higher confidence
        ...     {"korean": "용어2", "confidence": 0.7}
        ... ]
        >>> unique = deduplicate_terms(terms)
        >>> len(unique)  # 2 (용어1 with 0.9, 용어2 with 0.7)
        2
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
