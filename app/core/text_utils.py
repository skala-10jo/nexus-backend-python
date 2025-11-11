"""
Text processing utilities for AI agents.
Includes text chunking and term deduplication.
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def split_text_into_chunks(text: str, chunk_size: int = 5000) -> List[str]:
    """
    Split text into chunks of specified size.

    Splits on word boundaries to avoid breaking words.
    Useful for processing large documents with GPT models that have token limits.

    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk in characters (default: 5000)

    Returns:
        List of text chunks

    Example:
        >>> text = "This is a very long document..."
        >>> chunks = split_text_into_chunks(text, chunk_size=1000)
        >>> print(f"Split into {len(chunks)} chunks")
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
