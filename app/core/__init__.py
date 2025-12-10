"""
Core utilities for backend-python.
Provides shared resources like OpenAI client, text processing, file extraction, and caching.
"""
from .openai_client import get_openai_client
from .text_utils import split_text_into_chunks, deduplicate_terms
from .file_utils import extract_text_from_file
from .glossary_cache import glossary_cache, get_glossary_cache, GlossaryCache

__all__ = [
    # OpenAI
    "get_openai_client",
    # Text Processing
    "split_text_into_chunks",
    "deduplicate_terms",
    # File Processing
    "extract_text_from_file",
    # Caching
    "glossary_cache",
    "get_glossary_cache",
    "GlossaryCache",
]
