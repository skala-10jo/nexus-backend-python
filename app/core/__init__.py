"""
Core utilities for backend-python.
Provides shared resources like OpenAI client, text processing, and file extraction.
"""
from .openai_client import get_openai_client
from .text_utils import split_text_into_chunks, deduplicate_terms
from .file_utils import extract_text_from_file

__all__ = [
    "get_openai_client",
    "split_text_into_chunks",
    "deduplicate_terms",
    "extract_text_from_file",
]
