"""
File extraction utilities for various document formats.
Supports PDF, DOCX, and TXT files.
"""
import os
import logging
from typing import Optional
import PyPDF2
import docx
import chardet

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from PDF file using PyPDF2.

    Args:
        file_path: Path to PDF file

    Returns:
        str: Extracted text

    Raises:
        Exception: If PDF extraction fails
    """
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)

            logger.info(f"📄 Extracting text from PDF ({num_pages} pages)")

            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        return text.strip()

    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {str(e)}")
        raise Exception(f"PDF extraction failed: {str(e)}")


def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from DOCX file using python-docx.

    Args:
        file_path: Path to DOCX file

    Returns:
        str: Extracted text

    Raises:
        Exception: If DOCX extraction fails
    """
    try:
        doc = docx.Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        text = "\n".join(paragraphs)

        logger.info(f"📄 Extracted text from DOCX ({len(paragraphs)} paragraphs)")

        return text.strip()

    except Exception as e:
        logger.error(f"Failed to extract text from DOCX: {str(e)}")
        raise Exception(f"DOCX extraction failed: {str(e)}")


def extract_text_from_txt(file_path: str) -> str:
    """
    Extract text from TXT file with automatic encoding detection.

    Args:
        file_path: Path to TXT file

    Returns:
        str: Extracted text

    Raises:
        Exception: If TXT extraction fails
    """
    try:
        # Detect file encoding
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] or 'utf-8'

        # Read file with detected encoding
        with open(file_path, 'r', encoding=encoding) as file:
            text = file.read()

        logger.info(f"📄 Extracted text from TXT (encoding: {encoding})")

        return text.strip()

    except Exception as e:
        logger.error(f"Failed to extract text from TXT: {str(e)}")
        raise Exception(f"TXT extraction failed: {str(e)}")


def detect_file_type(file_path: str) -> Optional[str]:
    """
    Detect file type based on file extension.

    Args:
        file_path: Path to file

    Returns:
        str: File type ('pdf', 'docx', 'txt') or None if unsupported
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        return 'pdf'
    elif ext in ['.docx', '.doc']:
        return 'docx'
    elif ext in ['.txt', '.text']:
        return 'txt'
    else:
        return None


def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from file (PDF, DOCX, or TXT).
    Automatically detects file type and uses appropriate extraction method.

    Args:
        file_path: Path to document file

    Returns:
        str: Extracted text

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is unsupported or text is too short
        Exception: If extraction fails

    Example:
        >>> text = extract_text_from_file("/path/to/document.pdf")
        >>> print(f"Extracted {len(text)} characters")
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Detect file type
    file_type = detect_file_type(file_path)

    if file_type is None:
        raise ValueError(f"Unsupported file type: {file_path}")

    logger.info(f"📂 Extracting text from {file_type.upper()} file: {os.path.basename(file_path)}")

    # Extract text based on file type
    if file_type == 'pdf':
        text = extract_text_from_pdf(file_path)
    elif file_type == 'docx':
        text = extract_text_from_docx(file_path)
    elif file_type == 'txt':
        text = extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # Validate extracted text
    if not text or len(text.strip()) < 100:
        raise ValueError("Extracted text is too short (< 100 characters)")

    logger.info(f"✅ Text extraction complete: {len(text)} characters")

    return text
