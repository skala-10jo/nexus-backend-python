"""
File extraction utilities for various document formats.
Supports PDF, DOCX, and TXT files.
"""
import os
import io
import base64
import tempfile
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


def extract_text_from_base64(file_content: str, file_name: str) -> str:
    """
    Base64 인코딩된 파일 내용에서 텍스트를 추출합니다.
    내용을 디코딩하고 파일 확장자에 따라 텍스트를 추출합니다.

    Args:
        file_content: Base64 인코딩된 파일 내용
        file_name: 원본 파일명 (확장자 포함)

    Returns:
        str: 추출된 텍스트

    Raises:
        ValueError: 지원하지 않는 파일 타입이거나 텍스트가 너무 짧은 경우
        Exception: 추출 실패 시
    """
    # Base64 내용 디코딩
    try:
        file_bytes = base64.b64decode(file_content)
        logger.info(f"Base64 디코딩 완료: {len(file_bytes)} bytes")
    except Exception as e:
        raise ValueError(f"Base64 디코딩 실패: {str(e)}")

    # 파일명에서 파일 타입 감지
    file_type = detect_file_type(file_name)
    if file_type is None:
        raise ValueError(f"지원하지 않는 파일 타입: {file_name}")

    logger.info(f"{file_type.upper()} 파일에서 텍스트 추출 중 (Base64): {file_name}")

    # 임시 파일 생성 후 텍스트 추출
    suffix = os.path.splitext(file_name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        # 파일 타입에 따라 텍스트 추출
        if file_type == 'pdf':
            text = extract_text_from_pdf(tmp_path)
        elif file_type == 'docx':
            text = extract_text_from_docx(tmp_path)
        elif file_type == 'txt':
            text = extract_text_from_txt(tmp_path)
        else:
            raise ValueError(f"지원하지 않는 파일 타입: {file_type}")
    finally:
        # 임시 파일 정리
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # 추출된 텍스트 검증
    if not text or len(text.strip()) < 100:
        raise ValueError("추출된 텍스트가 너무 짧습니다 (100자 미만)")

    logger.info(f"Base64 텍스트 추출 완료: {len(text)}자")

    return text
