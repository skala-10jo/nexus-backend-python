"""
BizGuide.md를 H2 태그 기준으로 청킹하여 Qdrant에 인덱싱하는 스크립트

Usage:
    python scripts/index_bizguide.py --file /path/to/BizGuide.md

Author: NEXUS Team
Date: 2025-01-18
"""
import asyncio
import re
import argparse
import sys
from pathlib import Path
from typing import List, Dict
import logging

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import AsyncOpenAI
from app.config import settings
from app.core.qdrant_client import get_qdrant_client, ensure_bizguide_collection_exists
from qdrant_client.models import PointStruct
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 챕터별 topic 매핑 (수동 설정)
CHAPTER_TOPICS = {
    "Chapter 1": ["미팅 참여", "미팅에서의 기본 태도", "미팅", "회의", "meeting", "Starting a Meeting", "email"],
    "Chapter 2": ["미팅 참여", "주간미팅 진행하기", "미팅,", "회의", "meeting", "Conducting Weekly Meetings", "email"],
    "Chapter 3": ["미팅 참여", "미팅 마무리하기", "미팅", "회의", "meeting", "Ending the Meeting", "email"],
    "Chapter 4": ["업무 요청 및 개선", "업무 요청하기", "업무", "work", "Requesting Work", "email"],
    "Chapter 5": ["업무 요청 및 개선", "요청 사항 재확인하기", "업무", "work", "Reviewing Requested Information", "email"],
    "Chapter 6": ["업무 요청 및 개선", "개선 요구하기", "업무", "work", "Requesting Improvements", "email"],
    "Chapter 7": ["효과적인 이메일 작성", "요청 메일 작성하기", "이메일", "메일", "email", "Writing a Request Email"],
    "Chapter 8": ["효과적인 이메일 작성", "업무 설명 메일 작성하기", "이메일", "메일", "email", "Writing a Job Description Email"],
    "Chapter 9": ["효과적인 이메일 작성", "협상 메일 작성하기", "이메일", "email", "Writing a Negotiation Email"],
    "Chapter 10": ["업무 피드백 및 문제 해결", "업무 피드백하기", "피드백", "feedback", "Giving Feedback", "email"],
    "Chapter 11": ["업무 피드백 및 문제 해결", "이슈 해결 요청하기", "피드백", "feedback", "Request for an Issue Resolution", "email"],
    "Chapter 12": ["업무 피드백 및 문제 해결", "감정 공유하기", "피드백", "feedback", "Sharing Emotions", "email"],
}


def get_topics_for_chapter(chapter_name: str) -> List[str]:
    """챕터명으로 topic 배열 반환"""
    for key, topics in CHAPTER_TOPICS.items():
        if key in chapter_name:
            return topics
    return ["general"]


def chunk_by_h2(markdown_text: str) -> List[Dict[str, str]]:
    """
    Markdown 텍스트를 H2 태그 기준으로 청킹

    Args:
        markdown_text: BizGuide.md 전체 내용

    Returns:
        청크 리스트 [
            {
                "chapter": "Chapter 1. Starting a Meeting",
                "section": "Media Talk",
                "text": "표현: elephant in the room..."
            },
            ...
        ]
    """
    chunks = []

    # H1으로 챕터 분리
    chapters = re.split(r'^# (.+)$', markdown_text, flags=re.MULTILINE)

    for i in range(1, len(chapters), 2):
        chapter_title = chapters[i].strip()
        chapter_content = chapters[i + 1] if i + 1 < len(chapters) else ""

        # H2로 섹션 분리
        sections = re.split(r'^## (.+)$', chapter_content, flags=re.MULTILINE)

        for j in range(1, len(sections), 2):
            section_title = sections[j].strip()
            section_content = sections[j + 1].strip() if j + 1 < len(sections) else ""

            if section_content:  # 내용이 있는 섹션만
                chunks.append({
                    "chapter": chapter_title,
                    "section": section_title,
                    "text": section_content
                })

    logger.info(f"Total chunks created: {len(chunks)}")
    return chunks


async def index_bizguide(file_path: str, collection_name: str = None):
    """
    BizGuide.md를 Qdrant에 인덱싱

    Args:
        file_path: BizGuide.md 파일 경로
        collection_name: Qdrant 컬렉션 이름 (기본값: settings에서 가져옴)
    """
    if collection_name is None:
        collection_name = settings.QDRANT_BIZGUIDE_COLLECTION

    # 1. 파일 읽기
    logger.info(f"Reading file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 2. 청킹
    logger.info("Chunking by H2 tags...")
    chunks = chunk_by_h2(content)

    if not chunks:
        logger.error("No chunks found!")
        return

    # 3. OpenAI 클라이언트 초기화
    openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    # 4. Qdrant 초기화
    logger.info("Initializing Qdrant collection...")
    ensure_bizguide_collection_exists(collection_name)
    qdrant_client = get_qdrant_client()

    # 5. 각 청크 임베딩 및 업로드
    points = []
    for idx, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {idx + 1}/{len(chunks)}: {chunk['section']}")

        # 임베딩 생성
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk["text"]
        )
        embedding = response.data[0].embedding

        # topic 추출
        topics = get_topics_for_chapter(chunk["chapter"])

        # Point 생성
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk["text"],
                "chapter": chunk["chapter"],
                "section": chunk["section"],
                "topic": topics,
                "source_file": "BizGuide.md"
            }
        )
        points.append(point)

    # 6. 일괄 업로드
    logger.info(f"Uploading {len(points)} points to Qdrant...")
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )

    logger.info("BizGuide indexing completed successfully!")
    logger.info(f"Total indexed: {len(points)} chunks")


def main():
    parser = argparse.ArgumentParser(description="Index BizGuide.md to Qdrant")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to BizGuide.md"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant collection name (default: from settings)"
    )

    args = parser.parse_args()

    # 파일 존재 확인
    if not Path(args.file).exists():
        logger.error(f"File not found: {args.file}")
        sys.exit(1)

    # 인덱싱 실행
    asyncio.run(index_bizguide(args.file, args.collection))


if __name__ == "__main__":
    main()
