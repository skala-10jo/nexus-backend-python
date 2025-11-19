"""
Qdrant bizguide 컬렉션 삭제 후 재인덱싱 스크립트

Usage:
    python scripts/reset_bizguide.py --file /path/to/BizGuide.md

Author: NEXUS Team
Date: 2025-01-19
"""
import asyncio
import argparse
import sys
from pathlib import Path
import logging

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.qdrant_client import get_qdrant_client
from scripts.index_bizguide import index_bizguide

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def reset_and_reindex(file_path: str, collection_name: str = None):
    """
    Qdrant bizguide 컬렉션을 삭제하고 다시 인덱싱

    Args:
        file_path: BizGuide.md 파일 경로
        collection_name: Qdrant 컬렉션 이름 (기본값: settings에서 가져옴)
    """
    if collection_name is None:
        from app.config import settings
        collection_name = settings.QDRANT_BIZGUIDE_COLLECTION

    qdrant_client = get_qdrant_client()

    # 1. 기존 컬렉션 삭제
    try:
        collections = qdrant_client.get_collections().collections
        collection_exists = any(c.name == collection_name for c in collections)

        if collection_exists:
            logger.info(f"Deleting existing collection: {collection_name}")
            qdrant_client.delete_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully")
        else:
            logger.info(f"Collection '{collection_name}' does not exist, skipping deletion")
    except Exception as e:
        logger.error(f"Failed to delete collection: {str(e)}")
        raise

    # 2. 다시 인덱싱
    logger.info("Starting re-indexing...")
    await index_bizguide(file_path, collection_name)
    logger.info("Re-indexing completed!")


def main():
    parser = argparse.ArgumentParser(description="Reset and re-index BizGuide to Qdrant")
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

    # 리셋 및 재인덱싱 실행
    asyncio.run(reset_and_reindex(args.file, args.collection))


if __name__ == "__main__":
    main()
