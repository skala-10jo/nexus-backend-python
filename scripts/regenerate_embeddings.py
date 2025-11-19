"""
기존 임베딩 삭제 후 재생성 스크립트

Usage:
    python scripts/regenerate_embeddings.py <user_id>

Example:
    python scripts/regenerate_embeddings.py 123e4567-e89b-12d3-a456-426614174000
"""
import sys
import asyncio
from app.core.qdrant_client import get_qdrant_client
from app.config import settings
from qdrant_client.http import models


async def delete_user_embeddings(user_id: str):
    """특정 사용자의 모든 임베딩 삭제"""
    qdrant_client = get_qdrant_client()

    print(f"🗑️  Deleting embeddings for user: {user_id}")

    # user_id로 필터링하여 삭제
    qdrant_client.delete(
        collection_name=settings.QDRANT_EMAIL_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            )
        )
    )

    print(f"✅ Deleted all embeddings for user: {user_id}")
    print(f"\n📝 Next step: Call batch API to regenerate")
    print(f"   curl -X POST http://localhost:8000/api/ai/mail/embeddings/batch \\")
    print(f"        -H 'Content-Type: application/json' \\")
    print(f"        -d '{{\"user_id\": \"{user_id}\"}}'")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/regenerate_embeddings.py <user_id>")
        sys.exit(1)

    user_id = sys.argv[1]
    asyncio.run(delete_user_embeddings(user_id))
