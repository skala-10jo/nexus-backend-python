"""
Qdrant client singleton for vector database operations.

Author: NEXUS Team
Date: 2025-01-17
"""
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Global client instance
_client = None


def get_qdrant_client() -> QdrantClient:
    """
    Get or create Qdrant client singleton.

    This ensures only one Qdrant client instance is created and reused
    across all AI agents, optimizing resource usage and connection pooling.

    Returns:
        QdrantClient: Shared Qdrant client instance

    Example:
        >>> client = get_qdrant_client()
        >>> client.search(collection_name="email_embeddings", ...)
    """
    global _client

    if _client is None:
        _client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=60.0  # 60 seconds timeout
        )
        logger.info(f"Qdrant client initialized: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")

    return _client


def ensure_collection_exists():
    """
    Ensure the email_embeddings collection exists in Qdrant.

    Creates the collection if it doesn't exist with the correct schema:
    - Vector size: 1536 (OpenAI text-embedding-ada-002)
    - Distance metric: Cosine similarity

    This function is idempotent - safe to call multiple times.

    Example:
        >>> ensure_collection_exists()
        Collection 'email_embeddings' ready
    """
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION_NAME

    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_exists = any(c.name == collection_name for c in collections)

        if collection_exists:
            logger.info(f"Collection '{collection_name}' already exists")
            return

        # Create collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1536,  # OpenAI text-embedding-ada-002 dimension
                distance=models.Distance.COSINE  # Cosine similarity
            )
        )

        # Create index for faster search (optional but recommended)
        client.create_payload_index(
            collection_name=collection_name,
            field_name="email_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )

        client.create_payload_index(
            collection_name=collection_name,
            field_name="user_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )

        logger.info(f"Collection '{collection_name}' created successfully")

    except Exception as e:
        logger.error(f"Failed to ensure collection exists: {str(e)}")
        raise
