"""
Embedding service for saving embeddings to Qdrant.

This module provides common utilities for saving embeddings to Qdrant,
used by various agents (mail, document, etc.)

Author: NEXUS Team
Date: 2025-01-17
"""
from app.core.qdrant_client import get_qdrant_client
from qdrant_client.http import models
from typing import List, Dict, Any
import uuid
import logging

logger = logging.getLogger(__name__)


def save_embeddings_to_qdrant(
    embeddings: List[List[float]],
    payloads: List[Dict[str, Any]],
    collection_name: str
) -> int:
    """
    Save pre-generated embeddings to Qdrant.

    Args:
        embeddings: List of embedding vectors (1536-dim each for text-embedding-ada-002)
        payloads: List of metadata dicts for each embedding
        collection_name: Qdrant collection name (e.g., "email_embeddings")

    Returns:
        Number of points saved

    Raises:
        ValueError: If embeddings and payloads lengths don't match
        Exception: If Qdrant save fails

    Example:
        >>> embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...]]
        >>> payloads = [
        ...     {'email_id': 'uuid', 'chunk_index': 0},
        ...     {'email_id': 'uuid', 'chunk_index': 1}
        ... ]
        >>> save_embeddings_to_qdrant(embeddings, payloads, "email_embeddings")
        2
    """
    if len(embeddings) != len(payloads):
        raise ValueError(
            f"Embeddings and payloads length mismatch: "
            f"{len(embeddings)} != {len(payloads)}"
        )

    if not embeddings:
        logger.warning("No embeddings to save")
        return 0

    # Create Qdrant points
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload=payload
        )
        for embedding, payload in zip(embeddings, payloads)
    ]

    # Save to Qdrant
    try:
        client = get_qdrant_client()
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        logger.info(f"Saved {len(points)} points to Qdrant collection '{collection_name}'")
        return len(points)

    except Exception as e:
        logger.error(f"Failed to save to Qdrant: {str(e)}")
        raise


def delete_embeddings_from_qdrant(
    filter_conditions: Dict[str, Any],
    collection_name: str
) -> None:
    """
    Delete embeddings from Qdrant by filter conditions.

    Args:
        filter_conditions: Filter dict (e.g., {'email_id': 'uuid'})
        collection_name: Qdrant collection name

    Example:
        >>> delete_embeddings_from_qdrant(
        ...     {'email_id': 'some-uuid'},
        ...     "email_embeddings"
        ... )
    """
    try:
        client = get_qdrant_client()
        client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                        for key, value in filter_conditions.items()
                    ]
                )
            )
        )
        logger.info(f"Deleted embeddings from '{collection_name}' with filter: {filter_conditions}")

    except Exception as e:
        logger.error(f"Failed to delete from Qdrant: {str(e)}")
        raise
