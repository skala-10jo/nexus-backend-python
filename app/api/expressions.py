"""
API endpoints for expressions (random expressions for dashboard).
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import get_db
from app.models.expression import Expression
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/expressions/random")
async def get_random_expressions(
    limit: int = Query(5, ge=1, le=20, description="Number of random expressions to fetch"),
    db: Session = Depends(get_db)
):
    """
    Get random expressions for dashboard display.

    Args:
        limit: Number of expressions to return (default: 5, max: 20)

    Returns:
        List of random expressions with examples
    """
    try:
        # PostgreSQL random order
        expressions = db.query(Expression).order_by(func.random()).limit(limit).all()

        result = []
        for expr in expressions:
            # examples is JSONB array like [{"text": "...", "translation": "..."}]
            examples = expr.examples or []
            first_example = examples[0] if examples else {}

            # meaning에서 {} 제거
            meaning = expr.meaning or ""
            meaning = meaning.strip("{}")

            result.append({
                "id": str(expr.id),
                "expression": expr.expression,
                "meaning": meaning,
                "example_en": first_example.get("text", ""),
                "example_ko": first_example.get("translation", ""),
                "unit": expr.unit,
                "chapter": expr.chapter
            })

        logger.info(f"Fetched {len(result)} random expressions")

        return {
            "success": True,
            "data": result,
            "count": len(result)
        }

    except Exception as e:
        logger.error(f"Failed to fetch random expressions: {str(e)}")
        return {
            "success": False,
            "data": [],
            "error": str(e)
        }
