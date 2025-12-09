"""
Expression Match API 엔드포인트

비즈니스 표현-예문 매칭 API

Author: NEXUS Team
Date: 2025-01-18
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

from app.services.expression_match_service import ExpressionMatchService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/expression", tags=["Expression Match"])


# Request/Response Schemas
class FindMatchRequest(BaseModel):
    """단일 매칭 요청"""
    expression: str = Field(..., description="비즈니스 표현 (예: 'take (someone) through')")
    sentence: str = Field(..., description="예문 (예: 'Can you take me through the budget?')")
    use_ai_fallback: bool = Field(default=True, description="정규식 실패 시 AI 사용 여부")

    class Config:
        json_schema_extra = {
            "example": {
                "expression": "take (someone) through",
                "sentence": "Can you take me through the budget proposal?",
                "use_ai_fallback": True
            }
        }


class FindMatchResponse(BaseModel):
    """매칭 결과"""
    matched: bool = Field(..., description="매칭 성공 여부")
    start_index: int = Field(..., description="시작 인덱스 (0-based)")
    end_index: int = Field(..., description="끝 인덱스 (exclusive)")
    matched_text: str = Field(..., description="매칭된 텍스트")
    method: str = Field(..., description="매칭 방법 (regex/ai/none)")
    error: Optional[str] = Field(default=None, description="에러 메시지")


class BatchMatchItem(BaseModel):
    """배치 매칭 아이템"""
    expression: str
    sentence: str


class BatchMatchRequest(BaseModel):
    """배치 매칭 요청"""
    items: List[BatchMatchItem] = Field(..., description="매칭할 expression-sentence 쌍들")
    use_ai_fallback: bool = Field(default=True, description="정규식 실패 시 AI 사용 여부")

    class Config:
        json_schema_extra = {
            "example": {
                "items": [
                    {
                        "expression": "take (someone) through",
                        "sentence": "Can you take me through the budget?"
                    },
                    {
                        "expression": "get straight to the point",
                        "sentence": "Let's get straight to the point and discuss the issue."
                    }
                ],
                "use_ai_fallback": True
            }
        }


class BatchMatchResponse(BaseModel):
    """배치 매칭 결과"""
    results: List[FindMatchResponse]
    total: int
    matched_count: int
    regex_count: int
    ai_count: int


# 서비스 인스턴스
_service: Optional[ExpressionMatchService] = None


def get_service() -> ExpressionMatchService:
    """서비스 싱글톤 인스턴스"""
    global _service
    if _service is None:
        _service = ExpressionMatchService()
    return _service


@router.post("/find-match", response_model=FindMatchResponse)
async def find_match(request: FindMatchRequest):
    """
    expression이 sentence 어디에 해당하는지 찾기

    1. 정규식 매칭 시도 (빠름)
    2. 실패 시 AI 매칭 (정확함)

    사용 예:
    - 하이라이트: matched_text 부분을 강조 표시
    - 빈칸 퀴즈: matched_text 부분을 빈칸으로 치환
    """
    try:
        service = get_service()
        result = await service.find_match(
            expression=request.expression,
            sentence=request.sentence,
            use_ai_fallback=request.use_ai_fallback
        )
        return FindMatchResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Find match failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"매칭 처리 실패: {str(e)}")


@router.post("/find-match/batch", response_model=BatchMatchResponse)
async def find_match_batch(request: BatchMatchRequest):
    """
    여러 expression-sentence 쌍에 대해 배치 매칭

    대량의 예문을 한 번에 처리할 때 사용
    """
    try:
        service = get_service()

        items = [
            {"expression": item.expression, "sentence": item.sentence}
            for item in request.items
        ]

        results = await service.find_matches_batch(
            items=items,
            use_ai_fallback=request.use_ai_fallback
        )

        # 통계 계산
        matched_count = sum(1 for r in results if r.get("matched"))
        regex_count = sum(1 for r in results if r.get("method") == "regex")
        ai_count = sum(1 for r in results if r.get("method") == "ai")

        return BatchMatchResponse(
            results=[FindMatchResponse(**r) for r in results],
            total=len(results),
            matched_count=matched_count,
            regex_count=regex_count,
            ai_count=ai_count
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch match failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"배치 매칭 처리 실패: {str(e)}")


@router.get("/health")
async def health_check():
    """Expression Match API 상태 확인"""
    return {"status": "healthy", "service": "expression-match"}
