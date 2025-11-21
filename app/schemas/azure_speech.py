"""
Azure Speech Service 스키마 (요청/응답 검증)
"""
from pydantic import BaseModel, Field


class SpeechTokenResponse(BaseModel):
    """Azure Speech 토큰 응답 스키마"""

    token: str = Field(..., description="Azure Speech 인증 토큰 (10분 유효)")
    region: str = Field(..., description="Azure 리전 (예: koreacentral, eastus)")

    class Config:
        json_schema_extra = {
            "example": {
                "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "region": "koreacentral"
            }
        }
