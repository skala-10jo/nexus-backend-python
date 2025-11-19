"""
Azure Speech Service schemas for request/response validation.
"""
from pydantic import BaseModel, Field


class SpeechTokenResponse(BaseModel):
    """Response schema for Azure Speech token."""

    token: str = Field(..., description="Azure Speech authorization token (valid for 10 minutes)")
    region: str = Field(..., description="Azure region (e.g., koreacentral, eastus)")

    class Config:
        json_schema_extra = {
            "example": {
                "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "region": "koreacentral"
            }
        }
