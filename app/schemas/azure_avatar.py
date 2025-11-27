"""
Azure Avatar Response Schemas
"""
from pydantic import BaseModel


class AvatarTokenResponse(BaseModel):
    """Avatar 토큰 응답 스키마"""
    token: str
    region: str
