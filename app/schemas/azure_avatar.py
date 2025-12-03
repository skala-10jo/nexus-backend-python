"""
Azure Avatar Response Schemas
"""
from pydantic import BaseModel
from typing import Optional, List


class AvatarTokenResponse(BaseModel):
    """Avatar 토큰 응답 스키마"""
    token: str
    region: str


class AvatarApplyRequest(BaseModel):
    """Avatar 적용 요청 스키마 - TTS 결과를 아바타로 변환"""
    text: str
    audio_url: Optional[str] = None
    language: str = "en-US"
    avatar_style: str = "casual"  # casual, formal, presentation


class VisemeData(BaseModel):
    """Viseme 데이터 (입모양 동기화용)"""
    offset_ms: int
    viseme_id: int


class AvatarApplyResponse(BaseModel):
    """Avatar 적용 응답 스키마"""
    session_id: str
    avatar_type: str  # webrtc, viseme, video
    audio_url: Optional[str] = None
    visemes: Optional[List[VisemeData]] = None
    duration_ms: Optional[int] = None
