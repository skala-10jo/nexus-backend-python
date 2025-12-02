"""
Expression Speech Service

표현 학습을 위한 발음 평가 및 TTS 서비스
"""
from uuid import UUID
from sqlalchemy.orm import Session
from fastapi import HTTPException

from app.models.expression import Expression
from app.config import Settings
from agent.tts.azure_tts_agent import AzureTTSAgent


class ExpressionSpeechService:
    """
    표현 학습 음성 서비스

    - TTS: expressionId 또는 텍스트로 Azure TTS 음성 합성
    - 발음 평가: WebSocket 실시간 스트리밍 방식으로 제공 (API 엔드포인트에서 직접 처리)
    """

    def __init__(self, settings: Settings):
        """
        Args:
            settings: 애플리케이션 설정 (Azure 자격증명 포함)
        """
        self.settings = settings
        self.tts_agent = AzureTTSAgent.get_instance()

    async def synthesize_text(
        self,
        text: str,
        voice_name: str
    ) -> bytes:
        """
        텍스트를 직접 음성으로 합성

        Args:
            text: 음성으로 변환할 텍스트
            voice_name: 음성 이름

        Returns:
            bytes: WAV 오디오 데이터

        Raises:
            HTTPException: TTS 실패 시
        """
        try:
            audio_data = await self.tts_agent.process(
                text=text,
                voice_name=voice_name,
                rate=0.9  # 원어민 자연스러운 속도
            )
            return audio_data

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"TTS 합성 실패: {str(e)}"
            )

    async def synthesize_speech(
        self,
        expression_id: UUID,
        voice_name: str,
        db: Session
    ) -> bytes:
        """
        TTS 음성 합성

        Args:
            expression_id: 표현 ID
            voice_name: 음성 이름 (예: en-US-JennyNeural)
            db: 데이터베이스 세션

        Returns:
            bytes: WAV 오디오 데이터

        Raises:
            HTTPException: 표현을 찾을 수 없거나 TTS 실패 시
        """
        # 1. DB에서 표현 조회
        expression = db.query(Expression).filter(
            Expression.id == expression_id
        ).first()

        if not expression:
            raise HTTPException(
                status_code=404,
                detail=f"표현을 찾을 수 없습니다: {expression_id}"
            )

        # 2. TTS 합성
        try:
            audio_data = await self.tts_agent.process(
                text=expression.expression,
                voice_name=voice_name,
                rate=0.9  # 원어민 자연스러운 속도
            )
            return audio_data

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"TTS 합성 실패: {str(e)}"
            )
