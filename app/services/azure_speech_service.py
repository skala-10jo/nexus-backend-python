"""
Azure Speech Service 토큰 관리 (싱글톤 패턴)

브라우저에서 Azure Speech SDK를 사용하기 위한 인증 토큰을 발급하고 관리합니다.
"""
import requests
import logging
from typing import Optional
from datetime import datetime, timedelta
from app.config import settings

logger = logging.getLogger(__name__)


class AzureSpeechTokenService:
    """
    Azure Speech Service 토큰 관리 서비스 (싱글톤)

    Features:
    - Azure Speech API 토큰 발급
    - 자동 캐싱 (9분 캐싱, 10분 유효 기간)
    - 브라우저에서 Azure Speech SDK 사용을 위한 토큰 제공
    """

    def __init__(self):
        """Azure Speech Service 초기화"""
        self.api_key = settings.AZURE_SPEECH_KEY
        self.region = settings.AZURE_SPEECH_REGION
        self.token_endpoint = f"https://{self.region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"

        # 토큰 캐싱
        self._cached_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

        logger.info(f"Azure Speech Token Service 초기화 완료 (region={self.region})")

    def get_token(self) -> str:
        """
        Azure Speech 인증 토큰 발급

        캐시된 토큰이 유효하면 재사용하고, 만료되었으면 새로 발급합니다.

        Returns:
            str: Azure Speech authorization token (10분 유효)

        Raises:
            Exception: 토큰 발급 실패 시
        """
        # 캐시된 토큰 확인 (1분 여유 두고 검증)
        if self._cached_token and self._token_expiry:
            if datetime.now() < self._token_expiry:
                logger.debug("캐시된 Azure Speech 토큰 사용")
                return self._cached_token

        # 새 토큰 발급
        logger.info("Azure API에서 새 Speech 토큰 요청")

        try:
            response = requests.post(
                self.token_endpoint,
                headers={
                    "Ocp-Apim-Subscription-Key": self.api_key
                },
                timeout=10
            )
            response.raise_for_status()

            token = response.text

            # 토큰 캐싱 (9분, 실제 유효 기간 10분)
            self._cached_token = token
            self._token_expiry = datetime.now() + timedelta(minutes=9)

            logger.info("Azure Speech 토큰 발급 성공")
            return token

        except Exception as e:
            logger.error(f"Azure Speech 토큰 발급 실패: {str(e)}")
            raise Exception(f"Azure Speech 토큰 발급 실패: {str(e)}")

    def get_region(self) -> str:
        """
        Azure Speech 리전 반환

        Returns:
            str: Azure 리전 (예: "koreacentral")
        """
        return self.region

    def clear_cache(self):
        """토큰 캐시 초기화 (테스트 또는 강제 갱신용)"""
        self._cached_token = None
        self._token_expiry = None
        logger.debug("Azure Speech 토큰 캐시 초기화")


# 전역 싱글톤 인스턴스
azure_speech_service = AzureSpeechTokenService()
