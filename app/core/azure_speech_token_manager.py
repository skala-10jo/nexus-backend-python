"""
Azure Speech Token Manager (싱글톤 패턴)

Azure Speech SDK 인증 토큰을 발급하고 캐싱하는 유틸리티입니다.
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Tuple
from app.config import settings

logger = logging.getLogger(__name__)


class AzureSpeechTokenManager:
    """
    Azure Speech Service 토큰 관리자 (싱글톤)

    Azure Speech SDK 사용을 위한 인증 토큰을 발급하고 캐싱합니다.

    Features:
    - 토큰 자동 발급 (Azure Speech API)
    - 9분 캐싱 (10분 유효 기간, 1분 여유)
    - 싱글톤 패턴으로 인스턴스 재사용

    Example:
        >>> manager = AzureSpeechTokenManager.get_instance()
        >>> token, region = await manager.get_token()
        >>> print(f"Token: {token[:20]}..., Region: {region}")
    """

    _instance: Optional['AzureSpeechTokenManager'] = None

    def __init__(self):
        """
        Initialize Azure Speech Token Manager.

        Note: 직접 호출하지 말고 get_instance()를 사용하세요.
        """
        self.api_key = settings.AZURE_SPEECH_KEY
        self.region = settings.AZURE_SPEECH_REGION
        self.token_endpoint = f"https://{self.region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"

        # 토큰 캐싱 (10분 유효, 9분 캐싱)
        self._cached_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

        logger.info(f"Azure Speech Token Manager initialized for region: {self.region}")

    @classmethod
    def get_instance(cls) -> 'AzureSpeechTokenManager':
        """
        싱글톤 인스턴스 반환

        Returns:
            AzureSpeechTokenManager 싱글톤 인스턴스
        """
        if cls._instance is None:
            cls._instance = cls()
            logger.info("Created new AzureSpeechTokenManager singleton instance")
        return cls._instance

    async def get_token(self) -> Tuple[str, str]:
        """
        Azure Speech 토큰 발급 (캐싱 포함)

        캐시된 토큰이 유효하면 재사용하고, 만료되었으면 새로 발급합니다.

        Returns:
            Tuple[str, str]: (token, region)

        Raises:
            requests.RequestException: Azure API 호출 실패
            Exception: 기타 오류
        """
        # 캐시 확인
        if self._is_token_valid():
            logger.info("Using cached Azure Speech token")
            return self._cached_token, self.region

        # 새 토큰 발급
        logger.info(f"Requesting new Azure Speech token from region: {self.region}")

        try:
            response = requests.post(
                self.token_endpoint,
                headers={
                    "Ocp-Apim-Subscription-Key": self.api_key
                },
                timeout=10
            )
            response.raise_for_status()

            # 토큰 캐싱 (9분)
            self._cached_token = response.text
            self._token_expiry = datetime.now() + timedelta(minutes=9)

            logger.info("Azure Speech token issued successfully")
            return self._cached_token, self.region

        except requests.RequestException as e:
            logger.error(f"Failed to get Azure Speech token: {str(e)}")
            raise Exception(f"Failed to issue Azure Speech token: {str(e)}")

    def _is_token_valid(self) -> bool:
        """
        캐시된 토큰이 유효한지 확인

        Returns:
            bool: 토큰이 유효하면 True
        """
        if self._cached_token is None or self._token_expiry is None:
            return False

        return datetime.now() < self._token_expiry

    async def refresh_token(self) -> Tuple[str, str]:
        """
        토큰 강제 갱신 (캐시 무효화)

        Returns:
            Tuple[str, str]: (token, region)
        """
        logger.info("Forcing Azure Speech token refresh")
        self._cached_token = None
        self._token_expiry = None
        return await self.get_token()

    def get_region(self) -> str:
        """
        리전 반환

        Returns:
            str: Azure Speech 리전
        """
        return self.region


# 하위 호환성을 위한 별칭 (deprecated)
AzureSpeechAgent = AzureSpeechTokenManager
