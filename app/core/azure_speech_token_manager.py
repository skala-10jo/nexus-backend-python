"""
Azure Speech Token Manager (싱글톤 패턴)

Azure Speech SDK 인증 토큰을 발급하고 캐싱하는 유틸리티입니다.
aiohttp를 사용한 비동기 토큰 발급으로 블로킹을 방지합니다.
"""
import logging
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Tuple
from app.config import settings

logger = logging.getLogger(__name__)


class AzureSpeechTokenManager:
    """
    Azure Speech Service 토큰 관리자 (싱글톤)

    Azure Speech SDK 사용을 위한 인증 토큰을 발급하고 캐싱합니다.

    Features:
    - 토큰 비동기 발급 (aiohttp 사용, 논블로킹)
    - 9분 캐싱 (10분 유효 기간, 1분 여유)
    - 싱글톤 패턴으로 인스턴스 재사용
    - 앱 시작 시 토큰 사전 발급 지원

    Example:
        >>> manager = AzureSpeechTokenManager.get_instance()
        >>> token, region = await manager.get_token()
        >>> print(f"Token: {token[:20]}..., Region: {region}")
    """

    _instance: Optional['AzureSpeechTokenManager'] = None
    _lock: asyncio.Lock = asyncio.Lock()

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

        # 토큰 발급 진행 중 플래그 (중복 발급 방지)
        self._is_fetching: bool = False

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
        Azure Speech 토큰 발급 (캐싱 포함, 비동기)

        캐시된 토큰이 유효하면 재사용하고, 만료되었으면 새로 발급합니다.
        aiohttp를 사용하여 비동기로 처리하므로 이벤트 루프를 블로킹하지 않습니다.

        Returns:
            Tuple[str, str]: (token, region)

        Raises:
            aiohttp.ClientError: Azure API 호출 실패
            Exception: 기타 오류
        """
        # 캐시 확인
        if self._is_token_valid():
            logger.debug("Using cached Azure Speech token")
            return self._cached_token, self.region

        # 새 토큰 발급 (동시성 제어)
        async with self._lock:
            # 락 획득 후 다시 캐시 확인 (다른 코루틴이 이미 갱신했을 수 있음)
            if self._is_token_valid():
                logger.debug("Using cached Azure Speech token (after lock)")
                return self._cached_token, self.region

            logger.info(f"🔑 Requesting new Azure Speech token from region: {self.region}")
            start_time = datetime.now()

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.token_endpoint,
                        headers={"Ocp-Apim-Subscription-Key": self.api_key},
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        response.raise_for_status()
                        token = await response.text()

                        # 토큰 캐싱 (9분)
                        self._cached_token = token
                        self._token_expiry = datetime.now() + timedelta(minutes=9)

                        elapsed = (datetime.now() - start_time).total_seconds()
                        logger.info(f"✅ Azure Speech token issued successfully in {elapsed:.2f}s")
                        return self._cached_token, self.region

            except aiohttp.ClientError as e:
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.error(f"❌ Failed to get Azure Speech token after {elapsed:.2f}s: {str(e)}")
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

    async def prefetch_token(self) -> bool:
        """
        토큰 사전 발급 (앱 시작 시 호출)

        백그라운드에서 토큰을 미리 발급하여 첫 음성 인식 시 지연을 방지합니다.

        Returns:
            bool: 성공 여부
        """
        try:
            logger.info("🚀 Pre-fetching Azure Speech token...")
            await self.get_token()
            logger.info("✅ Azure Speech token pre-fetched successfully")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to pre-fetch Azure Speech token: {e}")
            return False

    def get_region(self) -> str:
        """
        리전 반환

        Returns:
            str: Azure Speech 리전
        """
        return self.region

    def get_cache_status(self) -> dict:
        """
        캐시 상태 반환 (디버깅용)

        Returns:
            dict: 캐시 상태 정보
        """
        return {
            "has_token": self._cached_token is not None,
            "is_valid": self._is_token_valid(),
            "expiry": self._token_expiry.isoformat() if self._token_expiry else None,
            "region": self.region
        }


# 하위 호환성을 위한 별칭 (deprecated)
AzureSpeechAgent = AzureSpeechTokenManager
