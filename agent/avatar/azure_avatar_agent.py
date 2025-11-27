"""
Azure Avatar Agent

Azure Speech Service의 Avatar 기능을 위한 토큰 발급 및 설정 제공
"""
import logging
import requests
from typing import Tuple, Optional
from datetime import datetime, timedelta
from app.config import settings

logger = logging.getLogger(__name__)


class AzureAvatarAgent:
    """
    Azure Avatar용 싱글톤 Agent
    Southeast Asia 리전의 Speech Service 사용
    """
    _instance: Optional['AzureAvatarAgent'] = None
    _token_cache: Optional[str] = None
    _token_expires_at: Optional[datetime] = None

    def __init__(self):
        # Pydantic Settings에서 Avatar용 Speech Key/Region 가져오기
        self.subscription_key = settings.AZURE_AVATAR_SPEECH_KEY
        self.region = settings.AZURE_AVATAR_SPEECH_REGION

        if not self.subscription_key:
            raise ValueError("AZURE_AVATAR_SPEECH_KEY environment variable not set")

        self.token_url = f"https://{self.region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"

        logger.info(f"AzureAvatarAgent initialized with region: {self.region}")

    @classmethod
    def get_instance(cls) -> 'AzureAvatarAgent':
        """싱글톤 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def process(self) -> Tuple[str, str]:
        """
        Azure Speech 인증 토큰 발급 (캐싱)

        Returns:
            (token, region) 튜플
        """
        # 캐시된 토큰이 유효하면 반환
        if self._token_cache and self._token_expires_at and datetime.now() < self._token_expires_at:
            logger.info("Using cached Avatar token")
            return self._token_cache, self.region

        # 새 토큰 발급
        token = await self._fetch_token()

        # 캐시 저장 (9분 후 만료 - 실제는 10분)
        self._token_cache = token
        self._token_expires_at = datetime.now() + timedelta(minutes=9)

        logger.info("New Avatar token issued and cached")
        return token, self.region

    async def _fetch_token(self) -> str:
        """
        Azure Speech Service에서 새 토큰 발급

        Returns:
            인증 토큰 문자열

        Raises:
            Exception: 토큰 발급 실패 시
        """
        try:
            headers = {
                "Ocp-Apim-Subscription-Key": self.subscription_key
            }

            response = requests.post(
                self.token_url,
                headers=headers,
                timeout=10
            )

            response.raise_for_status()
            token = response.text

            logger.info(f"Avatar token fetched successfully from {self.region}")
            return token

        except Exception as e:
            logger.error(f"Failed to fetch Avatar token: {str(e)}")
            raise Exception(f"Avatar 토큰 발급 실패: {str(e)}")

    async def refresh_token(self) -> Tuple[str, str]:
        """
        토큰 강제 갱신 (캐시 무시)

        Returns:
            (token, region) 튜플
        """
        logger.info("Forcing Avatar token refresh")
        self._token_cache = None
        self._token_expires_at = None
        return await self.process()

    def get_region(self) -> str:
        """리전 반환"""
        return self.region

    async def get_ice_servers(self) -> dict:
        """
        WebRTC ICE 서버 정보 조회 (TURN 서버)

        Azure Avatar는 전용 TURN 서버가 필요하며,
        REST API를 통해 서버 정보를 가져옵니다.

        Returns:
            dict: ICE 서버 정보 (urls, username, credential)

        Raises:
            Exception: ICE 서버 정보 조회 실패 시
        """
        try:
            ice_url = f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/avatar/relay/token/v1"
            headers = {
                "Ocp-Apim-Subscription-Key": self.subscription_key
            }

            response = requests.get(
                ice_url,
                headers=headers,
                timeout=10
            )

            response.raise_for_status()
            data = response.json()

            logger.info(f"ICE servers retrieved successfully from {self.region}")

            return {
                "urls": data.get("Urls", []),
                "username": data.get("Username", ""),
                "credential": data.get("Password", "")
            }

        except Exception as e:
            logger.error(f"Failed to fetch ICE servers: {str(e)}")
            raise Exception(f"ICE 서버 정보 조회 실패: {str(e)}")

    async def synthesize_avatar_video(self, text: str, language: str = "en-US") -> bytes:
        """
        텍스트를 Avatar 비디오로 변환 (음성만 반환 - 현재 Azure 구독 제한)

        주의: 현재 Azure 구독에서는 Avatar 비디오 생성이 지원되지 않습니다.
        대신 고품질 음성만 생성하여 반환합니다.

        Args:
            text: 변환할 텍스트
            language: 언어 코드 (기본값: en-US)

        Returns:
            오디오 파일 바이너리 데이터 (WebM Opus)

        Raises:
            Exception: 음성 합성 실패 시
        """
        try:
            # 언어별 음성 매핑
            voice_map = {
                'en-US': 'en-US-JennyNeural',
                'ko-KR': 'ko-KR-SunHiNeural',
                'ja-JP': 'ja-JP-NanamiNeural',
                'zh-CN': 'zh-CN-XiaoxiaoNeural'
            }
            voice_name = voice_map.get(language, 'en-US-JennyNeural')

            # SSML 생성 (표준 TTS)
            # 주의: Avatar SSML은 현재 구독에서 무시됨
            ssml = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis'
                       xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='{language}'>
                <voice name='{voice_name}'>
                    {text}
                </voice>
            </speak>"""

            # Azure TTS API 호출 (음성만 생성)
            url = f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"
            headers = {
                "Ocp-Apim-Subscription-Key": self.subscription_key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",  # 음성 전용 포맷
                "User-Agent": "NEXUS-TTS"
            }

            logger.info(f"Synthesizing audio for text: {text[:50]}...")

            response = requests.post(
                url,
                headers=headers,
                data=ssml.encode('utf-8'),
                timeout=60
            )

            response.raise_for_status()

            logger.info(f"Audio synthesized successfully, size: {len(response.content)} bytes")
            logger.warning("Note: Avatar video not available in current subscription - returning audio only")
            return response.content

        except Exception as e:
            logger.error(f"Failed to synthesize audio: {str(e)}")
            raise Exception(f"음성 합성 실패: {str(e)}")
