"""
Azure Translator Agent (싱글톤 패턴)

Azure Translator REST API를 사용하여 텍스트를 번역하는 Agent입니다.
"""
import logging
import uuid
from typing import Optional, List, Dict, Any, ClassVar
import aiohttp
from agent.base_agent import BaseAgent
from app.config import settings

logger = logging.getLogger(__name__)


class TranslationAgent(BaseAgent):
    """
    Azure Translator Agent (싱글톤)

    Azure Translator REST API를 사용하여 텍스트를 번역하는 Agent입니다.
    aiohttp 세션을 재사용하여 성능을 최적화합니다.
    """

    _instance: Optional['TranslationAgent'] = None
    _session: ClassVar[Optional[aiohttp.ClientSession]] = None

    def __init__(self):
        """Initialize Translation Agent."""
        super().__init__()
        self.api_key = settings.AZURE_TRANSLATOR_KEY
        self.endpoint = settings.AZURE_TRANSLATOR_ENDPOINT
        self.region = settings.AZURE_TRANSLATOR_REGION

    @classmethod
    async def get_session(cls) -> aiohttp.ClientSession:
        """aiohttp 세션 재사용 (TCP 연결 풀링)"""
        if cls._session is None or cls._session.closed:
            cls._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return cls._session

    @classmethod
    async def close_session(cls):
        """세션 종료 (앱 종료 시 호출)"""
        if cls._session and not cls._session.closed:
            await cls._session.close()
            cls._session = None

    @classmethod
    def get_instance(cls) -> 'TranslationAgent':
        """싱글톤 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def process(
        self,
        text: str,
        source_lang: str = "ko",
        target_lang: str = "en"
    ) -> str:
        """텍스트 번역 (세션 재사용으로 최적화)"""
        if not text or not text.strip():
            return ""

        try:
            session = await self.get_session()

            async with session.post(
                f"{self.endpoint}/translate",
                params={'api-version': '3.0', 'from': source_lang, 'to': target_lang},
                headers={
                    'Ocp-Apim-Subscription-Key': self.api_key,
                    'Ocp-Apim-Subscription-Region': self.region,
                    'Content-type': 'application/json',
                    'X-ClientTraceId': str(uuid.uuid4())
                },
                json=[{'text': text}]
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Translation API error: {response.status}, {error_text}")

                result = await response.json()
                if result and len(result) > 0:
                    translations = result[0].get('translations', [])
                    if translations:
                        return translations[0].get('text', text)
                return text

        except aiohttp.ClientError as e:
            raise Exception(f"번역 HTTP 오류: {str(e)}")
        except Exception as e:
            if "Translation API error" in str(e):
                raise
            raise Exception(f"번역 실패: {str(e)}")

    async def process_batch(
        self,
        texts: List[str],
        source_lang: str = "ko",
        target_lang: str = "en"
    ) -> List[str]:
        """여러 텍스트 일괄 번역 (최대 100개)"""
        if not texts:
            return []

        try:
            session = await self.get_session()

            async with session.post(
                f"{self.endpoint}/translate",
                params={'api-version': '3.0', 'from': source_lang, 'to': target_lang},
                headers={
                    'Ocp-Apim-Subscription-Key': self.api_key,
                    'Ocp-Apim-Subscription-Region': self.region,
                    'Content-type': 'application/json',
                    'X-ClientTraceId': str(uuid.uuid4())
                },
                json=[{'text': text} for text in texts[:100]]
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Batch translation API error: {response.status}, {error_text}")

                results = await response.json()
                return [
                    result.get('translations', [{}])[0].get('text', '')
                    for result in results
                ]

        except aiohttp.ClientError as e:
            raise Exception(f"일괄 번역 HTTP 오류: {str(e)}")
        except Exception as e:
            if "Batch translation API error" in str(e):
                raise
            raise Exception(f"일괄 번역 실패: {str(e)}")

    async def process_multi(
        self,
        text: str,
        source_lang: str,
        target_langs: List[str]
    ) -> List[Dict[str, str]]:
        """여러 언어로 동시 번역 (WebSocket 실시간 번역용, 세션 재사용)"""
        if not text or not text.strip() or not target_langs:
            return []

        try:
            session = await self.get_session()

            async with session.post(
                f"{self.endpoint}/translate",
                params={'api-version': '3.0', 'from': source_lang, 'to': target_langs},
                headers={
                    'Ocp-Apim-Subscription-Key': self.api_key,
                    'Ocp-Apim-Subscription-Region': self.region,
                    'Content-type': 'application/json',
                    'X-ClientTraceId': str(uuid.uuid4())
                },
                json=[{'text': text}]
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Multi translation API error: {response.status}, {error_text}")

                result = await response.json()
                if result and len(result) > 0:
                    return [
                        {"lang": t.get('to', ''), "text": t.get('text', '')}
                        for t in result[0].get('translations', [])
                    ]
                return []

        except aiohttp.ClientError as e:
            raise Exception(f"다중 번역 HTTP 오류: {str(e)}")
        except Exception as e:
            if "Multi translation API error" in str(e):
                raise
            raise Exception(f"다중 번역 실패: {str(e)}")


# 싱글톤 인스턴스 생성 함수
def get_translation_agent() -> TranslationAgent:
    """
    Translation Agent 싱글톤 인스턴스 반환

    Returns:
        TranslationAgent: 싱글톤 인스턴스
    """
    return TranslationAgent.get_instance()
