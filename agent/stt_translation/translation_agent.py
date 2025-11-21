"""
Azure Translator Agent (싱글톤 패턴)

Azure Translator REST API를 사용하여 텍스트를 번역하는 Agent입니다.
"""
import logging
import uuid
from typing import Optional, List, Dict, Any
import aiohttp
from agent.base_agent import BaseAgent
from app.config import settings

logger = logging.getLogger(__name__)


class TranslationAgent(BaseAgent):
    """
    Azure Translator Agent (싱글톤)

    Azure Translator REST API를 사용하여 텍스트를 번역하는 Agent입니다.

    Features:
    - Azure Translator Text API v3.0
    - 다국어 번역 지원 (ISO 639-1 언어 코드)
    - 싱글톤 패턴
    - 비동기 HTTP 클라이언트 (aiohttp)

    Example:
        >>> agent = TranslationAgent.get_instance()
        >>> result = await agent.process(
        ...     text="안녕하세요",
        ...     source_lang="ko",
        ...     target_lang="en"
        ... )
        >>> print(result)  # "Hello"
    """

    _instance: Optional['TranslationAgent'] = None

    def __init__(self):
        """
        Initialize Translation Agent.

        Note: 직접 호출하지 말고 get_instance()를 사용하세요.
        """
        super().__init__()
        self.api_key = settings.AZURE_TRANSLATOR_KEY
        self.endpoint = settings.AZURE_TRANSLATOR_ENDPOINT
        self.region = settings.AZURE_TRANSLATOR_REGION
        logger.info(f"Translation Agent initialized for region: {self.region}")

    @classmethod
    def get_instance(cls) -> 'TranslationAgent':
        """
        싱글톤 인스턴스 반환

        Returns:
            TranslationAgent 싱글톤 인스턴스
        """
        if cls._instance is None:
            cls._instance = cls()
            logger.info("Created new TranslationAgent singleton instance")
        return cls._instance

    async def process(
        self,
        text: str,
        source_lang: str = "ko",
        target_lang: str = "en"
    ) -> str:
        """
        텍스트 번역

        Args:
            text: 원본 텍스트
            source_lang: 원본 언어 (ISO 639-1 코드, 예: ko, en, ja, zh-Hans)
            target_lang: 목표 언어 (ISO 639-1 코드)

        Returns:
            str: 번역된 텍스트

        Raises:
            Exception: 번역 실패 시
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for translation")
            return ""

        try:
            logger.info(f"Translating text: {source_lang} -> {target_lang}, length={len(text)}")

            # Azure Translator API 엔드포인트
            path = '/translate'
            constructed_url = self.endpoint + path

            # 요청 파라미터
            params = {
                'api-version': '3.0',
                'from': source_lang,
                'to': target_lang
            }

            # 요청 헤더
            headers = {
                'Ocp-Apim-Subscription-Key': self.api_key,
                'Ocp-Apim-Subscription-Region': self.region,
                'Content-type': 'application/json',
                'X-ClientTraceId': str(uuid.uuid4())
            }

            # 요청 본문
            body = [{'text': text}]

            # HTTP POST 요청 (비동기)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    constructed_url,
                    params=params,
                    headers=headers,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    # 응답 확인
                    if response.status != 200:
                        error_text = await response.text()
                        error_msg = f"Translation API error: {response.status}, {error_text}"
                        logger.error(error_msg)
                        raise Exception(error_msg)

                    # 응답 파싱
                    result = await response.json()

                    # 번역 결과 추출
                    if result and len(result) > 0:
                        translations = result[0].get('translations', [])
                        if translations and len(translations) > 0:
                            translated_text = translations[0].get('text', '')
                            logger.info(f"Translation success: '{text[:50]}...' -> '{translated_text[:50]}...'")
                            return translated_text

                    # 번역 결과가 없는 경우
                    logger.warning("No translation result in response")
                    return text  # 원본 반환

        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error during translation: {str(e)}", exc_info=True)
            raise Exception(f"번역 HTTP 오류: {str(e)}")

        except Exception as e:
            logger.error(f"Translation failed: {str(e)}", exc_info=True)
            raise Exception(f"번역 실패: {str(e)}")

    async def process_batch(
        self,
        texts: List[str],
        source_lang: str = "ko",
        target_lang: str = "en"
    ) -> List[str]:
        """
        여러 텍스트 일괄 번역

        Args:
            texts: 원본 텍스트 리스트 (최대 100개)
            source_lang: 원본 언어 (ISO 639-1 코드)
            target_lang: 목표 언어 (ISO 639-1 코드)

        Returns:
            List[str]: 번역된 텍스트 리스트

        Raises:
            Exception: 번역 실패 시
        """
        if not texts:
            logger.warning("Empty texts list provided for batch translation")
            return []

        try:
            logger.info(f"Batch translating {len(texts)} texts: {source_lang} -> {target_lang}")

            # Azure Translator API 엔드포인트
            path = '/translate'
            constructed_url = self.endpoint + path

            # 요청 파라미터
            params = {
                'api-version': '3.0',
                'from': source_lang,
                'to': target_lang
            }

            # 요청 헤더
            headers = {
                'Ocp-Apim-Subscription-Key': self.api_key,
                'Ocp-Apim-Subscription-Region': self.region,
                'Content-type': 'application/json',
                'X-ClientTraceId': str(uuid.uuid4())
            }

            # 요청 본문 (최대 100개)
            body = [{'text': text} for text in texts[:100]]

            # HTTP POST 요청 (비동기)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    constructed_url,
                    params=params,
                    headers=headers,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    # 응답 확인
                    if response.status != 200:
                        error_text = await response.text()
                        error_msg = f"Batch translation API error: {response.status}, {error_text}"
                        logger.error(error_msg)
                        raise Exception(error_msg)

                    # 응답 파싱
                    results = await response.json()

                    # 번역 결과 추출
                    translated_texts = []
                    for result in results:
                        translations = result.get('translations', [])
                        if translations and len(translations) > 0:
                            translated_texts.append(translations[0].get('text', ''))
                        else:
                            translated_texts.append('')

                    logger.info(f"Batch translation success: {len(translated_texts)} texts")
                    return translated_texts

        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error during batch translation: {str(e)}", exc_info=True)
            raise Exception(f"일괄 번역 HTTP 오류: {str(e)}")

        except Exception as e:
            logger.error(f"Batch translation failed: {str(e)}", exc_info=True)
            raise Exception(f"일괄 번역 실패: {str(e)}")


# 싱글톤 인스턴스 생성 함수
def get_translation_agent() -> TranslationAgent:
    """
    Translation Agent 싱글톤 인스턴스 반환

    Returns:
        TranslationAgent: 싱글톤 인스턴스
    """
    return TranslationAgent.get_instance()
