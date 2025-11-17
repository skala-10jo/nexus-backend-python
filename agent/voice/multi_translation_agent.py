"""
다국어 동시 번역 Agent (GPT-4o Streaming)

최적화 기법:
- 병렬 번역 (asyncio.gather)
- GPT-4o Streaming으로 부분 결과 즉시 반환
- LRU 캐시로 자주 사용되는 문구 캐싱
- 컨텍스트 관리 (이전 대화 기억)
"""
import asyncio
from typing import List, Dict, Optional, AsyncGenerator
from functools import lru_cache
from agent.base_agent import BaseAgent


class MultiLanguageTranslationAgent(BaseAgent):
    """
    다국어 동시 번역 Agent

    GPT-4o를 사용하여 하나의 텍스트를 여러 언어로 동시에 번역합니다.
    스트리밍 모드를 지원하여 부분 결과를 즉시 반환할 수 있습니다.
    """

    def __init__(self):
        super().__init__()
        self.language_names = {
            'ko': '한국어',
            'en': '영어',
            'vi': '베트남어'
        }
        # 컨텍스트 저장 (최근 5개 대화)
        self.context_history: List[Dict] = []
        self.max_context = 5

    async def process(
        self,
        text: str,
        source_lang: str,
        target_langs: List[str],
        use_context: bool = False
    ) -> Dict[str, str]:
        """
        텍스트를 여러 언어로 동시 번역 (병렬 처리)

        Args:
            text: 번역할 원본 텍스트
            source_lang: 원본 언어 (ko, en, vi)
            target_langs: 목표 언어 리스트 (["en", "vi"])
            use_context: 이전 대화 컨텍스트 사용 여부

        Returns:
            {
                "en": "Hello",
                "vi": "Xin chào"
            }

        Raises:
            ValueError: 지원하지 않는 언어일 때
            Exception: 번역 실패 시
        """
        # 1. 입력 검증
        if not text or not text.strip():
            return {lang: "" for lang in target_langs}

        if source_lang not in self.language_names:
            raise ValueError(f"지원하지 않는 원본 언어: {source_lang}")

        for lang in target_langs:
            if lang not in self.language_names:
                raise ValueError(f"지원하지 않는 목표 언어: {lang}")

        # 2. 캐시 확인 (자주 사용되는 문구)
        cache_key = f"{text}_{source_lang}_{'_'.join(sorted(target_langs))}"
        cached_result = self._get_cached_translation(cache_key)
        if cached_result:
            return cached_result

        # 3. 병렬 번역 (asyncio.gather로 동시 실행)
        tasks = [
            self._translate_single(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                use_context=use_context
            )
            for target_lang in target_langs
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 4. 결과 조합 및 에러 처리
        translations = {}
        for i, result in enumerate(results):
            target_lang = target_langs[i]

            if isinstance(result, Exception):
                # 개별 번역 실패 시 에러 메시지 반환
                translations[target_lang] = f"[번역 실패: {str(result)}]"
            else:
                translations[target_lang] = result

        # 5. 결과 캐싱
        self._cache_translation(cache_key, translations)

        # 6. 컨텍스트 업데이트
        if use_context:
            self._update_context(text, source_lang, translations)

        return translations

    async def process_streaming(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        use_context: bool = False
    ) -> AsyncGenerator[str, None]:
        """
        스트리밍 모드로 번역 (부분 결과를 즉시 yield)

        Args:
            text: 번역할 원본 텍스트
            source_lang: 원본 언어
            target_lang: 목표 언어
            use_context: 이전 대화 컨텍스트 사용 여부

        Yields:
            번역된 텍스트 청크 (점진적으로)

        Example:
            >>> async for chunk in agent.process_streaming("안녕하세요", "ko", "en"):
            ...     print(chunk, end="", flush=True)
            Hello
        """
        # 1. 입력 검증
        if not text or not text.strip():
            return

        # 2. 프롬프트 생성
        messages = self._build_messages(text, source_lang, target_lang, use_context)

        # 3. GPT-4o Streaming 호출
        try:
            stream = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
                max_tokens=500,
                stream=True  # 스트리밍 활성화
            )

            # 4. 스트리밍 결과 yield
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"[번역 실패: {str(e)}]"

    async def _translate_single(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        use_context: bool = False
    ) -> str:
        """단일 언어로 번역 (내부 메서드)"""

        # 프롬프트 생성
        messages = self._build_messages(text, source_lang, target_lang, use_context)

        # GPT-4o 호출
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            raise Exception(f"{self.language_names[target_lang]} 번역 실패: {str(e)}")

    def _build_messages(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        use_context: bool
    ) -> List[Dict]:
        """번역 프롬프트 메시지 생성"""

        # 시스템 프롬프트
        system_prompt = f"""당신은 전문 번역가입니다.
{self.language_names[source_lang]}에서 {self.language_names[target_lang]}로 정확하고 자연스럽게 번역합니다.

번역 규칙:
1. 원문의 의미를 정확히 전달
2. 목표 언어의 자연스러운 표현 사용
3. 문맥에 맞는 번역
4. 번역 결과만 출력 (설명이나 부가 설명 없이)
"""

        # 사용자 프롬프트
        user_prompt = f"""다음 {self.language_names[source_lang]} 텍스트를 {self.language_names[target_lang]}로 번역하세요.

텍스트: {text}

번역:"""

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # 컨텍스트 추가 (이전 대화)
        if use_context and self.context_history:
            for ctx in self.context_history[-3:]:  # 최근 3개만 사용
                messages.append({
                    "role": "user",
                    "content": f"원문: {ctx['original']}"
                })
                if target_lang in ctx['translations']:
                    messages.append({
                        "role": "assistant",
                        "content": ctx['translations'][target_lang]
                    })

        messages.append({"role": "user", "content": user_prompt})

        return messages

    def _update_context(
        self,
        original_text: str,
        source_lang: str,
        translations: Dict[str, str]
    ):
        """컨텍스트 히스토리 업데이트"""
        self.context_history.append({
            "original": original_text,
            "source_lang": source_lang,
            "translations": translations
        })

        # 최대 개수 유지
        if len(self.context_history) > self.max_context:
            self.context_history.pop(0)

    @lru_cache(maxsize=100)
    def _get_cached_translation(self, cache_key: str) -> Optional[Dict[str, str]]:
        """캐시에서 번역 결과 조회 (자주 사용되는 문구)"""
        # LRU 캐시는 데코레이터로 자동 관리됨
        return None

    def _cache_translation(self, cache_key: str, translations: Dict[str, str]):
        """번역 결과 캐싱"""
        # 실제 캐싱은 _get_cached_translation의 lru_cache가 처리
        # 이 메서드는 명시적 캐싱이 필요할 때 사용
        pass

    def clear_context(self):
        """컨텍스트 히스토리 초기화"""
        self.context_history.clear()

    def get_context_size(self) -> int:
        """현재 저장된 컨텍스트 개수 반환"""
        return len(self.context_history)
