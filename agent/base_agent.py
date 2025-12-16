"""
모든 AI Agent의 베이스 클래스.
공유된 OpenAI 클라이언트를 제공하고 process() 인터페이스를 정의합니다.
"""
from abc import ABC, abstractmethod
from openai import AsyncOpenAI
from app.core.openai_client import get_openai_client


class BaseAgent(ABC):
    """
    모든 AI Agent의 추상 베이스 클래스.

    모든 AI Agent는 다음을 준수해야 합니다:
    1. BaseAgent를 상속받을 것
    2. process() 메서드를 구현할 것
    3. OpenAI API 호출 시 self.client를 사용할 것

    OpenAI 클라이언트는 싱글톤 패턴을 통해 모든 Agent 간에 자동으로 공유됩니다.

    사용 예시:
        >>> class MyAgent(BaseAgent):
        ...     async def process(self, text: str) -> str:
        ...         response = await self.client.chat.completions.create(
        ...             model="gpt-4o-mini",
        ...             messages=[{"role": "user", "content": text}]
        ...         )
        ...         return response.choices[0].message.content
        ...
        >>> agent = MyAgent()
        >>> result = await agent.process("안녕하세요")
    """

    def __init__(self):
        """
        공유된 OpenAI 클라이언트로 Agent를 초기화합니다.

        클라이언트는 싱글톤 패턴을 통해 획득되며,
        모든 Agent에서 단 하나의 인스턴스만 생성됨을 보장합니다.
        """
        self.client: AsyncOpenAI = get_openai_client()

    @abstractmethod
    async def process(self, *args, **kwargs):
        """
        각 Agent가 반드시 구현해야 하는 핵심 처리 메서드.

        Agent 기능의 메인 인터페이스입니다.
        각 Agent는 자신만의 입력/출력 시그니처를 정의합니다.

        Args:
            *args: Agent별 위치 인자
            **kwargs: Agent별 키워드 인자

        Returns:
            Agent별 결과 (구현에 따라 다름)

        Raises:
            Exception: AI 처리 실패 시

        참고:
            서브클래스는 반드시 이 메서드를 구현해야 하며,
            적절한 타입 힌트와 입출력을 설명하는 docstring을 포함해야 합니다.
        """
        pass
