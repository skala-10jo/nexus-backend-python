"""
회화 연습을 위한 힌트 생성 에이전트.

현재 step의 terminology를 기반으로 단계별 힌트를 생성합니다:
1. 단어 힌트 (keywords)
2. 구문 힌트 (phrase with blanks)
3. 완전한 문장 (full sentence)
"""
import logging
from typing import List, Dict, Any, Optional

from agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class HintAgent(BaseAgent):
    """
    대화 힌트를 단계별로 생성하는 AI 에이전트.

    현재 step의 terminology에서 원어민이 쓸 것 같은 표현 1개를 선택하고,
    사용자가 단계별로 학습할 수 있도록 힌트를 구조화합니다.

    Returns:
        - targetExpression: 목표 표현 (원어민 표현)
        - wordHints: 핵심 단어들
        - phraseHint: 빈칸이 있는 구문
        - fullSentence: 완전한 문장
        - explanation: 한국어 설명
    """

    LANG_MAP = {
        "en": "English",
        "ko": "Korean (한국어)",
        "zh": "Chinese (中文)",
        "ja": "Japanese (日本語)",
        "vi": "Vietnamese (Tiếng Việt)"
    }

    DIFFICULTY_GUIDELINES = {
        "beginner": {
            "sentence_length": "5-10단어",
            "complexity": "단순 문장 구조",
            "vocabulary": "기본 일상 어휘, 고등학생 수준"
        },
        "intermediate": {
            "sentence_length": "10-15단어",
            "complexity": "복합문 허용",
            "vocabulary": "비즈니스 용어, 일반적인 표현"
        },
        "advanced": {
            "sentence_length": "15-25단어",
            "complexity": "복잡한 문장 구조",
            "vocabulary": "세련된 비즈니스 어휘, 관용구"
        }
    }

    async def process(
        self,
        scenario_context: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        last_ai_message: str,
        current_step: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        단계별 힌트를 생성합니다.

        Args:
            scenario_context: 시나리오 정보 (title, roles, difficulty, language 등)
            conversation_history: 대화 히스토리
            last_ai_message: 마지막 AI 메시지
            current_step: 현재 대화 단계 정보 (선택)
                - name: 단계 영문 식별자
                - title: 단계 한글 제목
                - guide: 단계 가이드
                - terminology: 이 단계에서 사용할 표현 리스트

        Returns:
            단계별 힌트 딕셔너리:
                - targetExpression: 목표 표현 (원어민이 쓸 문장)
                - wordHints: 핵심 단어 리스트 (2-4개)
                - phraseHint: 빈칸이 포함된 구문
                - fullSentence: 완전한 문장
                - explanation: 한국어 설명
                - stepInfo: 현재 단계 정보 (있는 경우)
        """
        required_fields = ["roles", "difficulty", "language"]
        for field in required_fields:
            if field not in scenario_context:
                raise ValueError(f"Missing required field: {field}")

        target_lang = self.LANG_MAP.get(scenario_context.get("language", "en"), "English")
        difficulty = scenario_context.get("difficulty", "intermediate")
        guidelines = self.DIFFICULTY_GUIDELINES.get(difficulty, self.DIFFICULTY_GUIDELINES["intermediate"])

        # 현재 step의 terminology 우선 사용
        step_terminology = []
        step_info = None
        if current_step:
            step_terminology = current_step.get("terminology", [])
            step_info = {
                "name": current_step.get("name"),
                "title": current_step.get("title"),
                "guide": current_step.get("guide")
            }

        # step terminology가 없으면 시나리오 전체 terminology 사용
        if not step_terminology:
            step_terminology = scenario_context.get("required_terminology", [])

        system_prompt = self._build_system_prompt(
            scenario_context, target_lang, guidelines, step_info
        )
        user_prompt = self._build_user_prompt(
            scenario_context, conversation_history, last_ai_message,
            step_terminology, target_lang
        )

        logger.info(f"🎯 Generating stepped hints for scenario: {scenario_context.get('title', 'Unknown')}")
        if step_info:
            logger.info(f"   Current step: {step_info.get('name')} - {step_info.get('title')}")

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=500
            )

            import json
            result = json.loads(response.choices[0].message.content)

            # 응답 검증
            if not result.get("targetExpression"):
                logger.warning("GPT-4o returned no target expression, using fallback")
                return self._generate_fallback_hints(scenario_context, step_terminology, step_info)

            logger.info(f"✅ Generated stepped hints: {result.get('targetExpression', '')[:50]}...")

            return {
                "targetExpression": result.get("targetExpression", ""),
                "wordHints": result.get("wordHints", []),
                "phraseHint": result.get("phraseHint", ""),
                "fullSentence": result.get("fullSentence", result.get("targetExpression", "")),
                "explanation": result.get("explanation", ""),
                "stepInfo": step_info
            }

        except Exception as e:
            logger.error(f"Error generating hints: {str(e)}")
            return self._generate_fallback_hints(scenario_context, step_terminology, step_info)

    def _build_system_prompt(
        self,
        scenario_context: Dict[str, Any],
        target_lang: str,
        guidelines: Dict[str, str],
        step_info: Optional[Dict[str, Any]]
    ) -> str:
        """시스템 프롬프트 구성"""
        user_role = scenario_context.get("roles", {}).get("user", "User")
        ai_role = scenario_context.get("roles", {}).get("ai", "AI Partner")

        step_section = ""
        if step_info:
            step_section = f"""
## 현재 대화 단계
- 단계명: {step_info.get('name', 'N/A')}
- 단계 제목: {step_info.get('title', 'N/A')}
- 가이드: {step_info.get('guide', 'N/A')}
"""

        return f"""당신은 {target_lang} 회화 연습을 돕는 언어 튜터입니다.
사용자가 원어민처럼 자연스럽게 말할 수 있도록 단계별 힌트를 제공합니다.

## 시나리오 정보
- 제목: {scenario_context.get('title', 'N/A')}
- 상황: {scenario_context.get('scenario_text', 'N/A')}
- 사용자 역할: {user_role}
- 대화 상대: {ai_role}
{step_section}
## 난이도: {scenario_context.get('difficulty', 'intermediate').upper()}
- 문장 길이: {guidelines['sentence_length']}
- 문장 구조: {guidelines['complexity']}
- 어휘 수준: {guidelines['vocabulary']}

## 힌트 생성 규칙
1. 목표 표현(targetExpression)은 원어민이 실제로 사용할 자연스러운 {target_lang} 문장
2. 단어 힌트(wordHints)는 문장의 핵심 단어 2-4개 (순서대로)
3. 구문 힌트(phraseHint)는 핵심 단어를 빈칸(___)으로 대체한 문장
4. 설명(explanation)은 한국어로 이 표현을 언제/어떻게 쓰는지 설명
5. 난이도에 맞는 복잡도 유지 (beginner는 쉽게, advanced는 세련되게)"""

    def _build_user_prompt(
        self,
        scenario_context: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        last_ai_message: str,
        step_terminology: List[str],
        target_lang: str
    ) -> str:
        """사용자 프롬프트 구성"""
        # 최근 대화 히스토리
        history_text = ""
        recent_history = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
        for msg in recent_history:
            speaker = "AI" if msg.get("speaker") == "ai" else "User"
            history_text += f"{speaker}: {msg.get('message', '')}\n"

        # step terminology는 힌트 생성에서 제외 - 대화 맥락만으로 자연스러운 힌트 생성
        terminology_section = ""

        return f"""## 대화 상황
{history_text if history_text else "(대화 시작)"}

## 마지막 AI 메시지 (이것에 응답해야 함)
AI: {last_ai_message}
{terminology_section}
---

위 대화 맥락에서, 사용자가 응답할 수 있는 **원어민이 쓸 것 같은 자연스러운 표현 1개**를 선택하고,
단계별 힌트를 생성하세요.

다음 JSON 형식으로 응답하세요:
{{
  "targetExpression": "원어민이 쓸 자연스러운 {target_lang} 문장 1개",
  "wordHints": ["핵심단어1", "핵심단어2", "핵심단어3"],
  "phraseHint": "I'd like to ___ the project ___. (핵심 단어를 ___로 대체)",
  "fullSentence": "targetExpression과 동일한 완전한 문장",
  "explanation": "이 표현의 의미와 사용 상황을 한국어로 설명 (1-2문장)"
}}

중요:
- targetExpression은 대화 맥락에 자연스럽게 이어지는 문장이어야 함
- wordHints는 문장에서 학습자가 모를 수 있는 핵심 단어/구 2-4개
- phraseHint는 wordHints의 단어들을 ___로 대체한 문장
- explanation은 반드시 한국어로 작성"""

    def _generate_fallback_hints(
        self,
        scenario_context: Dict[str, Any],
        step_terminology: List[str],
        step_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """GPT 호출 실패 시 대체 힌트 생성"""
        language = scenario_context.get("language", "en")
        difficulty = scenario_context.get("difficulty", "intermediate")

        # step terminology가 있으면 첫 번째 사용
        if step_terminology:
            target = step_terminology[0]
            words = target.split()[:3]
            return {
                "targetExpression": target,
                "wordHints": words,
                "phraseHint": " ".join(["___" if i % 2 == 0 else w for i, w in enumerate(target.split())]),
                "fullSentence": target,
                "explanation": "이 표현을 사용해보세요.",
                "stepInfo": step_info
            }

        # 기본 폴백
        fallbacks = {
            "en": {
                "beginner": {
                    "targetExpression": "Yes, I understand.",
                    "wordHints": ["yes", "understand"],
                    "phraseHint": "___, I ___.",
                    "explanation": "상대방의 말을 이해했다고 표현하는 기본 문장입니다."
                },
                "intermediate": {
                    "targetExpression": "I'd like to discuss that further.",
                    "wordHints": ["like to", "discuss", "further"],
                    "phraseHint": "I'd ___ ___ that ___.",
                    "explanation": "더 깊이 논의하고 싶을 때 사용하는 공손한 표현입니다."
                },
                "advanced": {
                    "targetExpression": "Could you elaborate on that point?",
                    "wordHints": ["elaborate", "point"],
                    "phraseHint": "Could you ___ on that ___?",
                    "explanation": "상대방에게 더 자세한 설명을 요청하는 전문적인 표현입니다."
                }
            }
        }

        lang_fallbacks = fallbacks.get(language, fallbacks["en"])
        diff_fallback = lang_fallbacks.get(difficulty, lang_fallbacks["intermediate"])

        return {
            "targetExpression": diff_fallback["targetExpression"],
            "wordHints": diff_fallback["wordHints"],
            "phraseHint": diff_fallback["phraseHint"],
            "fullSentence": diff_fallback["targetExpression"],
            "explanation": diff_fallback["explanation"],
            "stepInfo": step_info
        }
