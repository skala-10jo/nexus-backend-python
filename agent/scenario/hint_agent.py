"""
Hint generation agent for conversation practice.

Generates contextually appropriate response suggestions based on:
- Scenario context (roles, topic, difficulty)
- Conversation history
- Required terminology
- User's role in the conversation
"""
import logging
from typing import List, Dict, Any, Optional

from agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class HintAgent(BaseAgent):
    """
    AI agent for generating conversation hints/suggestions.

    Uses GPT-4o to create contextually appropriate response suggestions
    that help users continue the conversation naturally while using
    required terminology and maintaining appropriate difficulty level.

    Example:
        >>> agent = HintAgent()
        >>> hints = await agent.process(
        ...     scenario_context={
        ...         "title": "프로젝트 킥오프 미팅",
        ...         "description": "새 프로젝트 시작 회의",
        ...         "scenario_text": "PM과 개발자가 프로젝트 범위를 논의",
        ...         "roles": {"user": "개발자", "ai": "PM"},
        ...         "required_terminology": ["scope", "timeline", "deliverables"],
        ...         "difficulty": "intermediate",
        ...         "language": "en"
        ...     },
        ...     conversation_history=[
        ...         {"speaker": "ai", "message": "Hi! Ready to discuss the project?"},
        ...         {"speaker": "user", "message": "Yes, I have some questions."}
        ...     ],
        ...     last_ai_message="Sure, what would you like to know?"
        ... )
        >>> print(hints)
    """

    # Language mapping for prompts
    LANG_MAP = {
        "en": "English",
        "ko": "Korean (한국어)",
        "zh": "Chinese (中文)",
        "ja": "Japanese (日本語)",
        "vi": "Vietnamese (Tiếng Việt)"
    }

    # Difficulty-based complexity guidelines
    DIFFICULTY_GUIDELINES = {
        "beginner": {
            "sentence_length": "5-10 words",
            "complexity": "simple sentence structures (subject + verb + object)",
            "vocabulary": "basic everyday vocabulary",
            "grammar": "present/past tense only, simple questions",
            "style": "direct and clear, avoid idioms"
        },
        "intermediate": {
            "sentence_length": "10-20 words",
            "complexity": "compound sentences, conditional clauses allowed",
            "vocabulary": "business terminology, common expressions",
            "grammar": "various tenses, polite forms, indirect questions",
            "style": "professional but conversational, some idioms okay"
        },
        "advanced": {
            "sentence_length": "15-30 words",
            "complexity": "complex sentences with multiple clauses",
            "vocabulary": "sophisticated business vocabulary, nuanced expressions",
            "grammar": "all tenses, subjunctive mood, complex modals",
            "style": "nuanced, diplomatic, idiomatic expressions encouraged"
        }
    }

    async def process(
        self,
        scenario_context: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        last_ai_message: str,
        hint_count: int = 3
    ) -> Dict[str, Any]:
        """
        Generate contextually appropriate response hints.

        Args:
            scenario_context: Scenario information containing:
                - title: Scenario title
                - description: Brief description
                - scenario_text: Detailed scenario situation
                - roles: {"user": "role", "ai": "role"}
                - required_terminology: List of terms to use
                - difficulty: beginner/intermediate/advanced
                - language: Target language code (en, ko, etc.)
                - category: Scenario category (optional)
            conversation_history: Previous messages in the conversation
            last_ai_message: The most recent AI message to respond to
            hint_count: Number of hints to generate (default: 3)

        Returns:
            Dictionary containing:
                - hints: List of suggested responses
                - hint_explanations: Brief explanation for each hint (in Korean)
                - terminology_suggestions: Terms that could be used
                - difficulty_appropriate: Whether hints match difficulty level

        Raises:
            ValueError: If scenario_context is missing required fields
        """
        # Validate required fields
        required_fields = ["roles", "difficulty", "language"]
        for field in required_fields:
            if field not in scenario_context:
                raise ValueError(f"Missing required field: {field}")

        target_lang = self.LANG_MAP.get(
            scenario_context.get("language", "en"),
            "English"
        )
        difficulty = scenario_context.get("difficulty", "intermediate")
        guidelines = self.DIFFICULTY_GUIDELINES.get(difficulty, self.DIFFICULTY_GUIDELINES["intermediate"])

        # Build the prompt
        system_prompt = self._build_system_prompt(
            scenario_context,
            target_lang,
            guidelines
        )
        user_prompt = self._build_user_prompt(
            scenario_context,
            conversation_history,
            last_ai_message,
            hint_count
        )

        logger.info(f"🎯 Generating {hint_count} hints for scenario: {scenario_context.get('title', 'Unknown')}")

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.8,  # Higher for variety in suggestions
                max_tokens=800
            )

            import json
            result = json.loads(response.choices[0].message.content)

            # Validate and structure the response
            hints = result.get("hints", [])
            if not hints:
                logger.warning("GPT-4o returned no hints, using fallback")
                hints = self._generate_fallback_hints(
                    scenario_context,
                    last_ai_message
                )

            logger.info(f"✅ Generated {len(hints)} hints successfully")

            return {
                "hints": hints,
                "hint_explanations": result.get("hint_explanations", []),
                "terminology_suggestions": result.get("terminology_suggestions", []),
                "difficulty_appropriate": True
            }

        except Exception as e:
            logger.error(f"Error generating hints: {str(e)}")
            # Return fallback hints on error
            return {
                "hints": self._generate_fallback_hints(scenario_context, last_ai_message),
                "hint_explanations": ["기본 응답 제안입니다."],
                "terminology_suggestions": scenario_context.get("required_terminology", [])[:2],
                "difficulty_appropriate": False
            }

    def _build_system_prompt(
        self,
        scenario_context: Dict[str, Any],
        target_lang: str,
        guidelines: Dict[str, str]
    ) -> str:
        """Build the system prompt for hint generation."""

        user_role = scenario_context.get("roles", {}).get("user", "User")
        ai_role = scenario_context.get("roles", {}).get("ai", "AI Partner")
        category = scenario_context.get("category", "Business")

        # Determine if this is a business or everyday scenario
        is_business = category not in ["Restaurant", "Hotel", "Shopping", "Hospital",
                                        "Bank", "Post Office", "Cafe", "Transportation",
                                        "Fitness", "Beauty", "Real Estate", "Car Rental",
                                        "Daily Life"]

        scenario_type = "비즈니스" if is_business else "일상"

        return f"""당신은 {target_lang} 회화 연습을 돕는 언어 튜터입니다.
사용자가 대화를 자연스럽게 이어갈 수 있도록 응답 힌트를 생성합니다.

## 시나리오 정보
- 제목: {scenario_context.get('title', 'N/A')}
- 설명: {scenario_context.get('description', 'N/A')}
- 상황: {scenario_context.get('scenario_text', 'N/A')}
- 유형: {scenario_type} 시나리오
- 카테고리: {category}

## 역할
- 사용자 역할: {user_role}
- 대화 상대 역할: {ai_role}

## 난이도: {scenario_context.get('difficulty', 'intermediate').upper()}
- 문장 길이: {guidelines['sentence_length']}
- 문장 구조: {guidelines['complexity']}
- 어휘 수준: {guidelines['vocabulary']}
- 문법 범위: {guidelines['grammar']}
- 스타일: {guidelines['style']}

## 필수 전문용어/표현
{', '.join(scenario_context.get('required_terminology', [])) or '없음'}

## 힌트 생성 규칙
1. 모든 힌트는 **반드시 {target_lang}**로 작성
2. 사용자 역할({user_role})의 관점에서 응답 작성
3. 대화 맥락과 시나리오 상황에 자연스럽게 맞아야 함
4. 난이도에 맞는 문장 복잡도 유지
5. 가능하면 필수 전문용어를 자연스럽게 포함
6. 대화가 앞으로 진행될 수 있도록 질문이나 의견 제시
7. 각 힌트는 서로 다른 방향의 응답 제안 (동의/질문/제안/명확화 요청 등)
8. 힌트 설명은 반드시 한국어로 작성"""

    def _build_user_prompt(
        self,
        scenario_context: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        last_ai_message: str,
        hint_count: int
    ) -> str:
        """Build the user prompt with conversation context."""

        # Format conversation history (last 6 messages for context)
        history_text = ""
        recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history

        for msg in recent_history:
            speaker = "AI" if msg.get("speaker") == "ai" else "User"
            history_text += f"{speaker}: {msg.get('message', '')}\n"

        required_terms = scenario_context.get("required_terminology", [])
        unused_terms = self._find_unused_terms(required_terms, conversation_history)

        return f"""## 대화 히스토리
{history_text if history_text else "(대화 시작)"}

## 마지막 AI 메시지 (이것에 대한 응답 힌트 생성)
AI: {last_ai_message}

## 아직 사용하지 않은 필수 용어
{', '.join(unused_terms) if unused_terms else '(모든 용어 사용됨)'}

---

위 대화 맥락을 바탕으로, 사용자가 자연스럽게 응답할 수 있는 {hint_count}개의 힌트를 생성하세요.

다음 JSON 형식으로 응답하세요:
{{
  "hints": [
    "응답 힌트 1 (목표 언어로)",
    "응답 힌트 2 (목표 언어로)",
    "응답 힌트 3 (목표 언어로)"
  ],
  "hint_explanations": [
    "이 표현의 사용 상황/맥락 설명 (한국어)",
    "이 표현의 사용 상황/맥락 설명 (한국어)",
    "이 표현의 사용 상황/맥락 설명 (한국어)"
  ],
  "terminology_suggestions": ["이 대화에서 사용할 수 있는 용어1", "용어2"]
}}

중요:
- hints 배열의 각 항목은 **목표 언어**로 작성
- hint_explanations는 **반드시 한국어**로 작성하되, "첫 번째 힌트는", "두 번째 힌트는" 같은 순서 표현 없이 바로 설명만 작성
- 각 힌트는 서로 다른 대화 방향을 제안 (예: 동의, 질문, 제안, 의견 표현 등)"""

    def _find_unused_terms(
        self,
        required_terms: List[str],
        conversation_history: List[Dict[str, str]]
    ) -> List[str]:
        """Find terms that haven't been used in user messages yet."""
        if not required_terms:
            return []

        # Combine all user messages
        user_messages = " ".join([
            msg.get("message", "").lower()
            for msg in conversation_history
            if msg.get("speaker") == "user"
        ])

        # Find unused terms
        unused = []
        for term in required_terms:
            if term.lower() not in user_messages:
                unused.append(term)

        return unused

    def _generate_fallback_hints(
        self,
        scenario_context: Dict[str, Any],
        last_ai_message: str
    ) -> List[str]:
        """Generate basic fallback hints when AI generation fails."""
        language = scenario_context.get("language", "en")
        difficulty = scenario_context.get("difficulty", "intermediate")

        # Language-specific fallbacks
        fallbacks = {
            "en": {
                "beginner": [
                    "Yes, I agree with you.",
                    "Could you explain more?",
                    "I have a question about that."
                ],
                "intermediate": [
                    "That's a great point. I think we should also consider...",
                    "Could you elaborate on that aspect?",
                    "I see what you mean. From my perspective..."
                ],
                "advanced": [
                    "I appreciate your insight on this matter. However, I'd like to propose an alternative approach...",
                    "That's a compelling argument. Could you elaborate on the potential implications?",
                    "While I understand your perspective, I believe we should also factor in..."
                ]
            },
            "ko": {
                "beginner": [
                    "네, 알겠습니다.",
                    "조금 더 설명해 주시겠어요?",
                    "질문이 있습니다."
                ],
                "intermediate": [
                    "좋은 의견이네요. 저도 생각해보면...",
                    "그 부분에 대해 좀 더 자세히 말씀해 주시겠어요?",
                    "말씀하신 내용 이해했습니다. 제 생각에는..."
                ],
                "advanced": [
                    "탁월한 통찰이십니다. 다만, 다른 관점에서 접근해 보면...",
                    "설득력 있는 논점이네요. 그에 따른 파급효과에 대해 좀 더 설명해 주시겠습니까?",
                    "말씀하신 관점은 이해하지만, 추가적으로 고려해야 할 요소가..."
                ]
            }
        }

        lang_fallbacks = fallbacks.get(language, fallbacks["en"])
        return lang_fallbacks.get(difficulty, lang_fallbacks["intermediate"])
