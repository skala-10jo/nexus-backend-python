"""
BizGuide에서 유용한 비즈니스 영어 표현(phrase)을 추출하여 JSON 파일로 저장

Usage:
    python scripts/extract_expressions.py

Output:
    expressions.json


"""
import asyncio
import sys
from pathlib import Path
import logging
import json

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import AsyncOpenAI
from app.config import settings
from app.core.qdrant_client import get_qdrant_client

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# Chapter별 unit명과 chapter명 매핑
CHAPTER_TOPICS = {
    "Chapter 1": ["미팅 참여", "미팅에서의 기본 태도"],
    "Chapter 2": ["미팅 참여", "주간미팅 진행하기"],
    "Chapter 3": ["미팅 참여", "미팅 마무리하기"],
    "Chapter 4": ["업무 요청 및 개선", "업무 요청하기"],
    "Chapter 5": ["업무 요청 및 개선", "요청 사항 재확인하기"],
    "Chapter 6": ["업무 요청 및 개선", "개선 요구하기"],
    "Chapter 7": ["효과적인 이메일 작성", "요청 메일 작성하기"],
    "Chapter 8": ["효과적인 이메일 작성", "업무 설명 메일 작성하기"],
    "Chapter 9": ["효과적인 이메일 작성", "협상 메일 작성하기"],
    "Chapter 10": ["업무 피드백 및 문제 해결", "업무 피드백하기"],
    "Chapter 11": ["업무 피드백 및 문제 해결", "이슈 해결 요청하기"],
    "Chapter 12": ["업무 피드백 및 문제 해결", "감정 공유하기"],
}

# 스킵할 섹션 (학습용으로 적합하지 않음)
SKIP_SECTIONS = [
    "Situation Practice",
    "Summary",
    "Understanding Global Business"
]


def should_skip_section(section: str) -> bool:
    """섹션을 스킵해야 하는지 확인"""
    return any(skip in section for skip in SKIP_SECTIONS)


def get_unit_and_chapter(chapter_name: str):
    """챕터명으로 unit명과 chapter명 반환"""
    for key, values in CHAPTER_TOPICS.items():
        # 정확한 매칭을 위해 "Chapter X." 형식으로 체크
        if key + "." in chapter_name or key + " " in chapter_name:
            return values[0], values[1]
    return "general", "general"


async def extract_expressions_from_chunk(client: AsyncOpenAI, text: str, chapter: str, section: str):
    """
    GPT를 사용하여 텍스트에서 유용한 비즈니스 영어 표현을 추출합니다.

    Args:
        client: OpenAI 클라이언트
        text: BizGuide 청크 텍스트
        chapter: 챕터명
        section: 섹션명

    Returns:
        표현 리스트 [{expression(영어 phrase), meaning(한국어 설명), examples: [{text(문장), translation}]}]
    """
    prompt = f"""Extract useful business English phrases from the following text.

Instructions:
1. Find meaningful business phrases/expressions (not full sentences)
2. Extract phrases that are actually useful for business communication
3. For each phrase, provide:
   - The English expression
   - Korean meaning (from curly braces {{}})
   - Exactly 3 example sentences (if less than 3, generate additional ones)
4. Skip if no useful phrases found

Text:
{text}

Response format (JSON):
{{
  "expressions": [
    {{
      "expression": "business phrase",
      "meaning": "Korean meaning",
      "examples": [
        {{"text": "Example sentence in English.", "translation": "한글 번역"}},
        {{"text": "Another example.", "translation": "다른 예시"}},
        {{"text": "Third example.", "translation": "세번째 예시"}}
      ]
    }}
  ]
}}

Important:
- Only extract actual phrases, not full sentences
- Generate additional examples if needed to have exactly 3
- Return empty array if no useful phrases found
- Respond ONLY in valid JSON format
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in business English education. Extract useful phrases and provide practical examples for learners."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        expressions = result.get("expressions", [])

        if expressions:
            logger.info(f"  ✓ Extracted {len(expressions)} phrases")
        else:
            logger.info(f"  - No phrases extracted")

        return expressions

    except Exception as e:
        logger.error(f"  ✗ Failed to extract: {str(e)}")
        return []


async def main():
    """메인 실행 함수"""
    logger.info("=" * 80)
    logger.info("BizGuide Expression Extraction - Step 1: Extract")
    logger.info("=" * 80)

    # 1. Qdrant에서 BizGuide 청크 읽기
    qdrant_client = get_qdrant_client()
    openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    source_collection = settings.QDRANT_BIZGUIDE_COLLECTION

    logger.info(f"Reading chunks from '{source_collection}'...")

    scroll_result = qdrant_client.scroll(
        collection_name=source_collection,
        limit=1000,
        with_payload=True,
        with_vectors=False
    )

    points = scroll_result[0]
    logger.info(f"Found {len(points)} chunks\n")

    # 2. 각 청크에서 표현 추출
    all_expressions = []
    processed = 0
    skipped = 0

    for point in points:
        text = point.payload.get("text", "")
        chapter = point.payload.get("chapter", "")
        section = point.payload.get("section", "")

        processed += 1

        # 스킵할 섹션인지 확인
        if should_skip_section(section):
            logger.info(f"[{processed}/{len(points)}] SKIP: {chapter} - {section}")
            skipped += 1
            continue

        logger.info(f"[{processed}/{len(points)}] Processing: {chapter} - {section}")

        # GPT로 표현 추출
        expressions = await extract_expressions_from_chunk(
            openai_client, text, chapter, section
        )

        if not expressions:
            continue

        # unit명과 chapter명 추출
        unit_name, chapter_name = get_unit_and_chapter(chapter)

        # 메타데이터 추가
        for expr in expressions:
            expr["unit"] = unit_name
            expr["chapter"] = chapter_name
            expr["source_section"] = section
            all_expressions.append(expr)

    # 3. JSON 파일로 저장
    output_file = Path(__file__).parent / "expressions.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_expressions, f, ensure_ascii=False, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info(f"✅ Extraction completed!")
    logger.info(f"   Processed chunks: {processed}")
    logger.info(f"   Skipped chunks: {skipped}")
    logger.info(f"   Total expressions: {len(all_expressions)}")
    logger.info(f"   Output file: {output_file}")
    logger.info("=" * 80)
    logger.info("\n다음 단계: expressions.json 파일을 확인한 후")
    logger.info("python scripts/upload_expressions.py 실행")


if __name__ == "__main__":
    asyncio.run(main())