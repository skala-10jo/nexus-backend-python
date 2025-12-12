"""
추출된 표현을 PostgreSQL expressions 테이블에 업로드

Usage:
    python scripts/upload_expressions_to_db.py

Input:
    expressions.json (extract_expressions.py 실행 결과)

Prerequisite:
    expressions 테이블이 이미 생성되어 있어야 함 (Flyway migration)


"""
import sys
from pathlib import Path
import logging
import json

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """메인 실행 함수"""
    logger.info("=" * 80)
    logger.info("Upload Expressions to PostgreSQL")
    logger.info("=" * 80)

    # 1. JSON 파일 읽기
    input_file = Path(__file__).parent / "expressions.json"

    if not input_file.exists():
        logger.error(f"File not found: {input_file}")
        logger.error("Run 'python scripts/extract_expressions.py' first!")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        expressions = json.load(f)

    logger.info(f"Loaded {len(expressions)} expressions from {input_file}\n")

    if not expressions:
        logger.warning("No expressions to upload!")
        return

    # 2. DB 연결
    engine = create_engine(settings.DATABASE_URL)

    with engine.connect() as conn:
        # 3. 테이블 존재 확인
        try:
            result = conn.execute(text("SELECT COUNT(*) FROM expressions"))
            existing_count = result.scalar()
            logger.info(f"Table 'expressions' exists with {existing_count} rows")

            # 기존 데이터 삭제 (연관 테이블 포함)
            if existing_count > 0:
                logger.info("Clearing existing data and related tables...")

                # 연관 테이블 먼저 삭제 (FK 참조로 인해)
                conn.execute(text("TRUNCATE TABLE user_expressions, user_expression_quiz_results CASCADE"))
                logger.info("  ✓ Cleared user_expressions, user_expression_quiz_results")

                # expressions 테이블 삭제
                conn.execute(text("DELETE FROM expressions"))
                conn.commit()
                logger.info(f"  ✓ Deleted {existing_count} expressions\n")
        except Exception as e:
            logger.error(f"Table 'expressions' does not exist or error: {str(e)}")
            logger.error("Please create the table first using Flyway migration!")
            sys.exit(1)

        # 4. 데이터 삽입
        logger.info("Inserting expressions...")
        inserted = 0
        failed = 0

        for idx, expr_data in enumerate(expressions, 1):
            try:
                # JSONB 형식으로 examples 변환
                examples_json = json.dumps(expr_data.get("examples", []))

                conn.execute(text("""
                    INSERT INTO expressions (expression, meaning, examples, unit, chapter, source_section)
                    VALUES (:expression, :meaning, CAST(:examples AS jsonb), :unit, :chapter, :source_section)
                    ON CONFLICT (expression) DO UPDATE SET
                        meaning = EXCLUDED.meaning,
                        examples = EXCLUDED.examples,
                        unit = EXCLUDED.unit,
                        chapter = EXCLUDED.chapter,
                        source_section = EXCLUDED.source_section
                """), {
                    "expression": expr_data.get("expression", ""),
                    "meaning": expr_data.get("meaning", ""),
                    "examples": examples_json,
                    "unit": expr_data.get("unit", ""),
                    "chapter": expr_data.get("chapter", ""),
                    "source_section": expr_data.get("source_section", "")
                })
                inserted += 1

                if idx % 50 == 0:
                    conn.commit()
                    logger.info(f"  Committed {idx}/{len(expressions)} expressions...")

            except Exception as e:
                logger.error(f"[{idx}] Failed: {expr_data.get('expression', '')[:30]}... - {str(e)}")
                failed += 1

        # 최종 커밋
        try:
            conn.commit()
        except Exception as e:
            logger.error(f"Final commit failed: {str(e)}")
            conn.rollback()

        # 5. 결과 확인
        result = conn.execute(text("SELECT COUNT(*) FROM expressions"))
        total_count = result.scalar()

        logger.info("\n" + "=" * 80)
        logger.info(f"✅ Upload completed!")
        logger.info(f"   Total expressions in JSON: {len(expressions)}")
        logger.info(f"   Successfully inserted: {inserted}")
        logger.info(f"   Failed: {failed}")
        logger.info(f"   Total in DB: {total_count}")
        logger.info("=" * 80)

        # 6. 샘플 조회 (유닛별)
        logger.info("\n📊 Sample data by unit:")
        units = conn.execute(text("""
            SELECT unit, COUNT(*) as count
            FROM expressions
            GROUP BY unit
            ORDER BY count DESC
        """)).fetchall()

        for unit, count in units:
            logger.info(f"   {unit}: {count} expressions")


if __name__ == "__main__":
    main()
