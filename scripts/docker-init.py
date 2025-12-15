#!/usr/bin/env python3
"""
Docker 최초 실행 시 데이터베이스 초기화 스크립트

이 스크립트는 다음 작업을 수행합니다:
1. DB 연결 대기 (PostgreSQL이 준비될 때까지)
2. expressions 테이블에 초기 데이터가 있는지 확인
3. 데이터가 없으면 expressions.json을 업로드

Usage:
    python scripts/docker-init.py
"""
import sys
import time
import json
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 상수 정의
MAX_RETRIES = 30  # DB 연결 최대 재시도 횟수
RETRY_INTERVAL = 2  # 재시도 간격 (초)
INIT_FLAG_FILE = "/app/.init_completed"  # 초기화 완료 플래그 파일


def wait_for_db(engine, max_retries: int = MAX_RETRIES) -> bool:
    """
    PostgreSQL이 준비될 때까지 대기

    Args:
        engine: SQLAlchemy 엔진
        max_retries: 최대 재시도 횟수

    Returns:
        bool: 연결 성공 여부
    """
    logger.info("PostgreSQL 연결 대기 중...")

    for attempt in range(1, max_retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.info(f"✅ PostgreSQL 연결 성공 (시도 {attempt}/{max_retries})")
                return True
        except OperationalError as e:
            logger.warning(f"DB 연결 실패 ({attempt}/{max_retries}): {str(e)[:50]}...")
            time.sleep(RETRY_INTERVAL)

    logger.error(f"❌ PostgreSQL 연결 실패 - {max_retries}회 시도 후 포기")
    return False


def check_expressions_table_exists(conn) -> bool:
    """expressions 테이블 존재 여부 확인"""
    try:
        conn.execute(text("SELECT 1 FROM expressions LIMIT 1"))
        return True
    except Exception:
        return False


def get_expressions_count(conn) -> int:
    """expressions 테이블의 레코드 수 조회"""
    try:
        result = conn.execute(text("SELECT COUNT(*) FROM expressions"))
        return result.scalar() or 0
    except Exception:
        return 0


def upload_expressions(conn, expressions_file: Path) -> int:
    """
    expressions.json 파일을 DB에 업로드

    Args:
        conn: DB 연결
        expressions_file: JSON 파일 경로

    Returns:
        int: 삽입된 레코드 수
    """
    if not expressions_file.exists():
        logger.warning(f"expressions.json 파일 없음: {expressions_file}")
        return 0

    with open(expressions_file, "r", encoding="utf-8") as f:
        expressions = json.load(f)

    logger.info(f"📦 {len(expressions)}개 표현 데이터 업로드 시작...")

    inserted = 0
    for idx, expr_data in enumerate(expressions, 1):
        try:
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

            # 진행 상황 로깅 (100개마다)
            if idx % 100 == 0:
                conn.commit()
                logger.info(f"  진행 중: {idx}/{len(expressions)} ({idx*100//len(expressions)}%)")

        except Exception as e:
            logger.error(f"[{idx}] 삽입 실패: {expr_data.get('expression', '')[:30]}... - {str(e)[:50]}")

    # 최종 커밋
    conn.commit()
    return inserted


def main():
    """메인 실행 함수"""
    logger.info("=" * 60)
    logger.info("🚀 Docker 초기화 스크립트 시작")
    logger.info("=" * 60)

    # 이미 초기화 완료된 경우 스킵
    if Path(INIT_FLAG_FILE).exists():
        logger.info("✅ 이미 초기화 완료됨 - 스킵")
        return 0

    # DB 엔진 생성
    engine = create_engine(settings.DATABASE_URL)

    # 1. DB 연결 대기
    if not wait_for_db(engine):
        logger.error("DB 연결 실패로 초기화 중단")
        return 1

    with engine.connect() as conn:
        # 2. expressions 테이블 확인
        if not check_expressions_table_exists(conn):
            logger.warning("⚠️ expressions 테이블이 없습니다. Flyway 마이그레이션 필요")
            logger.info("Java 백엔드가 테이블을 생성할 때까지 대기...")

            # 테이블 생성 대기 (최대 60초)
            for _ in range(30):
                time.sleep(2)
                if check_expressions_table_exists(conn):
                    logger.info("✅ expressions 테이블 생성 확인")
                    break
            else:
                logger.error("❌ expressions 테이블 생성 대기 시간 초과")
                return 1

        # 3. 기존 데이터 확인
        existing_count = get_expressions_count(conn)
        logger.info(f"현재 expressions 테이블: {existing_count}개 레코드")

        if existing_count > 0:
            logger.info("✅ 이미 데이터가 있음 - 업로드 스킵")
        else:
            # 4. expressions.json 업로드
            expressions_file = Path(__file__).parent / "expressions.json"
            inserted = upload_expressions(conn, expressions_file)

            if inserted > 0:
                logger.info(f"✅ {inserted}개 표현 데이터 업로드 완료!")
            else:
                logger.warning("⚠️ 업로드된 데이터 없음")

        # 5. 최종 확인
        final_count = get_expressions_count(conn)
        logger.info(f"\n📊 최종 expressions 테이블: {final_count}개 레코드")

    # 초기화 완료 플래그 생성
    try:
        Path(INIT_FLAG_FILE).touch()
        logger.info(f"✅ 초기화 완료 플래그 생성: {INIT_FLAG_FILE}")
    except Exception as e:
        logger.warning(f"플래그 파일 생성 실패 (무시): {e}")

    logger.info("=" * 60)
    logger.info("🎉 Docker 초기화 완료!")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
