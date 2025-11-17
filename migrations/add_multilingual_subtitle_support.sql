-- =====================================================
-- Migration: Add Multilingual Subtitle Support
-- Description: 다국어 자막 지원을 위한 컬럼 추가
-- Date: 2025-01-17
-- =====================================================

-- 1. original_language 컬럼 추가 (원본 언어 코드)
ALTER TABLE video_subtitles
ADD COLUMN IF NOT EXISTS original_language VARCHAR(10);

-- 2. translations 컬럼 추가 (다국어 번역 저장)
ALTER TABLE video_subtitles
ADD COLUMN IF NOT EXISTS translations JSONB DEFAULT '{}'::jsonb;

-- 3. 기존 데이터 마이그레이션
-- original_language를 'ko'로 초기화 (기본값)
UPDATE video_subtitles
SET original_language = 'ko'
WHERE original_language IS NULL;

-- 기존 translated_text를 translations JSON으로 마이그레이션
-- (translated_text가 NULL이 아닌 경우만)
UPDATE video_subtitles
SET translations = jsonb_build_object('en', translated_text)
WHERE translated_text IS NOT NULL
  AND translations = '{}'::jsonb;

-- 4. 인덱스 생성 (성능 최적화)
CREATE INDEX IF NOT EXISTS idx_video_subtitles_original_language
ON video_subtitles(original_language);

-- JSONB 컬럼에 GIN 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_video_subtitles_translations
ON video_subtitles USING GIN (translations);

-- 5. NOT NULL 제약조건 추가
ALTER TABLE video_subtitles
ALTER COLUMN original_language SET NOT NULL;

-- =====================================================
-- Rollback Script (필요 시)
-- =====================================================
-- ALTER TABLE video_subtitles DROP COLUMN IF EXISTS original_language;
-- ALTER TABLE video_subtitles DROP COLUMN IF EXISTS translations;
-- DROP INDEX IF EXISTS idx_video_subtitles_original_language;
-- DROP INDEX IF EXISTS idx_video_subtitles_translations;
