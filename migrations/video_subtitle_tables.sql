-- Video Translation Feature Database Migration
-- Created: 2025-01-17
-- Purpose: STT 및 번역된 영상 자막 저장

-- ========== 1. video_subtitles 테이블 생성 ==========
CREATE TABLE IF NOT EXISTS video_subtitles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- 자막 메타데이터
    subtitle_type VARCHAR(20) NOT NULL CHECK (subtitle_type IN ('ORIGINAL', 'TRANSLATED')),
    language VARCHAR(10) NOT NULL,
    file_path VARCHAR(500),

    -- 타임스탬프
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- 코멘트 추가
COMMENT ON TABLE video_subtitles IS '영상 자막 메타데이터';
COMMENT ON COLUMN video_subtitles.subtitle_type IS '자막 타입 (ORIGINAL: STT 결과, TRANSLATED: 번역된 자막)';
COMMENT ON COLUMN video_subtitles.language IS '언어 코드 (ko, en, ja, vi 등)';
COMMENT ON COLUMN video_subtitles.file_path IS 'SRT 파일 경로 (옵션)';

-- ========== 2. video_subtitle_segments 테이블 생성 ==========
CREATE TABLE IF NOT EXISTS video_subtitle_segments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subtitle_id UUID NOT NULL REFERENCES video_subtitles(id) ON DELETE CASCADE,

    -- 세그먼트 정보
    sequence_number INTEGER NOT NULL,
    start_time_ms BIGINT NOT NULL CHECK (start_time_ms >= 0),
    end_time_ms BIGINT NOT NULL CHECK (end_time_ms > start_time_ms),
    text TEXT NOT NULL,
    confidence DECIMAL(5, 4) CHECK (confidence >= 0 AND confidence <= 1),

    -- 타임스탬프
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- 코멘트 추가
COMMENT ON TABLE video_subtitle_segments IS '자막 세그먼트 (타임스탬프별 텍스트)';
COMMENT ON COLUMN video_subtitle_segments.sequence_number IS '세그먼트 순서 (1부터 시작)';
COMMENT ON COLUMN video_subtitle_segments.start_time_ms IS '시작 시간 (밀리초)';
COMMENT ON COLUMN video_subtitle_segments.end_time_ms IS '종료 시간 (밀리초)';
COMMENT ON COLUMN video_subtitle_segments.text IS '자막 텍스트';
COMMENT ON COLUMN video_subtitle_segments.confidence IS '신뢰도 (0.0 ~ 1.0)';

-- ========== 3. 인덱스 생성 ==========

-- video_subtitles 인덱스
CREATE INDEX IF NOT EXISTS idx_video_subtitles_document_id
    ON video_subtitles(document_id);

CREATE INDEX IF NOT EXISTS idx_video_subtitles_type_lang
    ON video_subtitles(subtitle_type, language);

CREATE INDEX IF NOT EXISTS idx_video_subtitles_created_at
    ON video_subtitles(created_at DESC);

-- video_subtitle_segments 인덱스
CREATE INDEX IF NOT EXISTS idx_video_subtitle_segments_subtitle_id
    ON video_subtitle_segments(subtitle_id);

CREATE INDEX IF NOT EXISTS idx_video_subtitle_segments_sequence
    ON video_subtitle_segments(subtitle_id, sequence_number);

CREATE INDEX IF NOT EXISTS idx_video_subtitle_segments_time
    ON video_subtitle_segments(start_time_ms, end_time_ms);

-- ========== 4. 제약조건 추가 ==========

-- 고유 제약: 같은 자막에서 sequence_number 중복 방지
ALTER TABLE video_subtitle_segments
    ADD CONSTRAINT uq_subtitle_sequence
    UNIQUE (subtitle_id, sequence_number);

-- ========== 5. 트리거 생성 (updated_at 자동 업데이트) ==========

-- updated_at 자동 업데이트 함수
CREATE OR REPLACE FUNCTION update_video_subtitle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 트리거 연결
DROP TRIGGER IF EXISTS trg_update_video_subtitle_updated_at ON video_subtitles;
CREATE TRIGGER trg_update_video_subtitle_updated_at
    BEFORE UPDATE ON video_subtitles
    FOR EACH ROW
    EXECUTE FUNCTION update_video_subtitle_updated_at();

-- ========== 6. 샘플 데이터 (테스트용, 선택사항) ==========

-- 테스트용 영상 문서 (이미 있다고 가정)
-- INSERT INTO documents (id, user_id, original_filename, stored_filename, file_path, ...)
-- VALUES (...);

-- 샘플 자막
-- INSERT INTO video_subtitles (id, document_id, subtitle_type, language)
-- VALUES (
--     'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
--     '{existing_video_document_id}',
--     'ORIGINAL',
--     'ko'
-- );

-- 샘플 세그먼트
-- INSERT INTO video_subtitle_segments (subtitle_id, sequence_number, start_time_ms, end_time_ms, text, confidence)
-- VALUES
--     ('a1b2c3d4-e5f6-7890-abcd-ef1234567890', 1, 0, 3500, '안녕하세요.', 0.95),
--     ('a1b2c3d4-e5f6-7890-abcd-ef1234567890', 2, 3500, 7000, '반갑습니다.', 0.92);

-- ========== 7. 롤백 스크립트 (필요시 사용) ==========

/*
-- 테이블 삭제 (순서 중요!)
DROP TRIGGER IF EXISTS trg_update_video_subtitle_updated_at ON video_subtitles;
DROP FUNCTION IF EXISTS update_video_subtitle_updated_at();
DROP TABLE IF EXISTS video_subtitle_segments CASCADE;
DROP TABLE IF EXISTS video_subtitles CASCADE;
*/
