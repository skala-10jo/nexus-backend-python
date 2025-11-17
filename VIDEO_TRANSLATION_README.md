# Video Translation Feature - 구현 가이드

## 개요

영상 자막 STT(Speech-to-Text) 및 번역 기능을 제공하는 AI Agent 시스템입니다.

**핵심 기능**:
- OpenAI Whisper API를 사용한 영상 음성 인식 (STT)
- 타임스탬프가 포함된 자막 생성
- 컨텍스트 기반 자막 번역
- SRT 형식 자막 파일 생성 및 다운로드

---

## 아키텍처

### Agent 계층 구조

```
API Layer (app/api/video_translation.py)
    ↓
Service Layer (app/services/video_translation_service.py)
    ↓
Agent Layer (agent/video/)
    ├── VideoSTTAgent          # STT 처리
    ├── SubtitleGeneratorAgent # SRT 파일 생성
    └── ContextEnhancedTranslationAgent (재사용) # 번역
```

### 데이터 플로우

**STT 처리**:
```
1. 영상 파일 (MP4/AVI/MOV)
    ↓ ffmpeg
2. 오디오 파일 (MP3, 16kHz mono)
    ↓ OpenAI Whisper API
3. 타임스탬프 세그먼트
    ↓ DB 저장
4. VideoSubtitle + VideoSubtitleSegment
```

**번역 처리**:
```
1. 원본 자막 세그먼트 조회
    ↓
2. 컨텍스트 문서 조회 (선택)
    ↓ ContextEnhancedTranslationAgent
3. 각 세그먼트 번역
    ↓ DB 저장
4. 번역된 VideoSubtitle + VideoSubtitleSegment
```

**자막 파일 생성**:
```
1. 자막 세그먼트 조회
    ↓ SubtitleGeneratorAgent
2. SRT 형식 변환
    ↓
3. 파일 저장 (uploads/subtitles/)
    ↓
4. 다운로드
```

---

## 파일 구조

```
backend-python/
├── agent/video/                              # Video Agent
│   ├── __init__.py
│   ├── stt_agent.py                          # STT Agent (Whisper API)
│   └── subtitle_generator_agent.py           # 자막 파일 생성 Agent
│
├── app/
│   ├── api/
│   │   └── video_translation.py              # API 엔드포인트
│   │
│   ├── services/
│   │   └── video_translation_service.py      # 비즈니스 로직
│   │
│   ├── models/
│   │   └── video_subtitle.py                 # DB 모델
│   │
│   └── schemas/
│       └── video_translation.py              # Pydantic 스키마
│
└── uploads/subtitles/                         # SRT 파일 저장 경로
```

---

## Agent 상세

### 1. VideoSTTAgent

**위치**: `agent/video/stt_agent.py`

**역할**: 영상에서 음성을 텍스트로 변환 (Whisper API)

**주요 메서드**:
```python
async def process(
    video_file_path: str,
    source_language: str = "ko"
) -> List[Dict[str, Any]]
```

**처리 과정**:
1. ffmpeg로 영상 → MP3 오디오 추출
2. OpenAI Whisper API 호출 (타임스탬프 포함)
3. 응답 파싱 및 세그먼트 변환

**반환 형식**:
```python
[
    {
        "sequence_number": 1,
        "start_time_ms": 0,
        "end_time_ms": 3500,
        "text": "안녕하세요, 오늘은 인공지능에 대해 말씀드리겠습니다.",
        "confidence": 0.95
    },
    ...
]
```

**필수 의존성**:
- `ffmpeg` (시스템에 설치 필요)
  - macOS: `brew install ffmpeg`
  - Ubuntu: `apt install ffmpeg`
  - Windows: [ffmpeg.org](https://ffmpeg.org/download.html)

### 2. SubtitleGeneratorAgent

**위치**: `agent/video/subtitle_generator_agent.py`

**역할**: 타임스탬프 세그먼트를 SRT 형식 파일로 변환

**주요 메서드**:
```python
async def process(
    segments: List[Dict[str, Any]],
    output_path: str,
    subtitle_type: str = "original"
) -> str
```

**SRT 형식 예시**:
```srt
1
00:00:00,000 --> 00:00:03,500
안녕하세요, 오늘은 인공지능에 대해 말씀드리겠습니다.

2
00:00:03,500 --> 00:00:07,200
인공지능은 컴퓨터가 인간처럼 학습하고 추론하는 기술입니다.
```

**타임스탬프 형식**: `HH:MM:SS,mmm` (시:분:초,밀리초)

### 3. ContextEnhancedTranslationAgent (재사용)

**위치**: `agent/translate/context_enhanced_translation_agent.py`

**역할**: 프로젝트 컨텍스트 기반 번역

**사용 예시**:
```python
translated_text = await self.context_translator.process(
    text=original_segment.text,
    source_lang="ko",
    target_lang="en",
    context=context_text,
    glossary_terms=[],
    detected_terms=[]
)
```

---

## 데이터베이스 모델

### VideoSubtitle

**테이블명**: `video_subtitles`

| 컬럼 | 타입 | 설명 |
|-----|------|------|
| id | UUID | Primary Key |
| document_id | UUID | 영상 문서 ID (FK → documents) |
| subtitle_type | Enum | 자막 타입 (ORIGINAL/TRANSLATED) |
| language | String(10) | 언어 코드 (ko, en, ja, vi) |
| file_path | String(500) | SRT 파일 경로 (옵션) |
| created_at | Timestamp | 생성 시간 |
| updated_at | Timestamp | 수정 시간 |

### VideoSubtitleSegment

**테이블명**: `video_subtitle_segments`

| 컬럼 | 타입 | 설명 |
|-----|------|------|
| id | UUID | Primary Key |
| subtitle_id | UUID | 자막 ID (FK → video_subtitles) |
| sequence_number | Integer | 세그먼트 순서 (1부터 시작) |
| start_time_ms | BigInteger | 시작 시간 (밀리초) |
| end_time_ms | BigInteger | 종료 시간 (밀리초) |
| text | Text | 자막 텍스트 |
| confidence | Decimal(5,4) | 신뢰도 (0.0 ~ 1.0) |
| created_at | Timestamp | 생성 시간 |

---

## API 엔드포인트

### 1. STT 처리

**POST** `/api/ai/video/stt`

**요청**:
```json
{
  "video_document_id": "123e4567-e89b-12d3-a456-426614174000",
  "source_language": "ko"
}
```

**응답** (201 Created):
```json
{
  "subtitle_id": "789...",
  "video_document_id": "123...",
  "language": "ko",
  "subtitle_type": "ORIGINAL",
  "segments": [
    {
      "sequence_number": 1,
      "start_time_ms": 0,
      "end_time_ms": 3500,
      "text": "안녕하세요...",
      "confidence": 0.95
    }
  ],
  "total_segments": 10,
  "created_at": "2025-01-17T10:30:00Z"
}
```

### 2. 자막 번역

**POST** `/api/ai/video/translate`

**요청**:
```json
{
  "video_document_id": "123e4567-e89b-12d3-a456-426614174000",
  "document_ids": ["789...", "012..."],
  "source_language": "ko",
  "target_language": "en"
}
```

**응답** (201 Created):
```json
{
  "subtitle_id": "456...",
  "video_document_id": "123...",
  "source_language": "ko",
  "target_language": "en",
  "subtitle_type": "TRANSLATED",
  "segments": [
    {
      "sequence_number": 1,
      "start_time_ms": 0,
      "end_time_ms": 3500,
      "text": "Hello, today I will talk about artificial intelligence.",
      "confidence": 0.95
    }
  ],
  "total_segments": 10,
  "context_used": true,
  "context_document_count": 2,
  "created_at": "2025-01-17T10:35:00Z"
}
```

### 3. 자막 파일 다운로드

**GET** `/api/ai/video/subtitle/{subtitle_id}/download`

**응답**: SRT 파일 (application/x-subrip)

**헤더**:
```
Content-Type: application/x-subrip
Content-Disposition: attachment; filename="subtitle_ko_ORIGINAL.srt"
```

### 4. 자막 정보 조회

**GET** `/api/ai/video/subtitle/{subtitle_id}/info`

**응답**:
```json
{
  "subtitle_id": "789...",
  "file_path": "/path/to/subtitle.srt",
  "language": "ko",
  "subtitle_type": "ORIGINAL",
  "file_size_bytes": 2048
}
```

---

## 사용 예시

### 1. 영상 업로드 및 STT

```python
# Step 1: 영상 업로드 (Java Backend)
POST /api/documents/upload
Content-Type: multipart/form-data
file: video.mp4

# Response:
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "filename": "video.mp4",
  "fileType": "VIDEO",
  ...
}

# Step 2: STT 처리 (Python Backend)
POST /api/ai/video/stt
{
  "video_document_id": "123e4567-e89b-12d3-a456-426614174000",
  "source_language": "ko"
}

# Response:
{
  "subtitle_id": "789...",
  "segments": [...],
  "total_segments": 15
}
```

### 2. 자막 번역

```python
# Step 1: 원본 자막 확인 (STT 완료 후)
# subtitle_id: "789..."

# Step 2: 번역 요청
POST /api/ai/video/translate
{
  "video_document_id": "123...",
  "document_ids": ["doc1...", "doc2..."],  # 컨텍스트 문서
  "source_language": "ko",
  "target_language": "en"
}

# Response:
{
  "subtitle_id": "456...",
  "segments": [...],  # 번역된 세그먼트
  "context_used": true,
  "context_document_count": 2
}
```

### 3. 자막 파일 다운로드

```python
# 원본 자막 다운로드
GET /api/ai/video/subtitle/789.../download

# 번역된 자막 다운로드
GET /api/ai/video/subtitle/456.../download
```

---

## 설치 및 설정

### 1. ffmpeg 설치

**macOS**:
```bash
brew install ffmpeg
```

**Ubuntu**:
```bash
sudo apt update
sudo apt install ffmpeg
```

**확인**:
```bash
ffmpeg -version
```

### 2. 환경변수 설정

`.env` 파일에 추가:
```bash
# OpenAI API (기존)
OPENAI_API_KEY=your-api-key

# 파일 저장 경로 (기존)
UPLOAD_BASE_DIR=/absolute/path/to/uploads
```

### 3. 디렉토리 생성

```bash
mkdir -p /path/to/uploads/subtitles
```

### 4. 데이터베이스 마이그레이션

```sql
-- video_subtitles 테이블 생성
CREATE TABLE video_subtitles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    subtitle_type VARCHAR(20) NOT NULL,
    language VARCHAR(10) NOT NULL,
    file_path VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- video_subtitle_segments 테이블 생성
CREATE TABLE video_subtitle_segments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subtitle_id UUID NOT NULL REFERENCES video_subtitles(id) ON DELETE CASCADE,
    sequence_number INTEGER NOT NULL,
    start_time_ms BIGINT NOT NULL,
    end_time_ms BIGINT NOT NULL,
    text TEXT NOT NULL,
    confidence DECIMAL(5, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스 생성
CREATE INDEX idx_video_subtitles_document_id ON video_subtitles(document_id);
CREATE INDEX idx_video_subtitle_segments_subtitle_id ON video_subtitle_segments(subtitle_id);
CREATE INDEX idx_video_subtitle_segments_sequence ON video_subtitle_segments(subtitle_id, sequence_number);
```

---

## 테스트

### cURL 예시

**STT 처리**:
```bash
curl -X POST http://localhost:8000/api/ai/video/stt \
  -H "Content-Type: application/json" \
  -d '{
    "video_document_id": "123e4567-e89b-12d3-a456-426614174000",
    "source_language": "ko"
  }'
```

**자막 번역**:
```bash
curl -X POST http://localhost:8000/api/ai/video/translate \
  -H "Content-Type: application/json" \
  -d '{
    "video_document_id": "123e4567-e89b-12d3-a456-426614174000",
    "document_ids": [],
    "source_language": "ko",
    "target_language": "en"
  }'
```

**자막 다운로드**:
```bash
curl -X GET http://localhost:8000/api/ai/video/subtitle/{subtitle_id}/download \
  -o subtitle.srt
```

---

## 에러 처리

### 일반적인 에러

**1. ffmpeg 설치 안 됨**:
```
FileNotFoundError: ffmpeg가 설치되지 않았습니다.
```
→ 해결: ffmpeg 설치 (`brew install ffmpeg`)

**2. 영상 파일 없음**:
```
FileNotFoundError: 영상 파일을 찾을 수 없습니다: /path/to/video.mp4
```
→ 해결: 파일 경로 확인 및 문서 업로드 확인

**3. 원본 자막 없음** (번역 시):
```
ValueError: 원본 자막을 찾을 수 없습니다. STT를 먼저 실행하세요.
```
→ 해결: STT 먼저 실행 (`POST /api/ai/video/stt`)

**4. OpenAI API 에러**:
```
RuntimeError: STT 처리 중 오류 발생: API key invalid
```
→ 해결: `.env` 파일의 `OPENAI_API_KEY` 확인

---

## 성능 고려사항

### 처리 시간

| 작업 | 예상 시간 | 비고 |
|-----|----------|------|
| STT (5분 영상) | 30초 ~ 1분 | Whisper API 속도에 따름 |
| 번역 (100개 세그먼트) | 1분 ~ 2분 | GPT-4o 호출 시간 |
| SRT 파일 생성 | < 1초 | 파일 I/O만 수행 |

### 최적화 방안

1. **병렬 번역**: 세그먼트를 배치로 묶어 병렬 처리 (추후 구현)
2. **캐싱**: 동일한 세그먼트 번역 결과 캐시
3. **청크 처리**: 긴 영상을 여러 부분으로 나눠 처리

---

## 추후 개선사항

- [ ] 세그먼트 병렬 번역 (현재: 순차 처리)
- [ ] VTT, ASS 등 다양한 자막 형식 지원
- [ ] 자막 편집 기능 (프론트엔드 연동)
- [ ] 자막 동기화 조정 (타임스탬프 수정)
- [ ] 다중 화자 인식 (Whisper speaker diarization)
- [ ] 자막 품질 점수 (번역 품질 평가)

---

## 참고 자료

- [OpenAI Whisper API](https://platform.openai.com/docs/guides/speech-to-text)
- [ffmpeg Documentation](https://ffmpeg.org/documentation.html)
- [SRT Format Specification](https://www.matroska.org/technical/subtitles.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
