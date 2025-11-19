# Video Translation Feature - 구현 완료 요약

## 구현 개요

OpenAI Whisper API를 활용한 영상 자막 STT 및 번역 기능을 Python Backend에 구현했습니다.

**구현 날짜**: 2025-01-17
**Agent 패턴**: BaseAgent 상속, process() 메서드 구현
**통합 방식**: 기존 TranslationService의 ContextEnhancedTranslationAgent 재사용

---

## 구현 파일 목록

### 1. Agent Layer (순수 AI 로직)

| 파일 | 경로 | 역할 | 코드 라인 수 |
|-----|------|------|-------------|
| **VideoSTTAgent** | `agent/video/stt_agent.py` | 영상 → STT (Whisper API) | ~180줄 |
| **SubtitleGeneratorAgent** | `agent/video/subtitle_generator_agent.py` | 세그먼트 → SRT 파일 | ~150줄 |
| **__init__.py** | `agent/video/__init__.py` | Agent 패키지 초기화 | ~10줄 |

**핵심 기능**:
- ✅ ffmpeg로 영상에서 오디오 추출 (MP3, 16kHz mono)
- ✅ OpenAI Whisper API 호출 (타임스탬프 포함)
- ✅ 밀리초 단위 타임스탬프 변환
- ✅ SRT 형식 자막 파일 생성 (`HH:MM:SS,mmm`)

### 2. Service Layer (비즈니스 로직)

| 파일 | 경로 | 역할 | 코드 라인 수 |
|-----|------|------|-------------|
| **VideoTranslationService** | `app/services/video_translation_service.py` | STT/번역/파일 생성 조율 | ~280줄 |

**핵심 기능**:
- ✅ STT 처리 및 DB 저장
- ✅ 컨텍스트 기반 세그먼트별 번역
- ✅ SRT 파일 생성 및 경로 저장
- ✅ 문서 조회 및 컨텍스트 텍스트 추출

### 3. Model Layer (데이터베이스)

| 파일 | 경로 | 역할 | 테이블 |
|-----|------|------|-------|
| **VideoSubtitle** | `app/models/video_subtitle.py` | 자막 메타데이터 | `video_subtitles` |
| **VideoSubtitleSegment** | `app/models/video_subtitle.py` | 자막 세그먼트 | `video_subtitle_segments` |
| **SubtitleType (Enum)** | `app/models/video_subtitle.py` | 자막 타입 | ORIGINAL/TRANSLATED |

**데이터베이스 스키마**:
```sql
-- video_subtitles (자막 메타데이터)
id UUID PRIMARY KEY
document_id UUID (FK → documents)
subtitle_type VARCHAR(20) (ORIGINAL/TRANSLATED)
language VARCHAR(10) (ko, en, ja, vi)
file_path VARCHAR(500) (SRT 파일 경로)
created_at, updated_at TIMESTAMP

-- video_subtitle_segments (세그먼트)
id UUID PRIMARY KEY
subtitle_id UUID (FK → video_subtitles)
sequence_number INTEGER (1부터 시작)
start_time_ms BIGINT (시작 시간, 밀리초)
end_time_ms BIGINT (종료 시간, 밀리초)
text TEXT (자막 텍스트)
confidence DECIMAL(5,4) (신뢰도)
created_at TIMESTAMP
```

### 4. Schema Layer (Pydantic 검증)

| 파일 | 경로 | 역할 | 스키마 수 |
|-----|------|------|----------|
| **VideoTranslationSchemas** | `app/schemas/video_translation.py` | 요청/응답 검증 | 5개 스키마 |

**스키마 목록**:
1. `VideoSTTRequest` - STT 요청
2. `VideoTranslationRequest` - 번역 요청
3. `VideoSTTResponse` - STT 응답
4. `VideoTranslationResponse` - 번역 응답
5. `SubtitleDownloadResponse` - 다운로드 정보

### 5. API Layer (엔드포인트)

| 파일 | 경로 | 엔드포인트 수 | 역할 |
|-----|------|--------------|------|
| **VideoTranslationAPI** | `app/api/video_translation.py` | 4개 | HTTP 엔드포인트 |

**API 목록**:
1. `POST /api/ai/video/stt` - STT 처리
2. `POST /api/ai/video/translate` - 자막 번역
3. `GET /api/ai/video/subtitle/{id}/download` - SRT 파일 다운로드
4. `GET /api/ai/video/subtitle/{id}/info` - 자막 정보 조회

### 6. 마이그레이션 & 문서

| 파일 | 경로 | 역할 |
|-----|------|------|
| **SQL Migration** | `migrations/video_subtitle_tables.sql` | DB 스키마 생성 |
| **README** | `VIDEO_TRANSLATION_README.md` | 상세 가이드 |
| **Summary** | `VIDEO_IMPLEMENTATION_SUMMARY.md` | 구현 요약 (본 문서) |

---

## 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Vue 3)                        │
│                     영상 업로드 & 자막 관리                     │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─ POST /api/ai/video/stt
             ├─ POST /api/ai/video/translate
             └─ GET  /api/ai/video/subtitle/{id}/download
             │
┌────────────▼────────────────────────────────────────────────┐
│                   API Layer (FastAPI)                        │
│                app/api/video_translation.py                  │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│                  Service Layer (Business Logic)              │
│              app/services/video_translation_service.py       │
│                                                              │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────┐  │
│  │ process_stt  │  │ process_      │  │ generate_       │  │
│  │              │  │ translation   │  │ subtitle_file   │  │
│  └──────┬───────┘  └───────┬───────┘  └────────┬────────┘  │
└─────────┼──────────────────┼──────────────────┼────────────┘
          │                  │                  │
┌─────────▼──────────┐ ┌─────▼──────────┐ ┌────▼──────────┐
│  VideoSTTAgent     │ │ ContextEnhanced │ │ Subtitle      │
│  (Whisper API)     │ │ TranslationAgent│ │ GeneratorAgent│
│                    │ │ (GPT-4o)        │ │ (SRT 생성)     │
└─────────┬──────────┘ └────────┬────────┘ └───────┬───────┘
          │                     │                  │
          ├─ ffmpeg (MP3 추출)   │                  │
          ├─ Whisper API 호출   ├─ GPT-4o 번역     ├─ SRT 파일 생성
          └─ 타임스탬프 파싱      └─ 컨텍스트 활용   └─ 파일 저장
          │                     │                  │
┌─────────▼──────────────────────▼──────────────────▼───────┐
│                   Database (PostgreSQL)                    │
│                                                            │
│  ┌──────────────────┐  ┌─────────────────────────────┐    │
│  │ video_subtitles  │  │ video_subtitle_segments     │    │
│  │  - id            │  │  - id                       │    │
│  │  - document_id   │  │  - subtitle_id              │    │
│  │  - subtitle_type │  │  - sequence_number          │    │
│  │  - language      │  │  - start_time_ms            │    │
│  │  - file_path     │  │  - end_time_ms              │    │
│  └──────────────────┘  │  - text                     │    │
│                        │  - confidence               │    │
│                        └─────────────────────────────┘    │
└────────────────────────────────────────────────────────────┘
```

---

## 데이터 플로우

### 1. STT 처리 플로우

```
1. 프론트엔드: 영상 업로드 (Java Backend → documents 테이블)
   ↓
2. 프론트엔드: POST /api/ai/video/stt
   {
     "video_document_id": "123...",
     "source_language": "ko"
   }
   ↓
3. VideoTranslationService.process_stt()
   ├─ Document 조회 (file_path 확인)
   ├─ VideoSTTAgent.process()
   │  ├─ ffmpeg로 MP3 추출
   │  ├─ Whisper API 호출
   │  └─ 타임스탬프 세그먼트 반환
   ├─ DB 저장:
   │  ├─ VideoSubtitle (ORIGINAL, ko)
   │  └─ VideoSubtitleSegment (각 세그먼트)
   └─ 응답 반환
   ↓
4. 프론트엔드: 자막 세그먼트 표시
```

### 2. 번역 처리 플로우

```
1. 프론트엔드: POST /api/ai/video/translate
   {
     "video_document_id": "123...",
     "document_ids": ["doc1", "doc2"],
     "source_language": "ko",
     "target_language": "en"
   }
   ↓
2. VideoTranslationService.process_translation()
   ├─ 원본 자막 조회 (VideoSubtitle + segments)
   ├─ 컨텍스트 문서 조회 (Document.contents)
   ├─ 각 세그먼트 번역 (for loop):
   │  └─ ContextEnhancedTranslationAgent.process()
   │     ├─ 원본 텍스트
   │     ├─ 컨텍스트 문서
   │     └─ GPT-4o 번역
   ├─ DB 저장:
   │  ├─ VideoSubtitle (TRANSLATED, en)
   │  └─ VideoSubtitleSegment (번역된 텍스트)
   └─ 응답 반환
   ↓
3. 프론트엔드: 번역된 자막 표시
```

### 3. 자막 다운로드 플로우

```
1. 프론트엔드: GET /api/ai/video/subtitle/{subtitle_id}/download
   ↓
2. VideoTranslationService.generate_subtitle_file()
   ├─ VideoSubtitle 조회
   ├─ 세그먼트 조회 (sequence_number 순서)
   ├─ SubtitleGeneratorAgent.process()
   │  ├─ 타임스탬프 형식 변환 (ms → HH:MM:SS,mmm)
   │  ├─ SRT 형식 생성
   │  └─ 파일 저장 (uploads/subtitles/)
   ├─ DB 업데이트 (file_path)
   └─ FileResponse 반환
   ↓
3. 프론트엔드: SRT 파일 다운로드
```

---

## 핵심 기능 상세

### 1. VideoSTTAgent

**입력**:
- `video_file_path`: 영상 파일 경로 (MP4/AVI/MOV)
- `source_language`: 음성 언어 코드 (ko, en, ja, vi)

**처리 과정**:
1. ffmpeg로 영상 → MP3 추출 (16kHz mono, 64kbps)
2. OpenAI Whisper API 호출:
   - 모델: `whisper-1`
   - response_format: `verbose_json` (타임스탬프 포함)
   - timestamp_granularities: `["segment"]`
3. 응답 파싱 및 변환:
   - 초 단위 → 밀리초 변환
   - avg_logprob → confidence 매핑

**출력**:
```python
[
    {
        "sequence_number": 1,
        "start_time_ms": 0,
        "end_time_ms": 3500,
        "text": "안녕하세요...",
        "confidence": 0.95
    },
    ...
]
```

**에러 처리**:
- ffmpeg 미설치: `FileNotFoundError`
- 영상 파일 없음: `FileNotFoundError`
- Whisper API 오류: `RuntimeError`

### 2. SubtitleGeneratorAgent

**입력**:
- `segments`: 타임스탬프 세그먼트 리스트
- `output_path`: 출력 파일 경로
- `subtitle_type`: 자막 타입 (original/translated)

**처리 과정**:
1. 세그먼트 유효성 검증:
   - 필수 필드 확인 (sequence_number, start_time_ms, end_time_ms, text)
   - 타임스탬프 검증 (start < end, >= 0)
2. SRT 형식 변환:
   - 밀리초 → `HH:MM:SS,mmm` 형식
   - 세그먼트 구분 (빈 줄)
3. 파일 저장 (UTF-8 인코딩)

**출력 (SRT 형식)**:
```srt
1
00:00:00,000 --> 00:00:03,500
안녕하세요, 오늘은 인공지능에 대해 말씀드리겠습니다.

2
00:00:03,500 --> 00:00:07,200
인공지능은 컴퓨터가 인간처럼 학습하고 추론하는 기술입니다.
```

### 3. ContextEnhancedTranslationAgent (재사용)

**기존 위치**: `agent/translate/context_enhanced_translation_agent.py`

**사용 방식**:
```python
translated_text = await self.context_translator.process(
    text=original_segment.text,
    source_lang="ko",
    target_lang="en",
    context=context_text,      # 프로젝트 문서 컨텍스트
    glossary_terms=[],          # 용어집 (추후 확장)
    detected_terms=[]
)
```

**장점**:
- 기존 코드 재사용 (중복 방지)
- 프로젝트 컨텍스트 활용
- 일관된 번역 품질

---

## 테스트 방법

### 1. 사전 준비

**ffmpeg 설치 확인**:
```bash
ffmpeg -version
```

**환경변수 확인** (`.env`):
```bash
OPENAI_API_KEY=sk-...
UPLOAD_BASE_DIR=/absolute/path/to/uploads
```

**디렉토리 생성**:
```bash
mkdir -p /path/to/uploads/subtitles
```

**DB 마이그레이션**:
```bash
psql -U postgres -d nexus -f migrations/video_subtitle_tables.sql
```

### 2. API 테스트

**Step 1: 영상 업로드** (Java Backend):
```bash
curl -X POST http://localhost:3000/api/documents/upload \
  -F "file=@test_video.mp4" \
  -F "userId=00000000-0000-0000-0000-000000000000"

# Response:
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "filename": "test_video.mp4",
  ...
}
```

**Step 2: STT 처리** (Python Backend):
```bash
curl -X POST http://localhost:8000/api/ai/video/stt \
  -H "Content-Type: application/json" \
  -d '{
    "video_document_id": "123e4567-e89b-12d3-a456-426614174000",
    "source_language": "ko"
  }'

# Response:
{
  "subtitle_id": "789...",
  "segments": [
    {
      "sequence_number": 1,
      "start_time_ms": 0,
      "end_time_ms": 3500,
      "text": "안녕하세요...",
      "confidence": 0.95
    }
  ],
  "total_segments": 10
}
```

**Step 3: 자막 번역**:
```bash
curl -X POST http://localhost:8000/api/ai/video/translate \
  -H "Content-Type: application/json" \
  -d '{
    "video_document_id": "123e4567-e89b-12d3-a456-426614174000",
    "document_ids": [],
    "source_language": "ko",
    "target_language": "en"
  }'

# Response:
{
  "subtitle_id": "456...",
  "segments": [
    {
      "sequence_number": 1,
      "text": "Hello, today I will talk about artificial intelligence."
    }
  ],
  "context_used": false
}
```

**Step 4: 자막 다운로드**:
```bash
curl -X GET http://localhost:8000/api/ai/video/subtitle/789.../download \
  -o subtitle_ko_original.srt

curl -X GET http://localhost:8000/api/ai/video/subtitle/456.../download \
  -o subtitle_en_translated.srt
```

---

## 성능 및 제한사항

### 처리 시간

| 작업 | 영상 길이 | 예상 시간 | 비고 |
|-----|----------|----------|------|
| STT | 1분 | 10~15초 | Whisper API 속도 |
| STT | 5분 | 30~60초 | |
| STT | 10분 | 1~2분 | |
| 번역 (50 세그먼트) | - | 30~60초 | 순차 처리 |
| 번역 (100 세그먼트) | - | 1~2분 | |
| SRT 파일 생성 | - | < 1초 | 파일 I/O만 |

### 제한사항

1. **순차 번역**: 현재 각 세그먼트를 순차적으로 번역
   - 개선 방안: 배치 처리 또는 병렬 번역

2. **ffmpeg 의존성**: 시스템에 ffmpeg 설치 필수
   - 개선 방안: Python 라이브러리로 대체 (moviepy 등)

3. **파일 크기**: 영상 파일 크기에 따라 처리 시간 증가
   - 개선 방안: 청크 단위 처리

4. **타임아웃**: ffmpeg 처리 시간 5분 제한
   - 개선 방안: 긴 영상은 분할 처리

---

## 보안 고려사항

1. **파일 검증**: 영상 파일 타입 및 크기 검증 (추후 구현)
2. **경로 탐색 방지**: 파일 경로 검증 (상대 경로 금지)
3. **인증 연동**: JWT 인증 추가 (추후 구현)
4. **API 속도 제한**: OpenAI API 호출 제한 고려

---

## 추후 개선사항

### 단기 (1~2주)
- [ ] JWT 인증 연동 (현재 임시 사용자 ID 사용)
- [ ] 파일 업로드 검증 (파일 타입, 크기)
- [ ] 에러 처리 강화 (재시도 로직)

### 중기 (1개월)
- [ ] 세그먼트 병렬 번역 (성능 개선)
- [ ] 용어집 연동 (번역 품질 향상)
- [ ] 자막 편집 기능 (타임스탬프 조정)
- [ ] VTT, ASS 등 다른 자막 형식 지원

### 장기 (2~3개월)
- [ ] 다중 화자 인식 (Whisper diarization)
- [ ] 자막 품질 평가 (번역 점수)
- [ ] 실시간 STT (WebSocket)
- [ ] 영상 미리보기와 자막 동기화 (프론트엔드)

---

## 참고 문서

1. **구현 가이드**: `VIDEO_TRANSLATION_README.md`
2. **마이그레이션**: `migrations/video_subtitle_tables.sql`
3. **Agent 패턴**: `agent/base_agent.py`
4. **기존 번역 서비스**: `app/services/translation_service.py`
5. **OpenAI Whisper**: https://platform.openai.com/docs/guides/speech-to-text

---

## 구현 통계

| 항목 | 수량 | 비고 |
|-----|------|------|
| **총 파일 수** | 10개 | 코드 + 문서 + 마이그레이션 |
| **총 코드 라인 수** | ~900줄 | 주석 포함 |
| **Agent 수** | 2개 (+ 1개 재사용) | VideoSTT, SubtitleGenerator |
| **API 엔드포인트** | 4개 | STT, 번역, 다운로드, 정보 조회 |
| **DB 테이블** | 2개 | video_subtitles, segments |
| **Pydantic 스키마** | 5개 | 요청/응답 검증 |

---

## 코드 품질 체크리스트

- [x] BaseAgent 패턴 준수
- [x] 타입 힌팅 완료 (Python)
- [x] Docstring 작성 (모든 메서드)
- [x] 에러 처리 구현
- [x] 로깅 추가 (INFO, ERROR 레벨)
- [x] Pydantic 검증 적용
- [x] DB 모델 관계 설정
- [x] 마이그레이션 스크립트 작성
- [x] API 문서화 (FastAPI 자동 생성)
- [x] README 작성

---

## 팀 협업 가이드

### 프론트엔드 개발자

**필요한 정보**:
- API 엔드포인트: `/api/ai/video/stt`, `/api/ai/video/translate`, `/api/ai/video/subtitle/{id}/download`
- 요청/응답 스키마: `VIDEO_TRANSLATION_README.md` 참고
- 에러 코드: 400 (잘못된 요청), 404 (파일 없음), 500 (서버 오류)

**구현 예시**:
```javascript
// STT 요청
const response = await fetch('/api/ai/video/stt', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    video_document_id: videoId,
    source_language: 'ko'
  })
})
const result = await response.json()
```

### Java Backend 개발자

**연동 포인트**:
- 영상 문서 업로드 후 `document_id` 전달
- 자막 다운로드 링크 생성 시 Python Backend URL 사용

**예시**:
```java
String pythonBackendUrl = "http://localhost:8000";
String downloadUrl = pythonBackendUrl + "/api/ai/video/subtitle/" + subtitleId + "/download";
```

---

## 완료 일자

**구현 완료**: 2025-01-17
**문서 작성**: 2025-01-17
**코드 리뷰**: 대기 중
**배포**: 대기 중
