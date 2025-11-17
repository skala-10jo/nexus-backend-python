# AI 메일 Agent 사용 가이드

## 📋 목차
1. [개요](#개요)
2. [시스템 아키텍처](#시스템-아키텍처)
3. [Qdrant 설정](#qdrant-설정)
4. [기능 설명](#기능-설명)
5. [API 엔드포인트](#api-엔드포인트)
6. [프론트엔드 사용법](#프론트엔드-사용법)
7. [트러블슈팅](#트러블슈팅)

---

## 개요

AI 메일 Agent는 Outlook 메일을 **Qdrant 벡터 데이터베이스**에 저장하고, **자연어 검색**을 제공하는 시스템입니다.

### 핵심 기능
- ✅ 메일 자동 임베딩 생성 (OpenAI text-embedding-ada-002)
- ✅ 자연어 기반 메일 검색 (RAG)
- ✅ 대화형 메일 검색 챗봇 (GPT-4o)
- ✅ 하이브리드 검색 (필터 + 벡터 유사도)

### 기술 스택
| 구분 | 기술 | 용도 |
|-----|------|------|
| 벡터 DB | Qdrant | 임베딩 저장 및 벡터 검색 |
| 임베딩 모델 | OpenAI text-embedding-ada-002 | 1536차원 벡터 생성 |
| LLM | GPT-4o | 쿼리 파싱 및 답변 생성 |
| 백엔드 | FastAPI | REST API |
| 프론트엔드 | Vue 3 | 채팅 UI |

---

## 시스템 아키텍처

### 전체 플로우

```
[Outlook 메일]
      ↓
[메일 동기화]
      ↓
[PostgreSQL]
      ↓
[임베딩 생성]
      ↓
[Qdrant 저장]
      ↓
[자연어 검색]
```

### Agent 구조

```
app/
├── api/
│   └── mail_agent.py          # REST 엔드포인트
├── services/
│   └── mail_agent_service.py  # 비즈니스 로직
├── models/
│   └── email.py               # DB 모델
└── schemas/
    └── mail_agent.py          # Pydantic 스키마

agent/
├── base_agent.py              # 베이스 클래스
└── mail/
    ├── embedding_agent.py     # 임베딩 생성
    ├── search_agent.py        # 벡터 검색
    └── query_agent.py         # 쿼리 파싱
```

### 데이터 흐름

#### 1. 임베딩 생성
```
메일 본문 → 청킹 (500자) → OpenAI Embedding → Qdrant 저장
```

#### 2. 검색
```
사용자 쿼리 → GPT-4o 파싱 → OpenAI Embedding → Qdrant 검색 → 결과 반환
```

---

## Qdrant 설정

### 1. Qdrant 실행 (Docker)

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### 2. 환경변수 설정

```bash
# backend-python/.env
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=email_chunks
OPENAI_API_KEY=your-openai-api-key
```

### 3. 컬렉션 생성 (자동)

FastAPI 앱 시작 시 자동으로 컬렉션이 생성됩니다:

```python
# app/main.py
@app.on_event("startup")
async def startup_event():
    ensure_qdrant_collection()
```

### 4. 컬렉션 스키마

```python
{
  "vectors": {
    "size": 1536,  # OpenAI text-embedding-ada-002
    "distance": "Cosine"  # 코사인 유사도
  },
  "payload_schema": {
    "email_id": "keyword",     # 메일 ID
    "user_id": "keyword",      # 사용자 ID
    "chunk_index": "integer",  # 청크 번호
    "chunk_text": "text",      # 청크 텍스트
    "subject": "text",         # 메일 제목
    "folder": "keyword",       # Inbox/SentItems
    "date": "datetime",        # 메일 날짜
    "from_name": "text",       # 보낸이
    "to_recipients": "text"    # 받는이
  }
}
```

---

## 기능 설명

### 1. 임베딩 생성 (EmbeddingAgent)

**위치**: `agent/mail/embedding_agent.py`

**역할**:
- 메일 본문을 500자 단위로 청킹
- OpenAI text-embedding-ada-002로 임베딩 생성
- Qdrant에 벡터 저장

**플로우**:
```python
1. 메일 본문 → 청킹 (500자, 100자 오버랩)
2. 각 청크 → OpenAI Embedding (1536차원)
3. Qdrant에 저장:
   - vector: [0.123, -0.456, ...]
   - payload: {email_id, user_id, chunk_text, ...}
```

**예시**:
```python
email_data = {
    'email_id': 'uuid',
    'user_id': 'uuid',
    'subject': '프로젝트 회의',
    'body': '내일 오후 3시에 회의가 있습니다...',
    'folder': 'Inbox',
    'date': '2025-01-17'
}

result = await embedding_agent.process(email_data)
# {'status': 'success', 'chunks_created': 3, 'email_id': 'uuid'}
```

### 2. 벡터 검색 (SearchAgent)

**위치**: `agent/mail/search_agent.py`

**역할**:
- 사용자 쿼리를 임베딩으로 변환
- Qdrant에서 유사 메일 검색
- 필터링 (user_id, folder, date)

**하이브리드 검색**:
```python
1. 쿼리 임베딩: "프로젝트 일정" → [0.789, -0.234, ...]
2. Qdrant 필터:
   - user_id = 'current-user'
   - folder = 'Inbox'  (선택)
   - date >= '2025-01-01'  (선택)
3. 벡터 검색: 코사인 유사도 계산
4. 상위 K개 반환 (유사도 ≥ 0.7)
```

**예시**:
```python
results = await search_agent.process(
    query="프로젝트 일정 회의",
    user_id="uuid",
    db=db,
    top_k=10,
    folder="Inbox",
    date_from="2025-01-01"
)

# [
#   {
#     'email_id': 'uuid',
#     'subject': '프로젝트 일정 회의',
#     'similarity': 0.92,
#     'matched_chunk': '제목: 프로젝트 일정...'
#   },
#   ...
# ]
```

### 3. 쿼리 파싱 (QueryAgent)

**위치**: `agent/mail/query_agent.py`

**역할**:
- 자연어 쿼리 → 구조화된 검색 파라미터
- GPT-4o로 의도 파악

**파싱 예시**:
```
입력: "어제 받은 프로젝트 관련 메일 찾아줘"
출력:
{
  "query": "프로젝트",
  "folder": "Inbox",
  "date_from": "2025-01-16",
  "needs_search": true,
  "response": "어제 받은 프로젝트 관련 메일을 검색하겠습니다."
}
```

**날짜 키워드 자동 해석**:
| 키워드 | 해석 |
|--------|------|
| 오늘 | 2025-01-17 |
| 어제 | 2025-01-16 |
| 이번 주 | date_from = 2025-01-13 (월요일) |
| 지난주 | date_from/to = 2025-01-06 ~ 2025-01-12 |
| 이번 달 | date_from = 2025-01-01 |

---

## API 엔드포인트

### 1. 단일 메일 임베딩 생성

```http
POST /api/ai/mail/embeddings/generate
Content-Type: application/json

{
  "email_id": "3be76c83-5473-4ebe-a8bf-8474c059ac45"
}
```

**응답**:
```json
{
  "status": "success",
  "chunks_created": 3,
  "email_id": "3be76c83-5473-4ebe-a8bf-8474c059ac45"
}
```

**응답 (이미 존재)**:
```json
{
  "status": "skipped",
  "reason": "Already has embeddings",
  "email_id": "3be76c83-5473-4ebe-a8bf-8474c059ac45"
}
```

### 2. 일괄 임베딩 생성

```http
POST /api/ai/mail/embeddings/batch
Content-Type: application/json

{
  "user_id": "user-uuid"
}
```

**응답**:
```json
{
  "status": "success",
  "total": 100,
  "processed": 95,
  "skipped": 3,
  "failed": 2
}
```

### 3. 메일 검색

```http
POST /api/ai/mail/search
Content-Type: application/json

{
  "query": "프로젝트 일정 회의",
  "user_id": "user-uuid",
  "top_k": 10,
  "folder": "Inbox",
  "date_from": "2025-01-01"
}
```

**응답**:
```json
{
  "success": true,
  "data": [
    {
      "email_id": "uuid",
      "subject": "프로젝트 일정 회의",
      "from_name": "홍길동",
      "to_recipients": "me@example.com",
      "folder": "Inbox",
      "date": "2025-01-15T10:30:00Z",
      "similarity": 0.92,
      "matched_chunk": "제목: 프로젝트 일정 회의\n\n내일 오후 3시..."
    }
  ],
  "count": 5
}
```

### 4. 대화형 검색 (챗봇)

```http
POST /api/ai/mail/chat
Content-Type: application/json

{
  "message": "어제 받은 프로젝트 관련 메일 찾아줘",
  "user_id": "user-uuid",
  "conversation_history": []
}
```

**응답**:
```json
{
  "query": "프로젝트",
  "folder": "Inbox",
  "date_from": "2025-01-16",
  "needs_search": true,
  "response": "어제 받은 프로젝트 관련 메일을 검색하겠습니다.",
  "search_results": [
    {
      "email_id": "uuid",
      "subject": "프로젝트 킥오프 미팅",
      "similarity": 0.89,
      ...
    }
  ]
}
```

---

## 프론트엔드 사용법

### 1. 자동 임베딩 생성

**Outlook 연동 시**:
```javascript
// Mail.vue
const connectOutlook = async () => {
  // ... 인증 로직

  // 인증 완료 후 자동으로 전체 임베딩 생성
  authCheckInterval = setInterval(async () => {
    const status = await checkAuthComplete()
    if (status) {
      await loadEmails()
      await generateAllEmbeddings()  // ⭐ 자동 임베딩
    }
  }, 5000)
}
```

**동기화 버튼 클릭 시**:
```javascript
const syncMails = async () => {
  await api.post('/outlook/sync')
  await loadEmails()
  await generateAllEmbeddings()  // ⭐ 신규 메일 임베딩
}
```

### 2. 채팅 인터페이스

**위치**: `frontend/src/views/collaboration/Mail.vue`

**UI 구성**:
```
┌─────────────────────────────────────┐
│  메일 리스트                  [챗봇]│
├─────────────────────────────────────┤
│                                     │
│  받은편지함 | 보낸편지함            │
│                                     │
│  📧 프로젝트 킥오프 미팅            │
│  📧 주간 업무 보고                  │
│                                     │
└─────────────────────────────────────┘

[챗봇] 클릭 시 →

┌──────────────────┬──────────────────┐
│  메일 리스트     │ AI 메일 Agent    │
│                  ├──────────────────┤
│  받은편지함      │ 사용자: 프로젝트 │
│                  │   관련 메일      │
│  📧 프로젝트...  │                  │
│  📧 주간 업무... │ AI: 검색하겠습... │
│                  │                  │
│                  │ 📧 프로젝트 킥... │
│                  │    유사도 92%     │
│                  │                  │
│                  │ [입력창]         │
└──────────────────┴──────────────────┘
```

**사용 예시**:
1. 우측 하단 플로팅 버튼 클릭
2. 채팅창 입력: "어제 받은 프로젝트 관련 메일 찾아줘"
3. AI가 자동으로 검색 수행
4. 검색 결과를 카드 형태로 표시
5. 카드 클릭 시 메일 상세 모달 열림
6. 메일 닫아도 채팅창은 유지됨

### 3. 반응형 레이아웃

채팅창이 열리면 메일 리스트가 자연스럽게 왼쪽으로 밀림:

```javascript
// 메인 콘텐츠 영역
<div
  class="flex-1 p-8 overflow-y-auto transition-all duration-300"
  :style="{ marginRight: showChatPanel ? '384px' : '0' }"
>
  <!-- 메일 리스트 -->
</div>

// 채팅 패널 (fixed, z-50)
<div class="fixed top-0 right-0 h-full w-96">
  <!-- 채팅 UI -->
</div>
```

---

## 트러블슈팅

### 1. Qdrant 연결 실패

**증상**:
```
Failed to connect to Qdrant: Connection refused
```

**해결**:
```bash
# Qdrant 실행 확인
docker ps | grep qdrant

# 재시작
docker run -p 6333:6333 qdrant/qdrant
```

### 2. OpenAI API 에러

**증상**:
```
openai.error.RateLimitError: You exceeded your current quota
```

**해결**:
- OpenAI API 키 확인
- Usage limit 확인 (https://platform.openai.com/usage)
- API 키 갱신

### 3. 임베딩 중복 생성

**증상**:
```json
{
  "status": "skipped",
  "reason": "Already has embeddings"
}
```

**설명**:
- 정상 동작입니다
- Qdrant에 이미 임베딩이 존재하여 스킵
- 중복 방지를 위한 체크

### 4. 검색 결과 없음

**체크리스트**:
1. 임베딩 생성 확인:
   ```bash
   # Qdrant UI 접속
   http://localhost:6333/dashboard

   # Collection 확인
   # email_chunks 컬렉션에 벡터가 있는지 확인
   ```

2. 필터 조건 확인:
   ```javascript
   // folder, date_from, date_to 조건이 너무 엄격하지 않은지 확인
   ```

3. 유사도 임계값:
   ```python
   # search_agent.py:129
   score_threshold=0.7  # 너무 높으면 0.5로 낮춤
   ```

### 5. 채팅창 422 에러

**증상**:
```
422 Unprocessable Entity
```

**원인**:
- `user_id`가 null

**해결**:
```javascript
// localStorage에서 user 객체 확인
const userStr = localStorage.getItem('user')
const userId = userStr ? JSON.parse(userStr).id : null

// 로그인 확인
if (!userId) {
  alert('로그인이 필요합니다')
  router.push('/login')
}
```

---

## 성능 최적화

### 1. 벡터 검색 성능

```python
# 상위 K개만 검색 (기본 10개)
top_k=10

# 필터 먼저 적용 후 벡터 검색
query_filter=models.Filter(must=[...])

# 유사도 임계값으로 조기 종료
score_threshold=0.7
```

### 2. 임베딩 생성 최적화

```python
# 청킹 크기 조정 (기본 500자)
chunk_size=500
chunk_overlap=100

# 배치 처리 (한 번에 여러 메일)
batch_generate_embeddings(user_id)
```

### 3. 캐싱 전략

```python
# TODO: Redis 캐싱 추가
# - 검색 결과 캐싱 (5분)
# - 임베딩 벡터 캐싱
```

---

## 다음 단계

### 계획된 기능

1. **Answer Agent** (우선순위 높음)
   - 검색 결과 기반 답변 생성
   - "회식 몇시지?" → "10시입니다"

2. **번역 Agent**
   - 메일 내용 번역
   - 다국어 지원

3. **요약 Agent**
   - 긴 메일 자동 요약
   - 메일 스레드 요약

4. **알림 Agent**
   - 중요 메일 자동 분류
   - 알림 우선순위 설정

---

## 참고 자료

### 공식 문서
- [Qdrant 공식 문서](https://qdrant.tech/documentation/)
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)

### 관련 파일
- `agent/mail/embedding_agent.py` - 임베딩 생성 로직
- `agent/mail/search_agent.py` - 벡터 검색 로직
- `agent/mail/query_agent.py` - 쿼리 파싱 로직
- `app/api/mail_agent.py` - REST API 엔드포인트
- `app/core/qdrant_client.py` - Qdrant 클라이언트

### 코드 예시
```bash
# 임베딩 생성 테스트
curl -X POST http://localhost:8000/api/ai/mail/embeddings/generate \
  -H "Content-Type: application/json" \
  -d '{"email_id": "your-email-id"}'

# 검색 테스트
curl -X POST http://localhost:8000/api/ai/mail/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "프로젝트 일정",
    "user_id": "your-user-id",
    "top_k": 5
  }'

# 챗봇 테스트
curl -X POST http://localhost:8000/api/ai/mail/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "어제 받은 메일 찾아줘",
    "user_id": "your-user-id"
  }'
```

---

**작성일**: 2025-01-17
**작성자**: NEXUS Team
**버전**: 1.0
