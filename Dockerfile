# =====================================================
# NEXUS Python AI 백엔드 - 멀티 스테이지 Dockerfile
# FastAPI + uvicorn
# AWS ECS/ECR 배포 최적화 (ARM64/AMD64 호환)
# =====================================================

# -----------------------------------------------------
# 1단계: 빌드 (의존성 설치)
# -----------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# 빌드 도구 설치 (일부 패키지 컴파일 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 가상환경 생성
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------
# 2단계: 프로덕션
# -----------------------------------------------------
FROM python:3.11-slim AS production

WORKDIR /app

# 런타임 필수 패키지 설치
# Azure Speech SDK requires: OpenSSL 3.x, libasound2, GStreamer (for audio processing)
# Debian 12 (Bookworm) uses OpenSSL 3.x - need libssl3 not libssl-dev
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ffmpeg \
    ca-certificates \
    libssl3 \
    libasound2 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    && rm -rf /var/lib/apt/lists/*

# 한국 시간대 설정
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 보안을 위한 non-root 사용자 생성
RUN groupadd -g 1001 appgroup && \
    useradd -r -u 1001 -g appgroup -d /app -s /sbin/nologin appuser

# 빌더에서 가상환경 복사
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 애플리케이션 코드 복사
COPY app ./app
COPY agent ./agent

# 초기화 스크립트 및 데이터 복사
COPY scripts/docker-init.py ./scripts/docker-init.py
COPY scripts/expressions.json ./scripts/expressions.json

# 업로드 디렉토리 생성 및 권한 설정
RUN mkdir -p /app/uploads && \
    chown -R appuser:appgroup /app

# non-root 사용자로 전환
USER appuser

# 환경변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHON_BACKEND_PORT=8000

# 8000 포트 노출
EXPOSE 8000

# 헬스체크 설정
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/ai/health || exit 1

# uvicorn 실행 (프로덕션 설정)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
