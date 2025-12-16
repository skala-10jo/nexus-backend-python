"""
FastAPI application entry point for Python AI backend.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api import (
    glossary,
    translate,
    mail_agent,
    scenarios,
    conversations,
    video_translation,
    azure_speech,
    expression_speech,
    azure_avatar,
    pronunciation,
    voice_stt_ws,
    voice_stt,
    voice_translate,
    voice_tts,
    voice_realtime,
    voice_stt_stream,  # STT 전용 WebSocket (번역 없음)
    speaking_tutor,
    slack_agent,
    expressions,  # 랜덤 표현 API
    document_process  # 문서 처리 (텍스트 추출 + 요약)
)

import logging

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="NEXUS Python AI Backend",
    description="AI/ML backend for NEXUS translation platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(glossary.router, prefix="/api/ai", tags=["Glossary AI"])
app.include_router(translate.router, tags=["Translation AI"])
app.include_router(mail_agent.router)  # prefix already defined in mail_agent.py
app.include_router(scenarios.router, prefix="/api/ai/scenarios", tags=["Scenarios AI"])
app.include_router(conversations.router, tags=["Conversations AI"])  # prefix already defined
app.include_router(video_translation.router, tags=["Video Translation AI"])
app.include_router(azure_speech.router, prefix="/api/ai", tags=["Azure Speech"])
app.include_router(expression_speech.router, prefix="/api/ai", tags=["Expression Speech AI"])
app.include_router(azure_avatar.router, prefix="/api/ai", tags=["Azure Avatar"])
app.include_router(pronunciation.router, prefix="/api/ai", tags=["Pronunciation Assessment"])
app.include_router(voice_stt_ws.router, tags=["Voice STT WebSocket"])  # WebSocket STT
app.include_router(voice_stt.router, prefix="/api/ai/voice", tags=["Voice STT REST"])  # REST STT
app.include_router(voice_translate.router, prefix="/api/ai/voice", tags=["Voice Translation API"])  # Translation API
app.include_router(voice_tts.router, prefix="/api/ai/voice", tags=["Voice TTS API"])  # TTS API
app.include_router(voice_realtime.router, tags=["Voice Realtime WebSocket"])  # 실시간 음성 번역 (자동감지 + 번역)
app.include_router(voice_stt_stream.router, tags=["Voice STT Stream WebSocket"])  # STT 전용 (번역 없음, 회화연습용)
app.include_router(speaking_tutor.router, prefix="/api/ai/speaking-tutor", tags=["Speaking Tutor AI"])
app.include_router(slack_agent.router, tags=["Slack Agent"])  # prefix already defined in slack_agent.py
app.include_router(expressions.router, prefix="/api/ai", tags=["Expressions API"])
app.include_router(document_process.router, prefix="/api/ai", tags=["Document Process"])


@app.on_event("startup")
async def startup_event():
    """Execute on application startup."""
    import asyncio
    from app.core.qdrant_client import ensure_collection_exists
    from app.core.azure_speech_token_manager import AzureSpeechTokenManager

    logger.info("NEXUS Python AI Backend starting...")
    logger.info(f"Port: {settings.PYTHON_BACKEND_PORT}")
    logger.info(f"CORS origins: {settings.ALLOWED_ORIGINS}")
    logger.info(f"Database: {settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else 'configured'}")

    # Initialize Qdrant collection
    try:
        ensure_collection_exists()
        logger.info(f"Qdrant collection ready: {settings.QDRANT_EMAIL_COLLECTION}")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {str(e)}")

    # Pre-fetch Azure Speech token (background task)
    # 앱 시작 시 토큰을 미리 발급하여 첫 음성 인식 시 지연 방지
    async def prefetch_azure_token():
        try:
            token_manager = AzureSpeechTokenManager.get_instance()
            await token_manager.prefetch_token()
        except Exception as e:
            logger.warning(f"Azure Speech token prefetch failed (non-critical): {e}")

    # 백그라운드에서 토큰 사전 발급 (앱 시작을 블로킹하지 않음)
    asyncio.create_task(prefetch_azure_token())


@app.on_event("shutdown")
async def shutdown_event():
    """Execute on application shutdown."""
    # aiohttp 세션 정리
    from agent.stt_translation.translation_agent import TranslationAgent
    await TranslationAgent.close_session()
    logger.info("NEXUS Python AI Backend shutting down...")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "service": "NEXUS Python AI Backend",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/api/ai/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Python AI backend is running"
    }
