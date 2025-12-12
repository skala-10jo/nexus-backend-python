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
    small_talk,
    speaking_tutor,
    slack_agent,
    expression_match,
    expressions  # 랜덤 표현 API
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
app.include_router(small_talk.router, tags=["Small Talk"])  # 스몰토크 대화
app.include_router(speaking_tutor.router, prefix="/api/ai/speaking-tutor", tags=["Speaking Tutor AI"])
app.include_router(slack_agent.router, tags=["Slack Agent"])  # prefix already defined in slack_agent.py
app.include_router(expression_match.router, prefix="/api/ai", tags=["Expression Match AI"])
app.include_router(expressions.router, prefix="/api/ai", tags=["Expressions API"])


@app.on_event("startup")
async def startup_event():
    """Execute on application startup."""
    from app.core.qdrant_client import ensure_collection_exists

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
