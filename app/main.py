"""
FastAPI application entry point for Python AI backend.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api import glossary, translate, mail_agent
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


@app.on_event("startup")
async def startup_event():
    """Execute on application startup."""
    logger.info("🚀 NEXUS Python AI Backend starting...")
    logger.info(f"📍 Port: {settings.PYTHON_BACKEND_PORT}")
    logger.info(f"🔗 CORS origins: {settings.ALLOWED_ORIGINS}")
    logger.info(f"🗄️  Database: {settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else 'configured'}")


@app.on_event("shutdown")
async def shutdown_event():
    """Execute on application shutdown."""
    logger.info("👋 NEXUS Python AI Backend shutting down...")


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
