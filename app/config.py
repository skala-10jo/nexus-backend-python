"""
Configuration settings for the Python backend.
Loads environment variables and provides application settings.
"""
import os
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings


def _get_default_upload_dir() -> str:
    """
    Get default upload directory based on project structure.

    Supports:
    - Relative path from backend-python: ../backend-java/uploads/documents
    - Automatically resolves to absolute path

    Returns:
        Absolute path to upload directory
    """
    # backend-python/app/config.py -> backend-python -> final_project
    current_file = Path(__file__).resolve()
    backend_python_dir = current_file.parent.parent  # backend-python/
    project_root = backend_python_dir.parent  # final_project/

    # Java backend uploads directory
    upload_dir = project_root / "backend-java" / "uploads" / "documents"

    return str(upload_dir)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/nexus"

    # JWT (must match Java backend)
    JWT_SECRET: str
    # Support multiple algorithms as Java uses hmacShaKeyFor which auto-selects based on key length
    JWT_ALGORITHMS: List[str] = ["HS256", "HS384", "HS512"]

    # OpenAI
    OPENAI_API_KEY: str

    # Server
    PYTHON_BACKEND_PORT: int = 8000
    LOG_LEVEL: str = "info"

    # Java Backend URL (분산 환경에서 내부 통신용)
    JAVA_BACKEND_URL: str = "http://localhost:3000"

    # File Storage (optional - auto-detected from project structure if not set)
    UPLOAD_BASE_DIR: Optional[str] = None

    # CORS (환경변수명: CORS_ALLOWED_ORIGINS 또는 ALLOWED_ORIGINS)
    CORS_ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # Qdrant Vector Database
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_EMAIL_COLLECTION: str = "email_embeddings"
    QDRANT_BIZGUIDE_COLLECTION: str = "bizguide"

    # Azure Speech (TTS/STT - Korea Central)
    AZURE_SPEECH_KEY: str = ""
    AZURE_SPEECH_REGION: str = "koreacentral"

    # Azure Translator
    AZURE_TRANSLATOR_KEY: str
    AZURE_TRANSLATOR_ENDPOINT: str
    AZURE_TRANSLATOR_REGION: str

    # Azure Avatar (Southeast Asia)
    AZURE_AVATAR_SPEECH_KEY: str = ""
    AZURE_AVATAR_SPEECH_REGION: str = "southeastasia"

    class Config:
        env_file = ".env"
        case_sensitive = True

    @property
    def allowed_origins_list(self) -> List[str]:
        """Convert CORS allowed origins string to list."""
        return [origin.strip() for origin in self.CORS_ALLOWED_ORIGINS.split(',')]

    @property
    def upload_dir(self) -> str:
        """
        Get upload directory path.

        Returns environment variable if set, otherwise auto-detects from project structure.
        This ensures portability across different development environments and deployments.

        Returns:
            Absolute path to upload directory
        """
        if self.UPLOAD_BASE_DIR:
            # Use environment variable if explicitly set
            return self.UPLOAD_BASE_DIR
        # Auto-detect from project structure
        return _get_default_upload_dir()


# Global settings instance
settings = Settings()
