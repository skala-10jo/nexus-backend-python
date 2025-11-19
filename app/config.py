"""
Configuration settings for the Python backend.
Loads environment variables and provides application settings.
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/nexus"

    # JWT (must match Java backend)
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"

    # OpenAI
    OPENAI_API_KEY: str

    # Server
    PYTHON_BACKEND_PORT: int = 8000
    LOG_LEVEL: str = "info"

    # File Storage
    UPLOAD_BASE_DIR: str

    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # Qdrant Vector Database
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "email_embeddings"

    class Config:
        env_file = ".env"
        case_sensitive = True

    @property
    def allowed_origins_list(self) -> List[str]:
        """Convert CORS allowed origins string to list."""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(',')]


# Global settings instance
settings = Settings()
