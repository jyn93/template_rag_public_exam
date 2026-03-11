"""Application settings loaded from environment variables via Pydantic Settings."""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "LLMProvider", "RetrieverType", "get_settings"]


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class RetrieverType(str, Enum):
    """Supported retriever strategies."""

    DENSE = "dense"
    HYBRID = "hybrid"


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_name: str = "RAG Oposiciones"
    debug: bool = False
    log_level: str = "INFO"

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    llm_model: str = "claude-3-5-sonnet-20241022"
    llm_temperature: float = 0.1
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "oposiciones_temario"

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    database_url: str = (
        "postgresql+asyncpg://opositor:opositor123@localhost:5432/oposiciones_db"
    )

    # ── MinIO ─────────────────────────────────────────────────────────────────
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin123"
    minio_bucket: str = "temario-docs"

    # ── RAG ───────────────────────────────────────────────────────────────────
    retriever_type: RetrieverType = RetrieverType.HYBRID
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Langfuse ──────────────────────────────────────────────────────────────
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance.

    Returns:
        Singleton Settings instance loaded from environment.
    """
    return Settings()
