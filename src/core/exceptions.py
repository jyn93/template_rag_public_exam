"""Domain exception hierarchy for the RAG Oposiciones application."""

from __future__ import annotations

__all__ = [
    "RAGOposicionesError",
    "IngestionError",
    "RetrievalError",
    "GenerationError",
    "VectorStoreError",
    "StorageError",
    "DatabaseError",
    "ConfigurationError",
]


class RAGOposicionesError(Exception):
    """Base class for all project exceptions."""


class IngestionError(RAGOposicionesError):
    """Error during document ingestion (loading, chunking, or indexing)."""


class RetrievalError(RAGOposicionesError):
    """Error during context retrieval from the vector store."""


class GenerationError(RAGOposicionesError):
    """Error during LLM response or exam generation."""


class VectorStoreError(RAGOposicionesError):
    """Error communicating with the Qdrant vector store."""


class StorageError(RAGOposicionesError):
    """Error communicating with the MinIO document storage."""


class DatabaseError(RAGOposicionesError):
    """Error communicating with the PostgreSQL database."""


class ConfigurationError(RAGOposicionesError):
    """Invalid or missing application configuration."""
