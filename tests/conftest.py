"""Shared pytest fixtures for all test suites."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.core.config.settings import Settings


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Settings with test values (no real API keys required).

    Returns:
        Settings instance configured for testing.
    """
    return Settings(
        app_name="RAG Oposiciones Test",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        openai_api_key="test-key-not-real",
        anthropic_api_key="test-key-not-real",
        qdrant_url="http://localhost:6333",
        qdrant_collection="test_collection",
        database_url="sqlite+aiosqlite:///:memory:",
        debug=True,
        log_level="DEBUG",
    )


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """Reusable async LLM client mock.

    Returns:
        AsyncMock configured with a default successful response.
    """
    client = AsyncMock()
    client.complete = AsyncMock(return_value='{"result": "ok"}')
    client.stream = AsyncMock()
    return client


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    """Reusable vector store mock.

    Returns:
        AsyncMock configured with default successful responses.
    """
    store = AsyncMock()
    store.add_documents = AsyncMock(return_value={"inserted": 10})
    store.search = AsyncMock(return_value=[])
    store.get_index = AsyncMock()
    return store


@pytest.fixture
def mock_doc_storage() -> AsyncMock:
    """Reusable document (MinIO) storage mock.

    Returns:
        AsyncMock configured with default successful responses.
    """
    storage = AsyncMock()
    storage.upload = AsyncMock(return_value={"bucket": "temario-docs", "key": "test.pdf"})
    storage.download = AsyncMock(return_value=b"file content")
    return storage
