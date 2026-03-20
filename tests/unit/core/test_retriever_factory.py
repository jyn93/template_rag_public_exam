"""Unit tests for RetrieverFactory."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.core.config.settings import RetrieverType, Settings
from src.core.retrieval.dense_retriever import DenseRetriever
from src.core.retrieval.factory import RetrieverFactory
from src.core.retrieval.hybrid_retriever import HybridRetriever

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    store = AsyncMock()
    store.search = AsyncMock(return_value=[])
    return store


def make_settings(retriever_type: RetrieverType, top_k: int = 5) -> Settings:
    return Settings(
        retriever_type=retriever_type,
        top_k=top_k,
        openai_api_key="test-key",
        _env_file=None,  # type: ignore[call-arg]
    )


# ── TestRetrieverFactory ──────────────────────────────────────────────────────


class TestRetrieverFactory:
    """Tests for RetrieverFactory.create."""

    def test_creates_dense_retriever_for_dense_type(self):
        """DENSE settings → DenseRetriever instance."""
        store = AsyncMock()
        settings = make_settings(RetrieverType.DENSE)

        retriever = RetrieverFactory.create(store, settings=settings)

        assert isinstance(retriever, DenseRetriever)

    def test_creates_hybrid_retriever_for_hybrid_type(self):
        """HYBRID settings → HybridRetriever instance."""
        store = AsyncMock()
        settings = make_settings(RetrieverType.HYBRID)

        retriever = RetrieverFactory.create(store, settings=settings)

        assert isinstance(retriever, HybridRetriever)

    def test_top_k_propagated_to_dense_retriever(self):
        """top_k from settings is passed to DenseRetriever."""
        store = AsyncMock()
        settings = make_settings(RetrieverType.DENSE, top_k=7)

        retriever = RetrieverFactory.create(store, settings=settings)

        assert retriever._top_k == 7

    def test_top_k_propagated_to_hybrid_retriever(self):
        """top_k from settings is passed to HybridRetriever."""
        store = AsyncMock()
        settings = make_settings(RetrieverType.HYBRID, top_k=10)

        retriever = RetrieverFactory.create(store, settings=settings)

        assert retriever._top_k == 10

    def test_vector_store_injected_into_dense_retriever(self):
        """The provided vector_store is stored in DenseRetriever."""
        store = AsyncMock()
        settings = make_settings(RetrieverType.DENSE)

        retriever = RetrieverFactory.create(store, settings=settings)

        assert retriever._vector_store is store

    def test_vector_store_injected_into_hybrid_retriever(self):
        """The provided vector_store is stored in HybridRetriever."""
        store = AsyncMock()
        settings = make_settings(RetrieverType.HYBRID)

        retriever = RetrieverFactory.create(store, settings=settings)

        assert retriever._vector_store is store

    def test_uses_get_settings_when_none_provided(self, monkeypatch):
        """When settings=None, get_settings() is called."""
        store = AsyncMock()
        called = []

        def fake_get_settings() -> Settings:
            s = make_settings(RetrieverType.DENSE)
            called.append(True)
            return s

        monkeypatch.setattr(
            "src.core.retrieval.factory.get_settings", fake_get_settings
        )

        RetrieverFactory.create(store)

        assert called
