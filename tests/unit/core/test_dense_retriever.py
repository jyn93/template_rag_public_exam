"""Unit tests for DenseRetriever."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.core.exceptions import RetrievalError
from src.core.retrieval.base import DEFAULT_TOP_K, RetrievalResult
from src.core.retrieval.dense_retriever import DenseRetriever

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_results(n: int = 3) -> list[RetrievalResult]:
    """Return *n* dummy RetrievalResult objects."""
    return [
        RetrievalResult(
            content=f"Chunk {i} content",
            score=1.0 - i * 0.1,
            metadata={"subject": "Law", "chunk_index": i},
            doc_id=f"doc-{i}_chunk_{i}",
        )
        for i in range(n)
    ]


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_vector_search() -> AsyncMock:
    """VectorSearchProtocol mock returning three results by default."""
    store = AsyncMock()
    store.search = AsyncMock(return_value=make_results(3))
    return store


@pytest.fixture
def retriever(mock_vector_search) -> DenseRetriever:
    """DenseRetriever wired with the mock vector store."""
    return DenseRetriever(vector_store=mock_vector_search, top_k=5)


# ── TestRetrieve ──────────────────────────────────────────────────────────────


class TestRetrieve:
    """Tests for DenseRetriever.retrieve."""

    async def test_returns_list_of_retrieval_results(
        self, retriever, mock_vector_search
    ):
        """retrieve() returns the list supplied by the vector store."""
        results = await retriever.retrieve("What is the appeal procedure?")

        assert isinstance(results, list)
        assert all(isinstance(r, RetrievalResult) for r in results)

    async def test_calls_vector_store_search_once(self, retriever, mock_vector_search):
        """Vector store search is called exactly once per retrieve() call."""
        await retriever.retrieve("Administrative law basics")

        mock_vector_search.search.assert_called_once()

    async def test_passes_query_to_vector_store(self, retriever, mock_vector_search):
        """The exact query string is forwarded to vector_store.search."""
        query = "What is the appeal procedure?"
        await retriever.retrieve(query)

        call_args = mock_vector_search.search.call_args
        assert call_args[0][0] == query or call_args[1].get("query") == query

    async def test_uses_instance_top_k_when_default_passed(self, mock_vector_search):
        """When top_k equals DEFAULT_TOP_K, the instance's top_k is used."""
        retriever = DenseRetriever(vector_store=mock_vector_search, top_k=7)
        await retriever.retrieve("test query", top_k=DEFAULT_TOP_K)

        positional = mock_vector_search.search.call_args[0]
        kwargs = mock_vector_search.search.call_args[1]
        effective_k = kwargs.get("top_k") or (
            positional[1] if len(positional) > 1 else None
        )
        assert effective_k == 7

    async def test_explicit_top_k_overrides_instance_default(self, mock_vector_search):
        """An explicit top_k != DEFAULT_TOP_K is forwarded as-is."""
        retriever = DenseRetriever(vector_store=mock_vector_search, top_k=5)
        await retriever.retrieve("test query", top_k=10)

        positional = mock_vector_search.search.call_args[0]
        kwargs = mock_vector_search.search.call_args[1]
        effective_k = kwargs.get("top_k") or (
            positional[1] if len(positional) > 1 else None
        )
        assert effective_k == 10

    async def test_returns_empty_list_when_no_results(
        self, retriever, mock_vector_search
    ):
        """An empty list from the store is returned unchanged."""
        mock_vector_search.search.return_value = []

        results = await retriever.retrieve("obscure query")

        assert results == []

    async def test_raises_value_error_for_empty_query(self, retriever):
        """ValueError is raised for an empty string query."""
        with pytest.raises(ValueError, match="empty"):
            await retriever.retrieve("")

    async def test_raises_value_error_for_blank_query(self, retriever):
        """ValueError is raised for a whitespace-only query."""
        with pytest.raises(ValueError, match="empty"):
            await retriever.retrieve("   ")

    async def test_raises_retrieval_error_on_store_exception(
        self, retriever, mock_vector_search
    ):
        """Generic exceptions from the store are wrapped as RetrievalError."""
        mock_vector_search.search.side_effect = ConnectionError("qdrant down")

        with pytest.raises(RetrievalError, match="Dense retrieval failed"):
            await retriever.retrieve("test query")

    async def test_propagates_retrieval_error_unchanged(
        self, retriever, mock_vector_search
    ):
        """RetrievalError from the store is re-raised without wrapping."""
        original = RetrievalError("store error")
        mock_vector_search.search.side_effect = original

        with pytest.raises(RetrievalError) as exc_info:
            await retriever.retrieve("test query")

        assert exc_info.value is original

    async def test_retrieval_error_chains_original_cause(
        self, retriever, mock_vector_search
    ):
        """The original exception is chained as __cause__ on RetrievalError."""
        cause = RuntimeError("unexpected crash")
        mock_vector_search.search.side_effect = cause

        with pytest.raises(RetrievalError) as exc_info:
            await retriever.retrieve("test query")

        assert exc_info.value.__cause__ is cause

    async def test_result_order_is_preserved(self, retriever, mock_vector_search):
        """Results are returned in the same order as the vector store response."""
        ordered = make_results(5)
        mock_vector_search.search.return_value = ordered

        results = await retriever.retrieve("test query")

        assert results == ordered


# ── TestDenseRetrieverInit ────────────────────────────────────────────────────


class TestDenseRetrieverInit:
    """Tests for DenseRetriever construction."""

    def test_default_top_k_is_default_constant(self, mock_vector_search):
        """When top_k is not provided, instance uses DEFAULT_TOP_K."""
        retriever = DenseRetriever(vector_store=mock_vector_search)
        assert retriever._top_k == DEFAULT_TOP_K

    def test_custom_top_k_is_stored(self, mock_vector_search):
        """A custom top_k value is stored correctly."""
        retriever = DenseRetriever(vector_store=mock_vector_search, top_k=12)
        assert retriever._top_k == 12
