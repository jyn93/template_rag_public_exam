"""Unit tests for HybridRetriever."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.core.exceptions import RetrievalError
from src.core.retrieval.base import DEFAULT_TOP_K, RetrievalResult
from src.core.retrieval.hybrid_retriever import (
    _DEFAULT_EXPANSION_FACTOR,
    HybridRetriever,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_result(
    doc_id: str,
    content: str = "Sample chunk text about legal procedure",
    score: float = 0.9,
) -> RetrievalResult:
    return RetrievalResult(
        content=content,
        score=score,
        metadata={"subject": "Law"},
        doc_id=doc_id,
    )


def make_candidates(n: int = 8) -> list[RetrievalResult]:
    """Return *n* candidates with distinct doc_ids and contents."""
    topics = [
        "administrative appeal procedure",
        "civil law contract obligations",
        "criminal procedure rights",
        "constitutional rights citizen",
        "labor law employment contract",
        "tax law income declaration",
        "property law real estate",
        "social security benefits",
        "environmental regulation waste",
        "consumer protection warranty",
    ]
    return [
        make_result(
            doc_id=f"doc-{i}_chunk_0",
            content=topics[i % len(topics)],
            score=1.0 - i * 0.05,
        )
        for i in range(n)
    ]


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_vector_search() -> AsyncMock:
    """VectorSearchProtocol mock returning 8 candidates by default."""
    store = AsyncMock()
    store.search = AsyncMock(return_value=make_candidates(8))
    return store


@pytest.fixture
def retriever(mock_vector_search) -> HybridRetriever:
    """HybridRetriever with top_k=3 and expansion_factor=4."""
    return HybridRetriever(
        vector_store=mock_vector_search,
        top_k=3,
        expansion_factor=4,
    )


# ── TestHybridRetrieverInit ───────────────────────────────────────────────────


class TestHybridRetrieverInit:
    """Tests for HybridRetriever construction."""

    def test_default_top_k(self, mock_vector_search):
        """Default top_k matches DEFAULT_TOP_K constant."""
        r = HybridRetriever(vector_store=mock_vector_search)
        assert r._top_k == DEFAULT_TOP_K

    def test_custom_top_k_stored(self, mock_vector_search):
        """Custom top_k is stored correctly."""
        r = HybridRetriever(vector_store=mock_vector_search, top_k=10)
        assert r._top_k == 10

    def test_default_expansion_factor(self, mock_vector_search):
        """Default expansion_factor matches constant."""
        r = HybridRetriever(vector_store=mock_vector_search)
        assert r._expansion_factor == _DEFAULT_EXPANSION_FACTOR

    def test_custom_weights_stored(self, mock_vector_search):
        """Custom bm25 and dense weights are stored."""
        r = HybridRetriever(
            vector_store=mock_vector_search,
            bm25_weight=0.3,
            dense_weight=0.7,
        )
        assert r._bm25_weight == pytest.approx(0.3)
        assert r._dense_weight == pytest.approx(0.7)


# ── TestRetrieve ──────────────────────────────────────────────────────────────


class TestRetrieve:
    """Tests for HybridRetriever.retrieve."""

    async def test_returns_list_of_retrieval_results(
        self, retriever, mock_vector_search
    ):
        """retrieve() returns a list of RetrievalResult."""
        results = await retriever.retrieve("appeal procedure")

        assert isinstance(results, list)
        assert all(isinstance(r, RetrievalResult) for r in results)

    async def test_returns_top_k_results(self, retriever, mock_vector_search):
        """Number of returned results does not exceed top_k."""
        results = await retriever.retrieve("appeal", top_k=3)

        assert len(results) <= 3

    async def test_calls_dense_search_with_expanded_k(
        self, retriever, mock_vector_search
    ):
        """Dense search is called with top_k * expansion_factor candidates."""
        await retriever.retrieve("appeal", top_k=3)

        call_args = mock_vector_search.search.call_args[0]
        called_k = call_args[1]
        assert called_k == 3 * retriever._expansion_factor

    async def test_uses_instance_top_k_when_default_passed(self, mock_vector_search):
        """Instance top_k is used when top_k == DEFAULT_TOP_K."""
        r = HybridRetriever(
            vector_store=mock_vector_search,
            top_k=2,
            expansion_factor=3,
        )
        await r.retrieve("query", top_k=DEFAULT_TOP_K)

        called_k = mock_vector_search.search.call_args[0][1]
        assert called_k == 2 * 3

    async def test_empty_candidates_returns_empty_list(
        self, retriever, mock_vector_search
    ):
        """Empty dense search result → empty final list."""
        mock_vector_search.search.return_value = []

        results = await retriever.retrieve("obscure query")

        assert results == []

    async def test_raises_value_error_for_empty_query(self, retriever):
        """ValueError raised for empty query."""
        with pytest.raises(ValueError, match="empty"):
            await retriever.retrieve("")

    async def test_raises_value_error_for_blank_query(self, retriever):
        """ValueError raised for whitespace-only query."""
        with pytest.raises(ValueError, match="empty"):
            await retriever.retrieve("   ")

    async def test_wraps_store_exception_as_retrieval_error(
        self, retriever, mock_vector_search
    ):
        """Generic exceptions from the store become RetrievalError."""
        mock_vector_search.search.side_effect = ConnectionError("timeout")

        with pytest.raises(RetrievalError, match="Hybrid retrieval failed"):
            await retriever.retrieve("query")

    async def test_propagates_retrieval_error_unchanged(
        self, retriever, mock_vector_search
    ):
        """RetrievalError from the store is re-raised as-is."""
        original = RetrievalError("store error")
        mock_vector_search.search.side_effect = original

        with pytest.raises(RetrievalError) as exc_info:
            await retriever.retrieve("query")

        assert exc_info.value is original

    async def test_error_chains_original_cause(self, retriever, mock_vector_search):
        """Original exception is chained on the wrapping RetrievalError."""
        cause = RuntimeError("crash")
        mock_vector_search.search.side_effect = cause

        with pytest.raises(RetrievalError) as exc_info:
            await retriever.retrieve("query")

        assert exc_info.value.__cause__ is cause

    async def test_bm25_reranking_can_change_order(self, mock_vector_search):
        """BM25-biased weights promote lexically matching content over dense rank.

        BM25 IDF requires >2 documents in the corpus so that log((N-n+0.5)/(n+0.5)) > 0.
        We use 4 candidates where only doc-3 lexically matches the query.
        """
        candidates = [
            make_result(
                "doc-0", content="color spectrum visible light photon", score=0.95
            ),
            make_result(
                "doc-1", content="climate change global warming ocean", score=0.90
            ),
            make_result(
                "doc-2", content="cooking recipes pasta ingredients", score=0.85
            ),
            make_result(
                "doc-3",
                content="administrative appeal procedure rights",
                score=0.50,
            ),
        ]
        mock_vector_search.search.return_value = candidates

        # BM25-dominant weights so lexical match overrides dense ranking
        r = HybridRetriever(
            vector_store=mock_vector_search,
            top_k=4,
            expansion_factor=1,
            bm25_weight=0.95,
            dense_weight=0.05,
        )
        results = await r.retrieve("administrative appeal procedure")

        # doc-3 is the only lexically matching document → promoted to rank 0
        assert results[0].doc_id == "doc-3"

    async def test_single_candidate_returned(self, mock_vector_search):
        """Single candidate is returned without error."""
        mock_vector_search.search.return_value = [
            make_result("doc-0", "sole result text")
        ]
        r = HybridRetriever(
            vector_store=mock_vector_search,
            top_k=5,
            expansion_factor=1,
        )
        results = await r.retrieve("query")

        assert len(results) == 1


# ── TestFuse ──────────────────────────────────────────────────────────────────


class TestFuse:
    """Tests for HybridRetriever._fuse (internal method)."""

    def test_fuse_returns_at_most_top_k(self, retriever):
        """_fuse never returns more than top_k results."""
        candidates = make_candidates(10)
        results = retriever._fuse("appeal", candidates, top_k=3)

        assert len(results) <= 3

    def test_fuse_preserves_all_result_fields(self, retriever):
        """Fused results retain content, score, and metadata from candidates."""
        candidates = [make_result("d1", "test content", 0.8)]
        results = retriever._fuse("test", candidates, top_k=1)

        assert results[0].content == "test content"
        assert results[0].doc_id == "d1"


# ── TestTokenize ──────────────────────────────────────────────────────────────


class TestTokenize:
    """Tests for HybridRetriever._tokenize (static helper)."""

    def test_lowercases_tokens(self):
        """All tokens are lowercased."""
        tokens = HybridRetriever._tokenize("Administrative APPEAL Procedure")
        assert tokens == ["administrative", "appeal", "procedure"]

    def test_splits_on_whitespace(self):
        """Tokens are split on whitespace."""
        tokens = HybridRetriever._tokenize("one two three")
        assert len(tokens) == 3

    def test_empty_string_returns_empty_list(self):
        """Empty string tokenises to empty list."""
        tokens = HybridRetriever._tokenize("")
        assert tokens == []
