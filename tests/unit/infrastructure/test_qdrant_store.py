"""Unit tests for QdrantVectorStore."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from qdrant_client.models import ScoredPoint

from src.core.exceptions import VectorStoreError
from src.core.ingestion.base import Document
from src.core.retrieval.base import RetrievalResult
from src.infrastructure.vector_store.qdrant_store import QdrantVectorStore

# ── Helpers ───────────────────────────────────────────────────────────────────

_COLLECTION = "test_collection"
_EMBED_DIM = 4  # small dimensionality for tests
_FAKE_VECTOR = [0.1, 0.2, 0.3, 0.4]


def make_document(doc_id: str = "doc-1", content: str = "Some text") -> Document:
    return Document(
        content=content,
        metadata={"subject": "Law", "chunk_index": 0},
        doc_id=doc_id,
    )


def make_scored_point(
    point_id: str = "abc-123",
    score: float = 0.9,
    content: str = "Result chunk",
    doc_id: str = "doc-1_chunk_0",
) -> ScoredPoint:
    return ScoredPoint(
        id=point_id,
        version=0,
        score=score,
        payload={
            "content": content,
            "doc_id": doc_id,
            "subject": "Law",
            "chunk_index": 0,
        },
        vector=_FAKE_VECTOR,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_qdrant_client() -> AsyncMock:
    """Async Qdrant client mock."""
    client = AsyncMock()
    client.collection_exists = AsyncMock(return_value=True)
    client.create_collection = AsyncMock()
    client.upsert = AsyncMock()
    client.search = AsyncMock(return_value=[make_scored_point()])
    return client


@pytest.fixture
def mock_embedding_model() -> AsyncMock:
    """LlamaIndex BaseEmbedding mock."""
    model = AsyncMock()
    model.aget_text_embedding_batch = AsyncMock(return_value=[_FAKE_VECTOR])
    model.aget_query_embedding = AsyncMock(return_value=_FAKE_VECTOR)
    return model


@pytest.fixture
def store(mock_qdrant_client, mock_embedding_model) -> QdrantVectorStore:
    """QdrantVectorStore with all dependencies mocked."""
    return QdrantVectorStore(
        client=mock_qdrant_client,
        embedding_model=mock_embedding_model,
        collection_name=_COLLECTION,
        embedding_dim=_EMBED_DIM,
    )


# ── TestEnsureCollection ──────────────────────────────────────────────────────


class TestEnsureCollection:
    """Tests for QdrantVectorStore._ensure_collection."""

    async def test_does_not_create_collection_if_already_exists(
        self, store, mock_qdrant_client
    ):
        """create_collection is NOT called when the collection already exists."""
        mock_qdrant_client.collection_exists.return_value = True

        await store._ensure_collection()

        mock_qdrant_client.create_collection.assert_not_called()

    async def test_creates_collection_when_missing(
        self, store, mock_qdrant_client
    ):
        """create_collection IS called when the collection does not exist."""
        mock_qdrant_client.collection_exists.return_value = False

        await store._ensure_collection()

        mock_qdrant_client.create_collection.assert_called_once()

    async def test_create_collection_uses_correct_name(
        self, store, mock_qdrant_client
    ):
        """create_collection is called with the configured collection name."""
        mock_qdrant_client.collection_exists.return_value = False

        await store._ensure_collection()

        call_kwargs = mock_qdrant_client.create_collection.call_args[1]
        assert call_kwargs.get("collection_name") == _COLLECTION or (
            mock_qdrant_client.create_collection.call_args[0]
            and mock_qdrant_client.create_collection.call_args[0][0] == _COLLECTION
        )

    async def test_raises_vector_store_error_on_client_failure(
        self, store, mock_qdrant_client
    ):
        """VectorStoreError is raised when Qdrant raises during collection check."""
        mock_qdrant_client.collection_exists.side_effect = ConnectionError("timeout")

        with pytest.raises(VectorStoreError, match="Failed to ensure collection"):
            await store._ensure_collection()


# ── TestAddDocuments ──────────────────────────────────────────────────────────


class TestAddDocuments:
    """Tests for QdrantVectorStore.add_documents."""

    async def test_returns_inserted_count(self, store, mock_embedding_model):
        """Result dict contains correct 'inserted' count."""
        docs = [make_document("d1"), make_document("d2")]
        mock_embedding_model.aget_text_embedding_batch.return_value = [
            _FAKE_VECTOR,
            _FAKE_VECTOR,
        ]

        result = await store.add_documents(docs)

        assert result["inserted"] == 2

    async def test_returns_collection_name(self, store):
        """Result dict contains the configured collection name."""
        result = await store.add_documents([make_document()])

        assert result["collection"] == _COLLECTION

    async def test_empty_documents_returns_zero_inserted(
        self, store, mock_qdrant_client
    ):
        """Empty input skips all Qdrant calls and returns inserted=0."""
        result = await store.add_documents([])

        mock_qdrant_client.upsert.assert_not_called()
        assert result["inserted"] == 0

    async def test_calls_embedding_batch_with_all_contents(
        self, store, mock_embedding_model
    ):
        """Embedding batch is called with the text of every document."""
        docs = [make_document("d1", "text one"), make_document("d2", "text two")]
        mock_embedding_model.aget_text_embedding_batch.return_value = [
            _FAKE_VECTOR,
            _FAKE_VECTOR,
        ]

        await store.add_documents(docs)

        call_args = mock_embedding_model.aget_text_embedding_batch.call_args[0][0]
        assert "text one" in call_args
        assert "text two" in call_args

    async def test_calls_upsert_once(self, store, mock_qdrant_client):
        """Qdrant upsert is called exactly once per add_documents call."""
        await store.add_documents([make_document()])

        mock_qdrant_client.upsert.assert_called_once()

    async def test_raises_vector_store_error_on_embedding_failure(
        self, store, mock_embedding_model
    ):
        """VectorStoreError is raised when the embedding model fails."""
        mock_embedding_model.aget_text_embedding_batch.side_effect = RuntimeError(
            "API error"
        )

        with pytest.raises(VectorStoreError, match="Embedding batch failed"):
            await store.add_documents([make_document()])

    async def test_raises_vector_store_error_on_upsert_failure(
        self, store, mock_qdrant_client
    ):
        """VectorStoreError is raised when Qdrant upsert fails."""
        mock_qdrant_client.upsert.side_effect = ConnectionError("qdrant down")

        with pytest.raises(VectorStoreError, match="Qdrant upsert failed"):
            await store.add_documents([make_document()])

    async def test_document_content_stored_in_payload(
        self, store, mock_qdrant_client
    ):
        """Each upserted point's payload contains the document content."""
        doc = make_document("d1", "Important legal text")

        await store.add_documents([doc])

        upsert_call = mock_qdrant_client.upsert.call_args
        all_args = list(upsert_call[0]) + list(upsert_call[1].values())
        points_list = next((a for a in all_args if isinstance(a, list)), None)
        assert points_list is not None
        assert any(
            p.payload.get("content") == "Important legal text"
            for p in points_list
        )


# ── TestSearch ────────────────────────────────────────────────────────────────


class TestSearch:
    """Tests for QdrantVectorStore.search."""

    async def test_returns_list_of_retrieval_results(self, store):
        """search() returns a list of RetrievalResult instances."""
        results = await store.search("appeal procedure", top_k=3)

        assert isinstance(results, list)
        assert all(isinstance(r, RetrievalResult) for r in results)

    async def test_result_content_matches_payload(
        self, store, mock_qdrant_client
    ):
        """RetrievalResult.content is extracted from the Qdrant payload."""
        mock_qdrant_client.search.return_value = [
            make_scored_point(content="Specific content here")
        ]

        results = await store.search("query", top_k=1)

        assert results[0].content == "Specific content here"

    async def test_result_score_matches_scored_point(
        self, store, mock_qdrant_client
    ):
        """RetrievalResult.score equals the ScoredPoint's score."""
        mock_qdrant_client.search.return_value = [
            make_scored_point(score=0.87)
        ]

        results = await store.search("query", top_k=1)

        assert results[0].score == pytest.approx(0.87)

    async def test_result_doc_id_from_payload(
        self, store, mock_qdrant_client
    ):
        """RetrievalResult.doc_id is taken from the 'doc_id' payload key."""
        mock_qdrant_client.search.return_value = [
            make_scored_point(doc_id="parent-1_chunk_0")
        ]

        results = await store.search("query", top_k=1)

        assert results[0].doc_id == "parent-1_chunk_0"

    async def test_metadata_excludes_content_and_doc_id_keys(
        self, store, mock_qdrant_client
    ):
        """'content' and 'doc_id' are stripped from RetrievalResult.metadata."""
        mock_qdrant_client.search.return_value = [make_scored_point()]

        results = await store.search("query", top_k=1)

        assert "content" not in results[0].metadata
        assert "doc_id" not in results[0].metadata

    async def test_metadata_preserves_other_payload_fields(
        self, store, mock_qdrant_client
    ):
        """Payload fields other than content/doc_id appear in metadata."""
        mock_qdrant_client.search.return_value = [make_scored_point()]

        results = await store.search("query", top_k=1)

        assert results[0].metadata.get("subject") == "Law"

    async def test_empty_results_returns_empty_list(
        self, store, mock_qdrant_client
    ):
        """Empty Qdrant response is returned as an empty list."""
        mock_qdrant_client.search.return_value = []

        results = await store.search("obscure query", top_k=5)

        assert results == []

    async def test_calls_embedding_model_for_query(
        self, store, mock_embedding_model
    ):
        """aget_query_embedding is called once with the exact query string."""
        await store.search("my search query", top_k=5)

        mock_embedding_model.aget_query_embedding.assert_called_once_with(
            "my search query"
        )

    async def test_calls_qdrant_search_with_top_k(
        self, store, mock_qdrant_client
    ):
        """Qdrant client search is called with the correct limit."""
        await store.search("query", top_k=7)

        call_kwargs = mock_qdrant_client.search.call_args[1]
        assert call_kwargs.get("limit") == 7 or (
            len(mock_qdrant_client.search.call_args[0]) > 2
            and mock_qdrant_client.search.call_args[0][2] == 7
        )

    async def test_raises_vector_store_error_on_embedding_failure(
        self, store, mock_embedding_model
    ):
        """VectorStoreError is raised when query embedding fails."""
        mock_embedding_model.aget_query_embedding.side_effect = RuntimeError(
            "rate limited"
        )

        with pytest.raises(VectorStoreError, match="Failed to embed query"):
            await store.search("query", top_k=5)

    async def test_raises_vector_store_error_on_search_failure(
        self, store, mock_qdrant_client
    ):
        """VectorStoreError is raised when Qdrant search fails."""
        mock_qdrant_client.search.side_effect = ConnectionError("timeout")

        with pytest.raises(VectorStoreError, match="Qdrant search failed"):
            await store.search("query", top_k=5)

    async def test_error_chains_original_cause(
        self, store, mock_qdrant_client
    ):
        """The original exception is chained as __cause__ on VectorStoreError."""
        cause = ConnectionError("network error")
        mock_qdrant_client.search.side_effect = cause

        with pytest.raises(VectorStoreError) as exc_info:
            await store.search("query", top_k=5)

        assert exc_info.value.__cause__ is cause
