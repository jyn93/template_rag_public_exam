"""Unit tests for the chat router (POST /chat)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_rag_generator, get_retriever
from src.api.main import app
from src.core.exceptions import GenerationError, RetrievalError
from src.core.generation.base import GenerationOutput
from src.core.retrieval.base import RetrievalResult

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_retrieval_result(content: str = "Some law text.") -> RetrievalResult:
    return RetrievalResult(content=content, score=0.9, doc_id="doc1")


def _make_retriever_mock(
    results: list[RetrievalResult] | None = None,
    exc: Exception | None = None,
) -> MagicMock:
    mock = MagicMock()
    if exc is not None:
        mock.retrieve = AsyncMock(side_effect=exc)
    else:
        default = results if results is not None else [_make_retrieval_result()]
        mock.retrieve = AsyncMock(return_value=default)
    return mock


def _make_generator_mock(
    answer: str = "The appeal is a formal remedy.",
    exc: Exception | None = None,
) -> MagicMock:
    mock = MagicMock()
    if exc is not None:
        mock.generate = AsyncMock(side_effect=exc)
    else:
        output = GenerationOutput(
            content=answer, sources=["Some law text..."], tokens_used=42
        )
        mock.generate = AsyncMock(return_value=output)
    return mock


def _override(
    retriever: MagicMock | None = None, generator: MagicMock | None = None
) -> None:
    if retriever is not None:
        app.dependency_overrides[get_retriever] = lambda: retriever
    if generator is not None:
        app.dependency_overrides[get_rag_generator] = lambda: generator


def _clear() -> None:
    app.dependency_overrides.pop(get_retriever, None)
    app.dependency_overrides.pop(get_rag_generator, None)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def cleanup():
    yield
    _clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def happy_path(client):
    """Client with retriever and generator mocks returning valid results."""
    retriever = _make_retriever_mock()
    generator = _make_generator_mock()
    _override(retriever=retriever, generator=generator)
    return client, retriever, generator


# ── TestChatSuccess ───────────────────────────────────────────────────────────


class TestChatSuccess:
    """Tests for successful POST /chat responses."""

    def test_returns_200(self, happy_path):
        """Valid request returns HTTP 200."""
        client, _, _ = happy_path
        response = client.post("/chat", json={"query": "What is an appeal?"})
        assert response.status_code == 200

    def test_response_contains_answer(self, happy_path):
        """Response body includes the generated answer."""
        client, _, _ = happy_path
        response = client.post("/chat", json={"query": "What is an appeal?"})
        assert response.json()["answer"] == "The appeal is a formal remedy."

    def test_response_contains_sources(self, happy_path):
        """Response body includes source excerpts."""
        client, _, _ = happy_path
        response = client.post("/chat", json={"query": "What is an appeal?"})
        assert isinstance(response.json()["sources"], list)

    def test_response_contains_tokens_used(self, happy_path):
        """Response body includes tokens_used count."""
        client, _, _ = happy_path
        response = client.post("/chat", json={"query": "What is an appeal?"})
        assert response.json()["tokens_used"] == 42

    def test_retriever_called_with_query(self, happy_path):
        """Retriever is called with the user query."""
        client, retriever, _ = happy_path
        client.post("/chat", json={"query": "habeas corpus"})
        retriever.retrieve.assert_called_once()
        args, kwargs = retriever.retrieve.call_args
        assert args[0] == "habeas corpus"

    def test_custom_top_k_forwarded(self, happy_path):
        """Custom top_k is forwarded to the retriever."""
        client, retriever, _ = happy_path
        client.post("/chat", json={"query": "query", "top_k": 10})
        _, kwargs = retriever.retrieve.call_args
        assert kwargs["top_k"] == 10


# ── TestChatValidation ────────────────────────────────────────────────────────


class TestChatValidation:
    """Tests for request validation in POST /chat."""

    def test_empty_query_returns_422(self, client):
        """Empty query string returns HTTP 422."""
        _override(retriever=_make_retriever_mock(), generator=_make_generator_mock())
        response = client.post("/chat", json={"query": ""})
        assert response.status_code == 422

    def test_no_results_returns_422(self, client):
        """No retrieval results returns HTTP 422."""
        _override(
            retriever=_make_retriever_mock(results=[]),
            generator=_make_generator_mock(),
        )
        response = client.post("/chat", json={"query": "unknown topic"})
        assert response.status_code == 422

    def test_top_k_too_large_returns_422(self, client):
        """top_k > 20 returns HTTP 422."""
        _override(retriever=_make_retriever_mock(), generator=_make_generator_mock())
        response = client.post("/chat", json={"query": "query", "top_k": 99})
        assert response.status_code == 422


# ── TestChatErrorHandling ─────────────────────────────────────────────────────


class TestChatErrorHandling:
    """Tests for domain exception mapping in POST /chat."""

    def test_retrieval_error_returns_500(self, client):
        """RetrievalError from retriever returns HTTP 500."""
        _override(
            retriever=_make_retriever_mock(exc=RetrievalError("qdrant down")),
            generator=_make_generator_mock(),
        )
        response = client.post("/chat", json={"query": "query"})
        assert response.status_code == 500

    def test_generation_error_returns_500(self, client):
        """GenerationError from generator returns HTTP 500."""
        _override(
            retriever=_make_retriever_mock(),
            generator=_make_generator_mock(exc=GenerationError("llm down")),
        )
        response = client.post("/chat", json={"query": "query"})
        assert response.status_code == 500

    def test_retrieval_error_detail_forwarded(self, client):
        """HTTP 500 detail includes the RetrievalError message."""
        _override(
            retriever=_make_retriever_mock(exc=RetrievalError("connection refused")),
            generator=_make_generator_mock(),
        )
        response = client.post("/chat", json={"query": "query"})
        assert "connection refused" in response.json()["detail"]
