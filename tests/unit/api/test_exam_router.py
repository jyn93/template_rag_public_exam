"""Unit tests for the exam router (POST /exam/generate, POST /exam/evaluate)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_answer_evaluator, get_llm_client, get_retriever
from src.api.main import app
from src.core.exceptions import GenerationError, RetrievalError
from src.core.generation.evaluator import EvaluationResult
from src.core.retrieval.base import RetrievalResult

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_result(content: str = "Law text.") -> RetrievalResult:
    return RetrievalResult(content=content, score=0.9, doc_id="doc1")


def _make_retriever_mock(
    results: list[RetrievalResult] | None = None,
    exc: Exception | None = None,
) -> MagicMock:
    mock = MagicMock()
    if exc is not None:
        mock.retrieve = AsyncMock(side_effect=exc)
    else:
        default = results if results is not None else [_make_result()]
        mock.retrieve = AsyncMock(return_value=default)
    return mock


def _make_llm_mock(response: str = '{"questions": []}') -> MagicMock:
    mock = MagicMock()
    mock.complete = AsyncMock(return_value=response)
    return mock


def _make_evaluator_mock(
    exc: Exception | None = None,
) -> MagicMock:
    mock = MagicMock()
    if exc is not None:
        mock.evaluate = AsyncMock(side_effect=exc)
    else:
        result = EvaluationResult(
            {
                "score": 8,
                "is_correct": True,
                "feedback": "Well done.",
                "missing_points": [],
                "strengths": ["Cited article 24."],
            }
        )
        mock.evaluate = AsyncMock(return_value=result)
    return mock


def _override(**deps) -> None:
    mapping = {
        "retriever": get_retriever,
        "llm_client": get_llm_client,
        "evaluator": get_answer_evaluator,
    }
    for name, dep in deps.items():
        if dep is not None:
            app.dependency_overrides[mapping[name]] = lambda d=dep: d


def _clear() -> None:
    for dep in (get_retriever, get_llm_client, get_answer_evaluator):
        app.dependency_overrides.pop(dep, None)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def cleanup():
    yield
    _clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def generate_client(client):
    """Client with mocked retriever and LLM for exam generation."""
    retriever = _make_retriever_mock()
    llm = _make_llm_mock()
    _override(retriever=retriever, llm_client=llm)
    return client, retriever, llm


@pytest.fixture
def evaluate_client(client):
    """Client with mocked evaluator for answer evaluation."""
    evaluator = _make_evaluator_mock()
    _override(evaluator=evaluator)
    return client, evaluator


# ── TestExamGenerate ──────────────────────────────────────────────────────────


class TestExamGenerate:
    """Tests for POST /exam/generate."""

    def test_returns_200(self, generate_client):
        """Valid request returns HTTP 200."""
        client, _, _ = generate_client
        response = client.post(
            "/exam/generate",
            json={"query": "Administrative appeal", "num_questions": 3},
        )
        assert response.status_code == 200

    def test_response_contains_exam_key(self, generate_client):
        """Response body contains the 'exam' dict."""
        client, _, _ = generate_client
        response = client.post("/exam/generate", json={"query": "appeal"})
        assert "exam" in response.json()

    def test_response_contains_sources(self, generate_client):
        """Response body includes source excerpts."""
        client, _, _ = generate_client
        response = client.post("/exam/generate", json={"query": "appeal"})
        assert "sources" in response.json()

    def test_retriever_result_used_by_generator(self, client):
        """Retriever results are passed to the generator (end-to-end wiring)."""
        _override(retriever=_make_retriever_mock(), llm_client=_make_llm_mock())
        response = client.post("/exam/generate", json={"query": "my topic"})
        # LLM mock returns '{"questions": []}' → exam key must be present and 200
        assert response.status_code == 200
        assert "exam" in response.json()

    def test_no_results_returns_422(self, client):
        """Empty retrieval returns HTTP 422."""
        _override(
            retriever=_make_retriever_mock(results=[]), llm_client=_make_llm_mock()
        )
        response = client.post("/exam/generate", json={"query": "unknown"})
        assert response.status_code == 422

    def test_retrieval_error_returns_500(self, client):
        """RetrievalError returns HTTP 500."""
        _override(
            retriever=_make_retriever_mock(exc=RetrievalError("qdrant down")),
            llm_client=_make_llm_mock(),
        )
        response = client.post("/exam/generate", json={"query": "topic"})
        assert response.status_code == 500

    def test_generation_error_returns_500(self, client):
        """GenerationError from LLM returns HTTP 500."""
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=GenerationError("llm down"))
        _override(retriever=_make_retriever_mock(), llm_client=llm)
        response = client.post("/exam/generate", json={"query": "topic"})
        assert response.status_code == 500

    def test_invalid_exam_type_returns_422(self, client):
        """Invalid exam_type value returns HTTP 422."""
        _override(retriever=_make_retriever_mock(), llm_client=_make_llm_mock())
        response = client.post(
            "/exam/generate",
            json={"query": "topic", "exam_type": "invalid"},
        )
        assert response.status_code == 422

    def test_invalid_difficulty_returns_422(self, client):
        """Invalid difficulty value returns HTTP 422."""
        _override(retriever=_make_retriever_mock(), llm_client=_make_llm_mock())
        response = client.post(
            "/exam/generate",
            json={"query": "topic", "difficulty": "extreme"},
        )
        assert response.status_code == 422


# ── TestExamEvaluate ──────────────────────────────────────────────────────────


class TestExamEvaluate:
    """Tests for POST /exam/evaluate."""

    def test_returns_200(self, evaluate_client):
        """Valid request returns HTTP 200."""
        client, _ = evaluate_client
        response = client.post(
            "/exam/evaluate",
            json={
                "question": "What is habeas corpus?",
                "correct_answer": "A writ...",
                "student_answer": "It protects liberty.",
            },
        )
        assert response.status_code == 200

    def test_response_contains_score(self, evaluate_client):
        """Response body contains score."""
        client, _ = evaluate_client
        response = client.post(
            "/exam/evaluate",
            json={
                "question": "Q?",
                "correct_answer": "Correct.",
                "student_answer": "Student answer.",
            },
        )
        assert response.json()["score"] == 8.0

    def test_response_contains_is_correct(self, evaluate_client):
        """Response body contains is_correct flag."""
        client, _ = evaluate_client
        response = client.post(
            "/exam/evaluate",
            json={
                "question": "Q?",
                "correct_answer": "Correct.",
                "student_answer": "Student answer.",
            },
        )
        assert response.json()["is_correct"] is True

    def test_response_contains_feedback(self, evaluate_client):
        """Response body contains feedback text."""
        client, _ = evaluate_client
        response = client.post(
            "/exam/evaluate",
            json={
                "question": "Q?",
                "correct_answer": "Correct.",
                "student_answer": "Student answer.",
            },
        )
        assert response.json()["feedback"] == "Well done."

    def test_evaluator_result_reflected_in_response(self, client):
        """Evaluator mock result (score=8) is faithfully serialised in the response."""
        _override(evaluator=_make_evaluator_mock())
        response = client.post(
            "/exam/evaluate",
            json={
                "question": "Q?",
                "correct_answer": "Correct.",
                "student_answer": "Student.",
            },
        )
        # Mock returns score=8 and feedback="Well done." — verifies evaluator was used
        assert response.status_code == 200
        assert response.json()["score"] == 8.0
        assert response.json()["feedback"] == "Well done."

    def test_generation_error_returns_500(self, client):
        """GenerationError from evaluator returns HTTP 500."""
        _override(evaluator=_make_evaluator_mock(exc=GenerationError("llm down")))
        response = client.post(
            "/exam/evaluate",
            json={
                "question": "Q?",
                "correct_answer": "Correct.",
                "student_answer": "Bad.",
            },
        )
        assert response.status_code == 500

    def test_empty_question_returns_422(self, client):
        """Empty question string returns HTTP 422."""
        _override(evaluator=_make_evaluator_mock())
        response = client.post(
            "/exam/evaluate",
            json={
                "question": "",
                "correct_answer": "Correct.",
                "student_answer": "Student.",
            },
        )
        assert response.status_code == 422
