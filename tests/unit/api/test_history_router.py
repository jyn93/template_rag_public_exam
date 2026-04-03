"""Unit tests for the history router (POST/GET /history/sessions)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.infrastructure.database.engine import get_async_session
from src.infrastructure.database.models import StudySession

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_session_record(**kwargs: object) -> StudySession:
    """Build a minimal StudySession ORM object for mocking."""
    record = MagicMock(spec=StudySession)
    record.id = kwargs.get("id", uuid.uuid4())
    record.session_type = kwargs.get("session_type", "exam")
    record.topic = kwargs.get("topic", None)
    record.subject = kwargs.get("subject", None)
    record.score_avg = kwargs.get("score_avg", None)
    record.num_questions = kwargs.get("num_questions", None)
    record.num_correct = kwargs.get("num_correct", None)
    record.percentage = kwargs.get("percentage", None)
    record.created_at = kwargs.get(
        "created_at", datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
    )
    return record


def _make_repo_mock(
    create_return: StudySession | None = None,
    list_return: list[StudySession] | None = None,
    get_return: StudySession | None = None,
) -> MagicMock:
    """Build a mocked StudySessionRepository."""
    repo = MagicMock()
    repo.create = AsyncMock(return_value=create_return or _make_session_record())
    repo.list_recent = AsyncMock(return_value=list_return or [])
    repo.get_by_id = AsyncMock(return_value=get_return)
    return repo


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_db_session():
    """Override the DB session dependency with a no-op async mock."""
    async def _fake_session():  # type: ignore[return]
        yield MagicMock()

    app.dependency_overrides[get_async_session] = _fake_session
    yield
    app.dependency_overrides.pop(get_async_session, None)


# ── TestRecordSession ─────────────────────────────────────────────────────────


class TestRecordSession:
    """Tests for POST /history/sessions."""

    def test_returns_201(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Valid body returns HTTP 201."""
        monkeypatch.setattr(
            "src.api.routers.history.StudySessionRepository",
            lambda _: _make_repo_mock(create_return=_make_session_record()),
        )
        response = client.post(
            "/history/sessions",
            json={"session_type": "exam", "topic": "Appeal"},
        )
        assert response.status_code == 201

    def test_response_contains_id(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Response body contains the UUID id field."""
        record = _make_session_record()
        monkeypatch.setattr(
            "src.api.routers.history.StudySessionRepository",
            lambda _: _make_repo_mock(create_return=record),
        )
        response = client.post(
            "/history/sessions", json={"session_type": "exam"}
        )
        assert "id" in response.json()

    def test_response_contains_session_type(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Response body contains session_type."""
        record = _make_session_record(session_type="chat")
        monkeypatch.setattr(
            "src.api.routers.history.StudySessionRepository",
            lambda _: _make_repo_mock(create_return=record),
        )
        response = client.post(
            "/history/sessions", json={"session_type": "chat"}
        )
        assert response.json()["session_type"] == "chat"

    def test_invalid_session_type_returns_422(self, client: TestClient) -> None:
        """session_type must be 'exam' or 'chat'."""
        response = client.post(
            "/history/sessions", json={"session_type": "quiz"}
        )
        assert response.status_code == 422

    def test_missing_session_type_returns_422(self, client: TestClient) -> None:
        """Missing required field returns HTTP 422."""
        response = client.post("/history/sessions", json={})
        assert response.status_code == 422

    def test_score_avg_out_of_range_returns_422(self, client: TestClient) -> None:
        """score_avg > 10 returns HTTP 422."""
        response = client.post(
            "/history/sessions",
            json={"session_type": "exam", "score_avg": 11.0},
        )
        assert response.status_code == 422

    def test_percentage_out_of_range_returns_422(self, client: TestClient) -> None:
        """percentage > 100 returns HTTP 422."""
        response = client.post(
            "/history/sessions",
            json={"session_type": "exam", "percentage": 101},
        )
        assert response.status_code == 422


# ── TestListSessions ──────────────────────────────────────────────────────────


class TestListSessions:
    """Tests for GET /history/sessions."""

    def test_returns_200(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns HTTP 200."""
        monkeypatch.setattr(
            "src.api.routers.history.StudySessionRepository",
            lambda _: _make_repo_mock(list_return=[]),
        )
        response = client.get("/history/sessions")
        assert response.status_code == 200

    def test_returns_list(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Response body is a list."""
        monkeypatch.setattr(
            "src.api.routers.history.StudySessionRepository",
            lambda _: _make_repo_mock(list_return=[]),
        )
        response = client.get("/history/sessions")
        assert isinstance(response.json(), list)

    def test_returns_sessions_from_repo(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sessions returned by the repo appear in the response."""
        records = [_make_session_record(session_type="exam")]
        monkeypatch.setattr(
            "src.api.routers.history.StudySessionRepository",
            lambda _: _make_repo_mock(list_return=records),
        )
        response = client.get("/history/sessions")
        assert len(response.json()) == 1
        assert response.json()[0]["session_type"] == "exam"

    def test_invalid_session_type_filter_returns_422(
        self, client: TestClient
    ) -> None:
        """Unsupported session_type query param returns 422."""
        response = client.get("/history/sessions?session_type=invalid")
        assert response.status_code == 422

    def test_limit_capped_at_200(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Limit > 200 is silently capped at 200."""
        repo_mock = _make_repo_mock(list_return=[])
        monkeypatch.setattr(
            "src.api.routers.history.StudySessionRepository",
            lambda _: repo_mock,
        )
        client.get("/history/sessions?limit=999")
        repo_mock.list_recent.assert_called_once()
        _, kwargs = repo_mock.list_recent.call_args
        assert kwargs.get("limit", 999) <= 200
