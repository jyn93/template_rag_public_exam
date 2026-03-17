"""Unit tests for FastAPI application endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.dependencies import SettingsDep
from src.api.main import app
from src.core.config.settings import Settings


def _override_settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


app.dependency_overrides[_override_settings] = _override_settings

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self) -> None:
        """Verify /health responds with HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_status_ok(self) -> None:
        """Verify /health body contains status=ok."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_response_has_uptime(self) -> None:
        """Verify /health body includes uptime_seconds as a number."""
        response = client.get("/health")
        data = response.json()
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], float)

    def test_health_response_has_service_name(self) -> None:
        """Verify /health body includes the service name."""
        response = client.get("/health")
        data = response.json()
        assert "service" in data
        assert isinstance(data["service"], str)


class TestRootEndpoint:
    """Tests for GET /."""

    def test_root_returns_200(self) -> None:
        """Verify / responds with HTTP 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_response_has_message(self) -> None:
        """Verify / body contains a welcome message."""
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert isinstance(data["message"], str)

    def test_root_response_has_docs_url(self) -> None:
        """Verify / body contains the docs URL."""
        response = client.get("/")
        data = response.json()
        assert data["docs"] == "/docs"


class TestDependencies:
    """Tests for FastAPI dependency aliases."""

    def test_settings_dep_is_annotated(self) -> None:
        """Verify SettingsDep is a valid type annotation (not None)."""
        assert SettingsDep is not None
