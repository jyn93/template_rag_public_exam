"""Unit tests for the ingestion router (POST /ingest)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_ingestion_pipeline
from src.api.main import app
from src.core.exceptions import IngestionError, StorageError, VectorStoreError

# ── Helpers ───────────────────────────────────────────────────────────────────

_INGEST_RESULT = {
    "total_documents": 3,
    "total_chunks": 12,
    "subject": "Civil Law",
}

PDF_BYTES = b"%PDF-1.4 fake pdf content"
TXT_BYTES = b"Some plain text content."


def _make_pipeline_mock(
    result: dict | None = None, exc: Exception | None = None
) -> MagicMock:
    """Return an IngestionPipeline mock."""
    pipeline = MagicMock()
    if exc is not None:
        pipeline.ingest = AsyncMock(side_effect=exc)
    else:
        pipeline.ingest = AsyncMock(return_value=result or _INGEST_RESULT)
    return pipeline


def _override_pipeline(pipeline: MagicMock):
    """Apply dependency override for get_ingestion_pipeline."""
    app.dependency_overrides[get_ingestion_pipeline] = lambda: pipeline


def _clear_overrides() -> None:
    app.dependency_overrides.pop(get_ingestion_pipeline, None)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def cleanup_overrides():
    """Ensure dependency overrides are cleaned up after each test."""
    yield
    _clear_overrides()


@pytest.fixture
def client() -> TestClient:
    """Synchronous test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """Default pipeline mock that returns a successful ingest result."""
    pipeline = _make_pipeline_mock()
    _override_pipeline(pipeline)
    return pipeline


# ── TestIngestSuccess ─────────────────────────────────────────────────────────


class TestIngestSuccess:
    """Tests for successful POST /ingest requests."""

    def test_pdf_upload_returns_202(self, client, mock_pipeline):
        """PDF upload returns HTTP 202 Accepted."""
        response = client.post(
            "/ingest",
            data={"subject": "Civil Law"},
            files={"file": ("doc.pdf", PDF_BYTES, "application/pdf")},
        )
        assert response.status_code == 202

    def test_txt_upload_returns_202(self, client, mock_pipeline):
        """TXT upload returns HTTP 202 Accepted."""
        response = client.post(
            "/ingest",
            data={"subject": "Admin Law"},
            files={"file": ("notes.txt", TXT_BYTES, "text/plain")},
        )
        assert response.status_code == 202

    def test_response_contains_total_chunks(self, client, mock_pipeline):
        """Response body includes total_chunks from the pipeline result."""
        response = client.post(
            "/ingest",
            data={"subject": "Civil Law"},
            files={"file": ("doc.pdf", PDF_BYTES, "application/pdf")},
        )
        assert response.json()["total_chunks"] == _INGEST_RESULT["total_chunks"]

    def test_response_contains_filename(self, client, mock_pipeline):
        """Response body echoes the original filename."""
        response = client.post(
            "/ingest",
            data={"subject": "Civil Law"},
            files={"file": ("my_doc.pdf", PDF_BYTES, "application/pdf")},
        )
        assert response.json()["filename"] == "my_doc.pdf"

    def test_pipeline_ingest_called_once(self, client, mock_pipeline):
        """pipeline.ingest is called exactly once per request."""
        client.post(
            "/ingest",
            data={"subject": "Civil Law"},
            files={"file": ("doc.pdf", PDF_BYTES, "application/pdf")},
        )
        mock_pipeline.ingest.assert_called_once()

    def test_pipeline_receives_correct_subject(self, client, mock_pipeline):
        """pipeline.ingest is called with the provided subject."""
        client.post(
            "/ingest",
            data={"subject": "Criminal Law"},
            files={"file": ("doc.pdf", PDF_BYTES, "application/pdf")},
        )
        _, kwargs = mock_pipeline.ingest.call_args
        assert kwargs["subject_name"] == "Criminal Law"


# ── TestIngestValidation ──────────────────────────────────────────────────────


class TestIngestValidation:
    """Tests for request validation in POST /ingest."""

    def test_unsupported_extension_returns_422(self, client):
        """Uploading a .docx file returns HTTP 422."""
        _override_pipeline(_make_pipeline_mock())
        response = client.post(
            "/ingest",
            data={"subject": "Civil Law"},
            files={"file": ("doc.docx", b"content", "application/octet-stream")},
        )
        assert response.status_code == 422

    def test_unsupported_extension_error_message(self, client):
        """Error message mentions unsupported file type."""
        _override_pipeline(_make_pipeline_mock())
        response = client.post(
            "/ingest",
            data={"subject": "Civil Law"},
            files={"file": ("doc.docx", b"content", "application/octet-stream")},
        )
        assert "Unsupported file type" in response.json()["detail"]

    def test_empty_subject_returns_422(self, client):
        """Blank subject string returns HTTP 422."""
        _override_pipeline(_make_pipeline_mock())
        response = client.post(
            "/ingest",
            data={"subject": "   "},
            files={"file": ("doc.pdf", PDF_BYTES, "application/pdf")},
        )
        assert response.status_code == 422

    def test_file_too_large_returns_413(self, client):
        """Files exceeding 50 MB return HTTP 413."""
        _override_pipeline(_make_pipeline_mock())
        big_content = b"x" * (51 * 1024 * 1024)
        response = client.post(
            "/ingest",
            data={"subject": "Civil Law"},
            files={"file": ("huge.pdf", big_content, "application/pdf")},
        )
        assert response.status_code == 413


# ── TestIngestErrorHandling ───────────────────────────────────────────────────


class TestIngestErrorHandling:
    """Tests for domain exception mapping in POST /ingest."""

    def test_ingestion_error_returns_500(self, client):
        """IngestionError from pipeline returns HTTP 500."""
        pipeline = _make_pipeline_mock(exc=IngestionError("parse failed"))
        _override_pipeline(pipeline)
        response = client.post(
            "/ingest",
            data={"subject": "Civil Law"},
            files={"file": ("doc.pdf", PDF_BYTES, "application/pdf")},
        )
        assert response.status_code == 500

    def test_storage_error_returns_500(self, client):
        """StorageError from pipeline returns HTTP 500."""
        pipeline = _make_pipeline_mock(exc=StorageError("minio down"))
        _override_pipeline(pipeline)
        response = client.post(
            "/ingest",
            data={"subject": "Civil Law"},
            files={"file": ("doc.pdf", PDF_BYTES, "application/pdf")},
        )
        assert response.status_code == 500

    def test_vector_store_error_returns_500(self, client):
        """VectorStoreError from pipeline returns HTTP 500."""
        pipeline = _make_pipeline_mock(exc=VectorStoreError("qdrant unreachable"))
        _override_pipeline(pipeline)
        response = client.post(
            "/ingest",
            data={"subject": "Civil Law"},
            files={"file": ("doc.pdf", PDF_BYTES, "application/pdf")},
        )
        assert response.status_code == 500

    def test_value_error_returns_422(self, client):
        """ValueError from pipeline (e.g. unsupported loader) returns 422."""
        pipeline = _make_pipeline_mock(exc=ValueError("no loader for .xyz"))
        _override_pipeline(pipeline)
        response = client.post(
            "/ingest",
            data={"subject": "Civil Law"},
            files={"file": ("doc.pdf", PDF_BYTES, "application/pdf")},
        )
        assert response.status_code == 422

    def test_ingestion_error_detail_contains_message(self, client):
        """HTTP 500 detail includes the IngestionError message."""
        pipeline = _make_pipeline_mock(exc=IngestionError("corrupted file"))
        _override_pipeline(pipeline)
        response = client.post(
            "/ingest",
            data={"subject": "Civil Law"},
            files={"file": ("doc.pdf", PDF_BYTES, "application/pdf")},
        )
        assert "corrupted file" in response.json()["detail"]
