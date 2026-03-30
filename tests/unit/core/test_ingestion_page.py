"""Unit tests for the ingestion API client helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.frontend.api_client import call_ingest_api

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_response(
    status_code: int = 200,
    json_data: dict | None = None,
    text: str = "",
) -> MagicMock:
    """Build a minimal httpx.Response-like mock."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.text = text
    if json_data is not None:
        mock.json.return_value = json_data
    else:
        mock.json.side_effect = ValueError("no json")
    return mock


def _make_client_mock(response: MagicMock) -> MagicMock:
    """Return a context-manager mock wrapping a fake httpx.Client."""
    client_mock = MagicMock()
    client_mock.__enter__ = MagicMock(return_value=client_mock)
    client_mock.__exit__ = MagicMock(return_value=False)
    client_mock.post.return_value = response
    return client_mock


_SAMPLE_RESULT = {
    "total_documents": 3,
    "total_chunks": 42,
    "subject": "Administrative Law",
    "filename": "tema1.pdf",
}

# ── TestCallIngestApi ─────────────────────────────────────────────────────────


class TestCallIngestApi:
    """Tests for call_ingest_api."""

    def test_success_returns_result_dict(self):
        """On 202, returns the result dict and empty error."""
        resp = _make_response(json_data=_SAMPLE_RESULT)
        resp.raise_for_status = MagicMock()
        with patch(
            "src.frontend.api_client.httpx.Client",
            return_value=_make_client_mock(resp),
        ):
            result, error = call_ingest_api(
                b"pdf bytes", "tema1.pdf", "Administrative Law"
            )

        assert result["total_chunks"] == 42
        assert result["total_documents"] == 3
        assert error == ""

    def test_success_error_string_is_empty(self):
        """On success, the error string is empty."""
        resp = _make_response(json_data=_SAMPLE_RESULT)
        resp.raise_for_status = MagicMock()
        with patch(
            "src.frontend.api_client.httpx.Client",
            return_value=_make_client_mock(resp),
        ):
            _, error = call_ingest_api(b"bytes", "doc.txt", "History")

        assert error == ""

    def test_sends_file_as_multipart(self):
        """File bytes and filename are sent as multipart files."""
        resp = _make_response(json_data=_SAMPLE_RESULT)
        resp.raise_for_status = MagicMock()
        client_mock = _make_client_mock(resp)
        with patch("src.frontend.api_client.httpx.Client", return_value=client_mock):
            call_ingest_api(b"content", "notes.pdf", "Civil Law")

        call_kwargs = client_mock.post.call_args.kwargs
        files = call_kwargs["files"]
        assert "file" in files
        assert files["file"][0] == "notes.pdf"
        assert files["file"][1] == b"content"

    def test_sends_subject_as_form_data(self):
        """Subject is sent in the form data field."""
        resp = _make_response(json_data=_SAMPLE_RESULT)
        resp.raise_for_status = MagicMock()
        client_mock = _make_client_mock(resp)
        with patch("src.frontend.api_client.httpx.Client", return_value=client_mock):
            call_ingest_api(b"bytes", "doc.pdf", "Constitutional Law")

        call_kwargs = client_mock.post.call_args.kwargs
        assert call_kwargs["data"]["subject"] == "Constitutional Law"

    def test_posts_to_ingest_endpoint(self):
        """Request is sent to the /ingest endpoint."""
        resp = _make_response(json_data=_SAMPLE_RESULT)
        resp.raise_for_status = MagicMock()
        client_mock = _make_client_mock(resp)
        with patch("src.frontend.api_client.httpx.Client", return_value=client_mock):
            call_ingest_api(b"bytes", "doc.pdf", "Law")

        url = client_mock.post.call_args.args[0]
        assert url.endswith("/ingest")

    def test_http_status_error_returns_error_string(self):
        """HTTPStatusError is converted to ({}, error_message)."""
        resp = _make_response(status_code=422, json_data={"detail": "Unsupported file"})
        request = MagicMock()
        exc = httpx.HTTPStatusError("422", request=request, response=resp)
        resp.raise_for_status = MagicMock(side_effect=exc)
        with patch(
            "src.frontend.api_client.httpx.Client",
            return_value=_make_client_mock(resp),
        ):
            result, error = call_ingest_api(b"bytes", "doc.exe", "Law")

        assert result == {}
        assert "422" in error

    def test_http_413_returns_size_error(self):
        """413 status returns an error referencing the status code."""
        resp = _make_response(status_code=413, json_data={"detail": "File too large"})
        request = MagicMock()
        exc = httpx.HTTPStatusError("413", request=request, response=resp)
        resp.raise_for_status = MagicMock(side_effect=exc)
        with patch(
            "src.frontend.api_client.httpx.Client",
            return_value=_make_client_mock(resp),
        ):
            result, error = call_ingest_api(b"huge bytes", "big.pdf", "Law")

        assert result == {}
        assert "413" in error

    def test_request_error_returns_error_string(self):
        """Network failure returns a user-friendly error message."""
        client_mock = MagicMock()
        client_mock.__enter__ = MagicMock(return_value=client_mock)
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.side_effect = httpx.RequestError("connection refused")
        with patch("src.frontend.api_client.httpx.Client", return_value=client_mock):
            result, error = call_ingest_api(b"bytes", "doc.pdf", "Law")

        assert result == {}
        assert "reach" in error or "API" in error

    def test_returns_empty_dict_on_failure(self):
        """On any error, result dict is empty."""
        client_mock = MagicMock()
        client_mock.__enter__ = MagicMock(return_value=client_mock)
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.side_effect = httpx.RequestError("timeout")
        with patch("src.frontend.api_client.httpx.Client", return_value=client_mock):
            result, _ = call_ingest_api(b"bytes", "doc.pdf", "Law")

        assert result == {}

    def test_uses_extended_timeout(self):
        """Ingestion uses a longer timeout than other API calls (120s)."""
        resp = _make_response(json_data=_SAMPLE_RESULT)
        resp.raise_for_status = MagicMock()
        with patch(
            "src.frontend.api_client.httpx.Client"
        ) as mock_client_cls:
            mock_client_cls.return_value = _make_client_mock(resp)
            call_ingest_api(b"bytes", "doc.pdf", "Law")

        _, kwargs = mock_client_cls.call_args
        assert kwargs.get("timeout", 0) >= 60


# ── TestIngestApiClientIntegration ────────────────────────────────────────────


class TestIngestApiClientIntegration:
    """Higher-level scenarios for call_ingest_api."""

    def test_pdf_and_txt_both_accepted(self):
        """Both PDF and TXT filenames are sent without modification."""
        resp = _make_response(json_data=_SAMPLE_RESULT)
        resp.raise_for_status = MagicMock()

        for filename in ("study.pdf", "notes.txt"):
            client_mock = _make_client_mock(resp)
            with patch(
                "src.frontend.api_client.httpx.Client", return_value=client_mock
            ):
                result, error = call_ingest_api(b"content", filename, "Law")

            assert error == ""
            sent_filename = client_mock.post.call_args.kwargs["files"]["file"][0]
            assert sent_filename == filename

    @pytest.mark.parametrize(
        "status_code,detail",
        [
            (500, "Internal server error"),
            (422, "Unsupported file type"),
            (413, "File too large"),
        ],
    )
    def test_all_error_status_codes_return_error(self, status_code: int, detail: str):
        """All backend error status codes result in a non-empty error string."""
        resp = _make_response(status_code=status_code, json_data={"detail": detail})
        request = MagicMock()
        exc = httpx.HTTPStatusError(str(status_code), request=request, response=resp)
        resp.raise_for_status = MagicMock(side_effect=exc)
        with patch(
            "src.frontend.api_client.httpx.Client",
            return_value=_make_client_mock(resp),
        ):
            result, error = call_ingest_api(b"bytes", "doc.pdf", "Law")

        assert result == {}
        assert error != ""
