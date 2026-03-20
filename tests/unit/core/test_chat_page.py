"""Unit tests for the frontend API client helpers (chat)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.frontend.api_client import call_chat_api, extract_detail

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


# ── TestExtractDetail ─────────────────────────────────────────────────────────


class TestExtractDetail:
    """Tests for extract_detail."""

    def test_returns_detail_from_json(self):
        """Returns the 'detail' field from a JSON response."""
        response = _make_response(json_data={"detail": "Not found"})
        assert extract_detail(response) == "Not found"

    def test_falls_back_to_text_when_no_detail_key(self):
        """Falls back to the text representation when 'detail' key is absent."""
        response = _make_response(json_data={"error": "bad"})
        result = extract_detail(response)
        # json().get("detail", response.text) → response.text (since no detail)
        assert result == response.text

    def test_falls_back_to_text_on_json_parse_error(self):
        """Returns response.text when JSON parsing fails."""
        response = _make_response(text="plain error text")
        response.json.side_effect = ValueError("not json")
        assert extract_detail(response) == "plain error text"


# ── TestCallChatApi ───────────────────────────────────────────────────────────


class TestCallChatApi:
    """Tests for call_chat_api."""

    def test_success_returns_answer_and_sources(self):
        """On 200, returns (answer, sources, '') tuple."""
        resp = _make_response(
            json_data={"answer": "An appeal is…", "sources": ["chunk1", "chunk2"]}
        )
        resp.raise_for_status = MagicMock()
        with patch(
            "src.frontend.api_client.httpx.Client",
            return_value=_make_client_mock(resp),
        ):
            answer, sources, error = call_chat_api("What is an appeal?", "", 5)
        assert answer == "An appeal is…"
        assert sources == ["chunk1", "chunk2"]
        assert error == ""

    def test_success_error_string_is_empty(self):
        """On 200, error string is empty."""
        resp = _make_response(json_data={"answer": "A.", "sources": []})
        resp.raise_for_status = MagicMock()
        with patch(
            "src.frontend.api_client.httpx.Client",
            return_value=_make_client_mock(resp),
        ):
            _, _, error = call_chat_api("query", "", 5)
        assert error == ""

    def test_subject_included_in_payload_when_provided(self):
        """Subject is added to payload when non-empty."""
        resp = _make_response(json_data={"answer": "A.", "sources": []})
        resp.raise_for_status = MagicMock()
        client_mock = _make_client_mock(resp)
        with patch("src.frontend.api_client.httpx.Client", return_value=client_mock):
            call_chat_api("query", "Civil Law", 5)
        call_kwargs = client_mock.post.call_args.kwargs
        assert call_kwargs["json"]["subject"] == "Civil Law"

    def test_blank_subject_excluded_from_payload(self):
        """Blank subject is not included in the request payload."""
        resp = _make_response(json_data={"answer": "A.", "sources": []})
        resp.raise_for_status = MagicMock()
        client_mock = _make_client_mock(resp)
        with patch("src.frontend.api_client.httpx.Client", return_value=client_mock):
            call_chat_api("query", "   ", 5)
        call_kwargs = client_mock.post.call_args.kwargs
        assert "subject" not in call_kwargs["json"]

    def test_http_status_error_returns_error_string(self):
        """HTTPStatusError is converted to (empty, empty, error_message)."""
        import httpx

        resp = _make_response(status_code=500, json_data={"detail": "LLM down"})
        request = MagicMock()
        exc = httpx.HTTPStatusError("500", request=request, response=resp)
        resp.raise_for_status = MagicMock(side_effect=exc)
        with patch(
            "src.frontend.api_client.httpx.Client",
            return_value=_make_client_mock(resp),
        ):
            answer, sources, error = call_chat_api("query", "", 5)
        assert answer == ""
        assert sources == []
        assert "500" in error

    def test_request_error_returns_error_string(self):
        """RequestError (network failure) returns a user-friendly error."""
        import httpx

        client_mock = MagicMock()
        client_mock.__enter__ = MagicMock(return_value=client_mock)
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.side_effect = httpx.RequestError("connection refused")
        with patch("src.frontend.api_client.httpx.Client", return_value=client_mock):
            answer, sources, error = call_chat_api("query", "", 5)
        assert answer == ""
        assert sources == []
        assert "API" in error or "reach" in error

    def test_top_k_forwarded_in_payload(self):
        """top_k value is included in the POST payload."""
        resp = _make_response(json_data={"answer": "A.", "sources": []})
        resp.raise_for_status = MagicMock()
        client_mock = _make_client_mock(resp)
        with patch("src.frontend.api_client.httpx.Client", return_value=client_mock):
            call_chat_api("query", "", 10)
        call_kwargs = client_mock.post.call_args.kwargs
        assert call_kwargs["json"]["top_k"] == 10
