"""Unit tests for the frontend API client helpers (exam)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.frontend.api_client import call_exam_evaluate_api, call_exam_generate_api

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


# ── TestCallExamGenerateApi ───────────────────────────────────────────────────


class TestCallExamGenerateApi:
    """Tests for call_exam_generate_api."""

    def test_success_returns_exam_and_sources(self):
        """On 200, returns (exam_dict, sources, '') tuple."""
        exam_payload = {"questions": [{"id": 1, "question": "What is X?"}]}
        resp = _make_response(
            json_data={"exam": exam_payload, "sources": ["chunk1", "chunk2"]}
        )
        resp.raise_for_status = MagicMock()
        with patch(
            "src.frontend.api_client.httpx.Client",
            return_value=_make_client_mock(resp),
        ):
            exam, sources, error = call_exam_generate_api(
                "appeal", "", 3, "test", "media", 5
            )
        assert exam == exam_payload
        assert sources == ["chunk1", "chunk2"]
        assert error == ""

    def test_error_string_empty_on_success(self):
        """On 200, error string is empty."""
        resp = _make_response(json_data={"exam": {}, "sources": []})
        resp.raise_for_status = MagicMock()
        with patch(
            "src.frontend.api_client.httpx.Client",
            return_value=_make_client_mock(resp),
        ):
            _, _, error = call_exam_generate_api("topic", "", 3, "test", "media", 5)
        assert error == ""

    def test_subject_included_in_payload_when_provided(self):
        """Non-empty subject is forwarded in the request payload."""
        resp = _make_response(json_data={"exam": {}, "sources": []})
        resp.raise_for_status = MagicMock()
        client_mock = _make_client_mock(resp)
        with patch("src.frontend.api_client.httpx.Client", return_value=client_mock):
            call_exam_generate_api("topic", "Civil Law", 3, "test", "media", 5)
        payload = client_mock.post.call_args.kwargs["json"]
        assert payload["subject"] == "Civil Law"

    def test_blank_subject_excluded_from_payload(self):
        """Blank subject is not included in the request payload."""
        resp = _make_response(json_data={"exam": {}, "sources": []})
        resp.raise_for_status = MagicMock()
        client_mock = _make_client_mock(resp)
        with patch("src.frontend.api_client.httpx.Client", return_value=client_mock):
            call_exam_generate_api("topic", "   ", 3, "test", "media", 5)
        payload = client_mock.post.call_args.kwargs["json"]
        assert "subject" not in payload

    def test_exam_params_forwarded_in_payload(self):
        """num_questions, exam_type, difficulty, top_k are all in the payload."""
        resp = _make_response(json_data={"exam": {}, "sources": []})
        resp.raise_for_status = MagicMock()
        client_mock = _make_client_mock(resp)
        with patch("src.frontend.api_client.httpx.Client", return_value=client_mock):
            call_exam_generate_api("topic", "", 7, "desarrollo", "dificil", 10)
        payload = client_mock.post.call_args.kwargs["json"]
        assert payload["num_questions"] == 7
        assert payload["exam_type"] == "desarrollo"
        assert payload["difficulty"] == "dificil"
        assert payload["top_k"] == 10

    def test_http_status_error_returns_error_string(self):
        """HTTPStatusError returns ({}, [], error_message)."""
        import httpx

        resp = _make_response(status_code=500, json_data={"detail": "LLM down"})
        request = MagicMock()
        exc = httpx.HTTPStatusError("500", request=request, response=resp)
        resp.raise_for_status = MagicMock(side_effect=exc)
        with patch(
            "src.frontend.api_client.httpx.Client",
            return_value=_make_client_mock(resp),
        ):
            exam, sources, error = call_exam_generate_api(
                "topic", "", 3, "test", "media", 5
            )
        assert exam == {}
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
            exam, sources, error = call_exam_generate_api(
                "topic", "", 3, "test", "media", 5
            )
        assert exam == {}
        assert sources == []
        assert "API" in error or "reach" in error

    def test_missing_exam_key_returns_empty_dict(self):
        """If the response has no 'exam' key, returns empty dict."""
        resp = _make_response(json_data={"sources": []})
        resp.raise_for_status = MagicMock()
        with patch(
            "src.frontend.api_client.httpx.Client",
            return_value=_make_client_mock(resp),
        ):
            exam, _, _ = call_exam_generate_api("topic", "", 3, "test", "media", 5)
        assert exam == {}


# ── TestCallExamEvaluateApi ───────────────────────────────────────────────────


class TestCallExamEvaluateApi:
    """Tests for call_exam_evaluate_api."""

    def test_success_returns_result_dict(self):
        """On 200, returns (result_dict, '') tuple."""
        eval_data = {
            "score": 8,
            "is_correct": True,
            "feedback": "Well done.",
            "missing_points": [],
            "strengths": ["Cited article 24."],
        }
        resp = _make_response(json_data=eval_data)
        resp.raise_for_status = MagicMock()
        with patch(
            "src.frontend.api_client.httpx.Client",
            return_value=_make_client_mock(resp),
        ):
            result, error = call_exam_evaluate_api("Q?", "Correct.", "Student answer.")
        assert result["score"] == 8
        assert result["is_correct"] is True
        assert result["feedback"] == "Well done."
        assert error == ""

    def test_error_string_empty_on_success(self):
        """On 200, error string is empty."""
        resp = _make_response(
            json_data={"score": 5, "is_correct": False, "feedback": "Ok."}
        )
        resp.raise_for_status = MagicMock()
        with patch(
            "src.frontend.api_client.httpx.Client",
            return_value=_make_client_mock(resp),
        ):
            _, error = call_exam_evaluate_api("Q?", "Correct.", "Student.")
        assert error == ""

    def test_payload_fields_forwarded(self):
        """question, correct_answer, student_answer are all in the payload."""
        resp = _make_response(json_data={"score": 7})
        resp.raise_for_status = MagicMock()
        client_mock = _make_client_mock(resp)
        with patch("src.frontend.api_client.httpx.Client", return_value=client_mock):
            call_exam_evaluate_api(
                "What is habeas corpus?", "A writ...", "It protects liberty."
            )
        payload = client_mock.post.call_args.kwargs["json"]
        assert payload["question"] == "What is habeas corpus?"
        assert payload["correct_answer"] == "A writ..."
        assert payload["student_answer"] == "It protects liberty."

    def test_http_status_error_returns_error_string(self):
        """HTTPStatusError returns ({}, error_message)."""
        import httpx

        resp = _make_response(status_code=422, json_data={"detail": "empty question"})
        request = MagicMock()
        exc = httpx.HTTPStatusError("422", request=request, response=resp)
        resp.raise_for_status = MagicMock(side_effect=exc)
        with patch(
            "src.frontend.api_client.httpx.Client",
            return_value=_make_client_mock(resp),
        ):
            result, error = call_exam_evaluate_api("", "Correct.", "Student.")
        assert result == {}
        assert "422" in error

    def test_request_error_returns_error_string(self):
        """RequestError returns ({}, user-friendly error)."""
        import httpx

        client_mock = MagicMock()
        client_mock.__enter__ = MagicMock(return_value=client_mock)
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.side_effect = httpx.RequestError("timeout")
        with patch("src.frontend.api_client.httpx.Client", return_value=client_mock):
            result, error = call_exam_evaluate_api("Q?", "Correct.", "Student.")
        assert result == {}
        assert "API" in error or "reach" in error
