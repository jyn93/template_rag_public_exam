"""HTTP client helpers for the Streamlit frontend pages."""

from __future__ import annotations

import os

import httpx

__all__ = [
    "call_chat_api",
    "call_exam_evaluate_api",
    "call_exam_export_api",
    "call_exam_generate_api",
    "call_ingest_api",
    "call_history_list_api",
    "call_history_record_api",
    "extract_detail",
]

API_URL = os.getenv("API_URL", "http://localhost:8000")
_TIMEOUT = 60.0


# ── Ingestion ─────────────────────────────────────────────────────────────────


def call_ingest_api(
    file_bytes: bytes,
    filename: str,
    subject: str,
) -> tuple[dict[str, object], str]:
    """Call POST /ingest and return (result_dict, error_message).

    Uploads a document file as multipart form data together with the subject
    label. The backend validates the extension and size before indexing.

    Args:
        file_bytes: Raw file content to upload.
        filename: Original file name including extension (e.g. ``"tema1.pdf"``).
        subject: Human-readable subject label (e.g. ``"Administrative Law"``).

    Returns:
        Tuple of ``(result, error)``.  On success, *error* is an empty string
        and *result* contains ``total_documents``, ``total_chunks``,
        ``subject``, and ``filename``.  On failure, *result* is empty and
        *error* contains a human-readable message.
    """
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{API_URL}/ingest",
                files={"file": (filename, file_bytes)},
                data={"subject": subject},
            )
        response.raise_for_status()
        return dict(response.json()), ""
    except httpx.HTTPStatusError as exc:
        detail = extract_detail(exc.response)
        return {}, f"API error {exc.response.status_code}: {detail}"
    except httpx.RequestError as exc:
        return {}, f"Could not reach the API: {exc}"


# ── Chat ──────────────────────────────────────────────────────────────────────


def call_chat_api(
    query: str,
    subject: str,
    top_k: int,
) -> tuple[str, list[str], str]:
    """Call POST /chat and return (answer, sources, error_message).

    Args:
        query: The user's question.
        subject: Optional subject filter (empty string = no filter).
        top_k: Number of context chunks to retrieve.

    Returns:
        Tuple of ``(answer, sources, error)``.  On success, *error* is an
        empty string.  On failure, *answer* and *sources* are empty and
        *error* contains a human-readable message.
    """
    payload: dict[str, object] = {"query": query, "top_k": top_k}
    if subject.strip():
        payload["subject"] = subject.strip()

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            response = client.post(f"{API_URL}/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return str(data.get("answer", "")), list(data.get("sources", [])), ""
    except httpx.HTTPStatusError as exc:
        detail = extract_detail(exc.response)
        return "", [], f"API error {exc.response.status_code}: {detail}"
    except httpx.RequestError as exc:
        return "", [], f"Could not reach the API: {exc}"


# ── Exam generation ───────────────────────────────────────────────────────────


def call_exam_generate_api(
    query: str,
    subject: str,
    num_questions: int,
    exam_type: str,
    difficulty: str,
    top_k: int,
) -> tuple[dict[str, object], list[str], str]:
    """Call POST /exam/generate and return (exam_dict, sources, error_message).

    Args:
        query: Topic or concept to examine.
        subject: Optional subject label.
        num_questions: Number of questions to generate.
        exam_type: One of ``"test"``, ``"desarrollo"``, ``"mixto"``.
        difficulty: One of ``"facil"``, ``"media"``, ``"dificil"``.
        top_k: Number of context chunks to retrieve.

    Returns:
        Tuple of ``(exam, sources, error)``.  On success, *error* is empty.
    """
    payload: dict[str, object] = {
        "query": query,
        "num_questions": num_questions,
        "exam_type": exam_type,
        "difficulty": difficulty,
        "top_k": top_k,
    }
    if subject.strip():
        payload["subject"] = subject.strip()

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            response = client.post(f"{API_URL}/exam/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        return dict(data.get("exam", {})), list(data.get("sources", [])), ""
    except httpx.HTTPStatusError as exc:
        detail = extract_detail(exc.response)
        return {}, [], f"API error {exc.response.status_code}: {detail}"
    except httpx.RequestError as exc:
        return {}, [], f"Could not reach the API: {exc}"


# ── Exam export ───────────────────────────────────────────────────────────────


def call_exam_export_api(
    exam: dict[str, object],
    include_answers: bool = False,
) -> tuple[bytes, str]:
    """Call POST /exam/export and return (pdf_bytes, error_message).

    Args:
        exam: Structured exam dict as returned by :func:`call_exam_generate_api`.
        include_answers: If ``True``, request the server to append an answer key.

    Returns:
        Tuple of ``(pdf_bytes, error)``.  On success, *error* is empty and
        *pdf_bytes* contains the raw PDF content.  On failure, *pdf_bytes* is
        empty and *error* contains a human-readable message.
    """
    payload = {"exam": exam, "include_answers": include_answers}
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            response = client.post(f"{API_URL}/exam/export", json=payload)
        response.raise_for_status()
        return response.content, ""
    except httpx.HTTPStatusError as exc:
        detail = extract_detail(exc.response)
        return b"", f"API error {exc.response.status_code}: {detail}"
    except httpx.RequestError as exc:
        return b"", f"Could not reach the API: {exc}"


# ── Answer evaluation ─────────────────────────────────────────────────────────


def call_exam_evaluate_api(
    question: str,
    correct_answer: str,
    student_answer: str,
) -> tuple[dict[str, object], str]:
    """Call POST /exam/evaluate and return (result_dict, error_message).

    Args:
        question: The exam question.
        correct_answer: Reference correct answer.
        student_answer: The student's answer to evaluate.

    Returns:
        Tuple of ``(result, error)``.  On success, *error* is empty and
        *result* contains ``score``, ``is_correct``, ``feedback``,
        ``missing_points``, and ``strengths``.
    """
    payload = {
        "question": question,
        "correct_answer": correct_answer,
        "student_answer": student_answer,
    }

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            response = client.post(f"{API_URL}/exam/evaluate", json=payload)
        response.raise_for_status()
        return dict(response.json()), ""
    except httpx.HTTPStatusError as exc:
        detail = extract_detail(exc.response)
        return {}, f"API error {exc.response.status_code}: {detail}"
    except httpx.RequestError as exc:
        return {}, f"Could not reach the API: {exc}"


# ── Study history ─────────────────────────────────────────────────────────────


def call_history_record_api(
    session_type: str,
    topic: str | None = None,
    subject: str | None = None,
    score_avg: float | None = None,
    num_questions: int | None = None,
    num_correct: int | None = None,
    percentage: int | None = None,
) -> tuple[dict[str, object], str]:
    """Call POST /history/sessions and return (session_dict, error_message).

    Args:
        session_type: ``"exam"`` or ``"chat"``.
        topic: Topic or query text.
        subject: Subject label.
        score_avg: Average score per question (0–10).
        num_questions: Total number of questions.
        num_correct: Number of correct answers.
        percentage: Percentage correct (0–100).

    Returns:
        Tuple of ``(session, error)``.  On success, *error* is empty.
    """
    payload: dict[str, object] = {"session_type": session_type}
    if topic is not None:
        payload["topic"] = topic
    if subject is not None:
        payload["subject"] = subject
    if score_avg is not None:
        payload["score_avg"] = score_avg
    if num_questions is not None:
        payload["num_questions"] = num_questions
    if num_correct is not None:
        payload["num_correct"] = num_correct
    if percentage is not None:
        payload["percentage"] = percentage

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            response = client.post(f"{API_URL}/history/sessions", json=payload)
        response.raise_for_status()
        return dict(response.json()), ""
    except httpx.HTTPStatusError as exc:
        detail = extract_detail(exc.response)
        return {}, f"API error {exc.response.status_code}: {detail}"
    except httpx.RequestError as exc:
        return {}, f"Could not reach the API: {exc}"


def call_history_list_api(
    limit: int = 50,
    subject: str | None = None,
    session_type: str | None = None,
) -> tuple[list[dict[str, object]], str]:
    """Call GET /history/sessions and return (sessions_list, error_message).

    Args:
        limit: Maximum number of records to retrieve.
        subject: Optional subject filter.
        session_type: Optional type filter (``"exam"`` or ``"chat"``).

    Returns:
        Tuple of ``(sessions, error)``.  On success, *error* is empty.
    """
    params: dict[str, str | int] = {"limit": limit}
    if subject:
        params["subject"] = subject
    if session_type:
        params["session_type"] = session_type

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            response = client.get(f"{API_URL}/history/sessions", params=params)
        response.raise_for_status()
        return list(response.json()), ""
    except httpx.HTTPStatusError as exc:
        detail = extract_detail(exc.response)
        return [], f"API error {exc.response.status_code}: {detail}"
    except httpx.RequestError as exc:
        return [], f"Could not reach the API: {exc}"


# ── Helpers ───────────────────────────────────────────────────────────────────


def extract_detail(response: httpx.Response) -> str:
    """Extract the ``detail`` field from a JSON error response body.

    Args:
        response: The failed HTTP response.

    Returns:
        The ``detail`` string if present in JSON, otherwise the raw text.
    """
    try:
        return str(response.json().get("detail", response.text))
    except Exception:
        return response.text
