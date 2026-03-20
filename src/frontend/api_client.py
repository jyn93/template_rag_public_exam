"""HTTP client helpers for the Streamlit frontend pages."""

from __future__ import annotations

import os

import httpx

__all__ = [
    "call_chat_api",
    "call_exam_evaluate_api",
    "call_exam_generate_api",
    "extract_detail",
]

API_URL = os.getenv("API_URL", "http://localhost:8000")
_TIMEOUT = 60.0


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
