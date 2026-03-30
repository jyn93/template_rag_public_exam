"""Pure helper functions for the exam simulation page.

Extracted into a dedicated module so they can be unit-tested
independently without importing Streamlit.
"""

from __future__ import annotations


def format_remaining(seconds: int) -> str:
    """Format a number of seconds as a MM:SS string.

    Args:
        seconds: Non-negative number of seconds remaining.

    Returns:
        Zero-padded ``"MM:SS"`` string (e.g. ``"04:07"``).
    """
    seconds = max(0, seconds)
    minutes, secs = divmod(seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def compute_score_summary(
    questions: list[dict[str, object]],
    evaluations: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Compute a score summary from evaluated answers.

    Args:
        questions: List of question dicts (each must have an ``"id"`` key).
        evaluations: Mapping of question id → evaluation result dict
            (must contain ``"score"`` and ``"is_correct"`` keys).

    Returns:
        Dict with keys:
        - ``total``: total number of questions.
        - ``answered``: number of questions that were evaluated.
        - ``correct``: number of questions marked as correct.
        - ``score_sum``: sum of numeric scores.
        - ``score_avg``: average score (0.0 if no answers).
        - ``percentage``: percentage of correct answers (0–100).
    """
    total = len(questions)
    answered = len(evaluations)
    correct = sum(
        1 for ev in evaluations.values() if bool(ev.get("is_correct", False))
    )
    score_sum: float = sum(
        float(ev.get("score") or 0)  # type: ignore[arg-type, misc]
        for ev in evaluations.values()
    )
    score_avg = round(score_sum / answered, 1) if answered > 0 else 0.0
    percentage = round((correct / total) * 100) if total > 0 else 0

    return {
        "total": total,
        "answered": answered,
        "correct": correct,
        "score_sum": score_sum,
        "score_avg": score_avg,
        "percentage": percentage,
    }
