"""Unit tests for the simulacro page pure helper functions."""

from __future__ import annotations

import pytest

from src.frontend.pages._simulacro_helpers import (
    compute_score_summary,
    format_remaining,
)

# ── TestFormatRemaining ───────────────────────────────────────────────────────


class TestFormatRemaining:
    """Tests for format_remaining."""

    def test_zero_seconds(self) -> None:
        """Zero seconds formats as 00:00."""
        assert format_remaining(0) == "00:00"

    def test_negative_seconds_clamped_to_zero(self) -> None:
        """Negative input is clamped to 00:00."""
        assert format_remaining(-5) == "00:00"

    def test_one_second(self) -> None:
        """One second formats as 00:01."""
        assert format_remaining(1) == "00:01"

    def test_exactly_one_minute(self) -> None:
        """60 seconds formats as 01:00."""
        assert format_remaining(60) == "01:00"

    def test_one_minute_one_second(self) -> None:
        """61 seconds formats as 01:01."""
        assert format_remaining(61) == "01:01"

    def test_large_value(self) -> None:
        """3599 seconds = 59:59."""
        assert format_remaining(3599) == "59:59"

    def test_ninety_minutes(self) -> None:
        """5400 seconds = 90:00."""
        assert format_remaining(5400) == "90:00"

    def test_zero_padded_minutes(self) -> None:
        """Minutes below 10 are zero-padded."""
        result = format_remaining(125)  # 2 min 5 sec
        assert result == "02:05"

    def test_zero_padded_seconds(self) -> None:
        """Seconds below 10 are zero-padded."""
        result = format_remaining(603)  # 10 min 3 sec
        assert result == "10:03"

    def test_format_has_colon_separated_seconds(self) -> None:
        """Return value always ends with a colon and exactly two digit seconds."""
        import re

        for secs in (0, 30, 60, 3600, 7200):
            result = format_remaining(secs)
            assert re.fullmatch(r"\d+:\d{2}", result), (
                f"Bad format for {secs}s: {result}"
            )


# ── TestComputeScoreSummary ───────────────────────────────────────────────────


def _make_questions(n: int) -> list[dict[str, object]]:
    """Build a list of n minimal question dicts with sequential ids."""
    return [{"id": str(i + 1)} for i in range(n)]


def _make_evaluations(
    ids: list[str],
    score: float = 10.0,
    is_correct: bool = True,
) -> dict[str, dict[str, object]]:
    """Build a uniform evaluation dict for the given ids."""
    return {qid: {"score": score, "is_correct": is_correct} for qid in ids}


class TestComputeScoreSummary:
    """Tests for compute_score_summary."""

    def test_total_equals_question_count(self) -> None:
        """total reflects the number of questions, not evaluations."""
        questions = _make_questions(5)
        evals = _make_evaluations(["1", "2"])
        summary = compute_score_summary(questions, evals)
        assert summary["total"] == 5

    def test_answered_equals_evaluation_count(self) -> None:
        """answered reflects how many questions were evaluated."""
        questions = _make_questions(5)
        evals = _make_evaluations(["1", "3"])
        summary = compute_score_summary(questions, evals)
        assert summary["answered"] == 2

    def test_correct_counts_is_correct_true(self) -> None:
        """correct counts only evaluations with is_correct=True."""
        questions = _make_questions(4)
        evals: dict[str, dict[str, object]] = {
            "1": {"score": 10.0, "is_correct": True},
            "2": {"score": 0.0, "is_correct": False},
            "3": {"score": 5.0, "is_correct": True},
        }
        summary = compute_score_summary(questions, evals)
        assert summary["correct"] == 2

    def test_score_sum_adds_all_scores(self) -> None:
        """score_sum is the arithmetic sum of all evaluation scores."""
        questions = _make_questions(3)
        evals: dict[str, dict[str, object]] = {
            "1": {"score": 7.0, "is_correct": True},
            "2": {"score": 3.0, "is_correct": False},
            "3": {"score": 5.5, "is_correct": True},
        }
        summary = compute_score_summary(questions, evals)
        assert summary["score_sum"] == pytest.approx(15.5)

    def test_score_avg_is_mean_of_scores(self) -> None:
        """score_avg is the rounded mean across answered questions."""
        questions = _make_questions(2)
        evals: dict[str, dict[str, object]] = {
            "1": {"score": 8.0, "is_correct": True},
            "2": {"score": 6.0, "is_correct": True},
        }
        summary = compute_score_summary(questions, evals)
        assert summary["score_avg"] == pytest.approx(7.0)

    def test_score_avg_is_zero_when_no_answers(self) -> None:
        """score_avg defaults to 0.0 when no answers were evaluated."""
        questions = _make_questions(3)
        summary = compute_score_summary(questions, {})
        assert summary["score_avg"] == 0.0

    def test_percentage_based_on_total_not_answered(self) -> None:
        """percentage = correct / total * 100, not correct / answered."""
        questions = _make_questions(4)
        evals: dict[str, dict[str, object]] = {
            "1": {"score": 10.0, "is_correct": True},
            "2": {"score": 10.0, "is_correct": True},
        }
        summary = compute_score_summary(questions, evals)
        # 2 correct out of 4 total = 50%
        assert summary["percentage"] == 50

    def test_percentage_zero_for_no_questions(self) -> None:
        """percentage is 0 when there are no questions."""
        summary = compute_score_summary([], {})
        assert summary["percentage"] == 0

    def test_all_correct_gives_100_percent(self) -> None:
        """All questions answered correctly yields 100%."""
        questions = _make_questions(3)
        evals = _make_evaluations(["1", "2", "3"], score=10.0, is_correct=True)
        summary = compute_score_summary(questions, evals)
        assert summary["percentage"] == 100

    def test_none_correct_gives_0_percent(self) -> None:
        """All questions answered incorrectly yields 0%."""
        questions = _make_questions(3)
        evals = _make_evaluations(["1", "2", "3"], score=0.0, is_correct=False)
        summary = compute_score_summary(questions, evals)
        assert summary["percentage"] == 0
        assert summary["correct"] == 0

    def test_empty_questions_and_evaluations(self) -> None:
        """All fields are zero/empty when inputs are empty."""
        summary = compute_score_summary([], {})
        assert summary["total"] == 0
        assert summary["answered"] == 0
        assert summary["correct"] == 0
        assert summary["score_sum"] == 0.0
        assert summary["score_avg"] == 0.0
        assert summary["percentage"] == 0

    def test_missing_score_defaults_to_zero(self) -> None:
        """Evaluations without 'score' key are treated as score=0."""
        questions = _make_questions(2)
        evals: dict[str, dict[str, object]] = {
            "1": {"is_correct": True},
            "2": {"score": 8.0, "is_correct": True},
        }
        summary = compute_score_summary(questions, evals)
        assert summary["score_sum"] == pytest.approx(8.0)

    def test_missing_is_correct_defaults_to_false(self) -> None:
        """Evaluations without 'is_correct' key are treated as incorrect."""
        questions = _make_questions(2)
        evals: dict[str, dict[str, object]] = {
            "1": {"score": 10.0},
            "2": {"score": 10.0, "is_correct": True},
        }
        summary = compute_score_summary(questions, evals)
        assert summary["correct"] == 1
