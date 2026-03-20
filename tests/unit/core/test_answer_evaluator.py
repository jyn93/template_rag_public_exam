"""Unit tests for AnswerEvaluator and EvaluationResult."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from src.core.exceptions import GenerationError
from src.core.generation.evaluator import (
    _EMPTY_EVALUATION,
    AnswerEvaluator,
    EvaluationResult,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

VALID_EVAL_JSON = json.dumps(
    {
        "score": 8,
        "is_correct": True,
        "feedback": "Good answer covering the main points.",
        "missing_points": ["cite article 24"],
        "strengths": ["correct definition"],
    }
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm() -> AsyncMock:
    """LLM mock returning valid evaluation JSON by default."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=VALID_EVAL_JSON)
    return llm


@pytest.fixture
def evaluator(mock_llm) -> AnswerEvaluator:
    """AnswerEvaluator with mocked LLM client."""
    return AnswerEvaluator(llm_client=mock_llm)


# ── TestEvaluationResult ──────────────────────────────────────────────────────


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass-like wrapper."""

    def test_score_extracted(self):
        """score is read from the data dict."""
        r = EvaluationResult.from_dict(
            {"score": 7, "is_correct": True, "feedback": "ok"}
        )
        assert r.score == 7

    def test_is_correct_extracted(self):
        """is_correct is read from the data dict."""
        r = EvaluationResult.from_dict(
            {"score": 0, "is_correct": False, "feedback": "bad"}
        )
        assert r.is_correct is False

    def test_feedback_extracted(self):
        """feedback is read from the data dict."""
        r = EvaluationResult.from_dict(
            {"score": 5, "is_correct": True, "feedback": "decent"}
        )
        assert r.feedback == "decent"

    def test_missing_points_default_empty(self):
        """missing_points defaults to [] when absent."""
        r = EvaluationResult.from_dict(
            {"score": 0, "is_correct": False, "feedback": ""}
        )
        assert r.missing_points == []

    def test_strengths_default_empty(self):
        """strengths defaults to [] when absent."""
        r = EvaluationResult.from_dict(
            {"score": 0, "is_correct": False, "feedback": ""}
        )
        assert r.strengths == []

    def test_raw_preserved(self):
        """raw attribute holds the original dict."""
        data = {"score": 5, "is_correct": True, "feedback": "ok"}
        r = EvaluationResult.from_dict(data)
        assert r.raw is data

    def test_to_dict_returns_all_fields(self):
        """to_dict() includes score, is_correct, feedback, lists."""
        r = EvaluationResult.from_dict(
            {
                "score": 8,
                "is_correct": True,
                "feedback": "good",
                "missing_points": ["point1"],
                "strengths": ["strength1"],
            }
        )
        d = r.to_dict()
        assert d["score"] == 8
        assert d["is_correct"] is True
        assert d["feedback"] == "good"
        assert d["missing_points"] == ["point1"]
        assert d["strengths"] == ["strength1"]


# ── TestValidate ──────────────────────────────────────────────────────────────


class TestValidate:
    """Tests for AnswerEvaluator._validate."""

    def test_raises_for_empty_question(self, evaluator):
        """ValueError raised when question is empty."""
        with pytest.raises(ValueError, match="question"):
            evaluator._validate("", "correct", "student")

    def test_raises_for_blank_question(self, evaluator):
        """ValueError raised when question is whitespace-only."""
        with pytest.raises(ValueError, match="question"):
            evaluator._validate("   ", "correct", "student")

    def test_raises_for_empty_correct_answer(self, evaluator):
        """ValueError raised when correct_answer is empty."""
        with pytest.raises(ValueError, match="correct_answer"):
            evaluator._validate("question", "", "student")

    def test_raises_for_empty_student_answer(self, evaluator):
        """ValueError raised when student_answer is empty."""
        with pytest.raises(ValueError, match="student_answer"):
            evaluator._validate("question", "correct", "")

    def test_raises_for_blank_student_answer(self, evaluator):
        """ValueError raised when student_answer is whitespace-only."""
        with pytest.raises(ValueError, match="student_answer"):
            evaluator._validate("question", "correct", "   ")

    def test_does_not_raise_for_valid_inputs(self, evaluator):
        """No exception is raised when all inputs are non-empty."""
        evaluator._validate("Q?", "A.", "S.")  # must not raise


# ── TestCallLLM ───────────────────────────────────────────────────────────────


class TestCallLLM:
    """Tests for AnswerEvaluator._call_llm."""

    async def test_returns_llm_response(self, evaluator, mock_llm):
        """_call_llm returns the raw text from the LLM."""
        mock_llm.complete.return_value = "raw response"
        result = await evaluator._call_llm("prompt")
        assert result == "raw response"

    async def test_propagates_generation_error(self, evaluator, mock_llm):
        """GenerationError from the LLM is re-raised as-is."""
        original = GenerationError("llm down")
        mock_llm.complete.side_effect = original
        with pytest.raises(GenerationError) as exc_info:
            await evaluator._call_llm("prompt")
        assert exc_info.value is original

    async def test_wraps_unexpected_exception(self, evaluator, mock_llm):
        """Unexpected exceptions are wrapped in GenerationError."""
        mock_llm.complete.side_effect = ConnectionError("timeout")
        with pytest.raises(GenerationError, match="Answer evaluation LLM call failed"):
            await evaluator._call_llm("prompt")

    async def test_error_chains_cause(self, evaluator, mock_llm):
        """Original exception is chained on the wrapping GenerationError."""
        cause = RuntimeError("boom")
        mock_llm.complete.side_effect = cause
        with pytest.raises(GenerationError) as exc_info:
            await evaluator._call_llm("prompt")
        assert exc_info.value.__cause__ is cause


# ── TestExtractJson ───────────────────────────────────────────────────────────


class TestExtractJson:
    """Tests for AnswerEvaluator._extract_json (static helper)."""

    def test_parses_clean_json(self):
        """Direct JSON string is parsed correctly."""
        raw = '{"score": 9, "is_correct": true}'
        result = AnswerEvaluator._extract_json(raw)
        assert result["score"] == 9
        assert result["is_correct"] is True

    def test_extracts_json_from_prose(self):
        """JSON embedded in prose is extracted via incremental decoder."""
        raw = f"Here is my evaluation:\n{VALID_EVAL_JSON}\nEnd."
        result = AnswerEvaluator._extract_json(raw)
        assert result["score"] == 8

    def test_skips_invalid_brace_block_and_finds_valid_one(self):
        """When multiple {...} blocks exist, the first valid JSON is returned."""
        raw = 'Bad: {not valid} Good: {"score": 5, "is_correct": false}'
        result = AnswerEvaluator._extract_json(raw)
        assert result["score"] == 5

    def test_returns_empty_evaluation_on_failure(self):
        """Completely unparseable input returns the empty evaluation fallback."""
        result = AnswerEvaluator._extract_json("No JSON here.")
        assert result == _EMPTY_EVALUATION

    def test_fallback_is_independent_copy(self):
        """The returned fallback dict is a copy, not the module-level constant."""
        r1 = AnswerEvaluator._extract_json("no json")
        r2 = AnswerEvaluator._extract_json("no json")
        r1["score"] = 99
        assert r2["score"] == 0  # r2 must not be affected


# ── TestEvaluate ──────────────────────────────────────────────────────────────


class TestEvaluate:
    """End-to-end tests for AnswerEvaluator.evaluate."""

    async def test_returns_evaluation_result(self, evaluator):
        """evaluate() returns an EvaluationResult instance."""
        result = await evaluator.evaluate(
            question="What is habeas corpus?",
            correct_answer="A writ requiring a person to be brought before a judge.",
            student_answer="It protects personal liberty.",
        )
        assert isinstance(result, EvaluationResult)

    async def test_score_from_llm_response(self, evaluator):
        """Score matches the value returned by the LLM."""
        result = await evaluator.evaluate(
            question="Q?", correct_answer="A.", student_answer="S."
        )
        assert result.score == 8

    async def test_llm_called_once(self, evaluator, mock_llm):
        """LLM is called exactly once per evaluate() call."""
        await evaluator.evaluate(
            question="Q?", correct_answer="A.", student_answer="S."
        )
        mock_llm.complete.assert_called_once()

    async def test_raises_value_error_for_empty_question(self, evaluator):
        """ValueError raised when question is empty."""
        with pytest.raises(ValueError, match="question"):
            await evaluator.evaluate(
                question="", correct_answer="A.", student_answer="S."
            )

    async def test_raises_value_error_for_empty_student_answer(self, evaluator):
        """ValueError raised when student_answer is empty."""
        with pytest.raises(ValueError, match="student_answer"):
            await evaluator.evaluate(
                question="Q?", correct_answer="A.", student_answer=""
            )

    async def test_handles_malformed_llm_response_gracefully(self, evaluator, mock_llm):
        """Malformed LLM response returns empty evaluation without raising."""
        mock_llm.complete.return_value = "I cannot evaluate this."
        result = await evaluator.evaluate(
            question="Q?", correct_answer="A.", student_answer="S."
        )
        assert result.score == 0
        assert result.is_correct is False

    async def test_to_dict_serialisable(self, evaluator):
        """evaluate() result converts to a plain dict without error."""
        result = await evaluator.evaluate(
            question="Q?", correct_answer="A.", student_answer="S."
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "score" in d
