"""Unit tests for RAGASEvaluator and OposicionesEvaluator."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.evals.base import EvalResult, EvalSample, Evaluator
from src.core.evals.custom_eval import (
    _EMPTY_JUDGE,
    _JUDGE_DIMENSIONS,
    OposicionesEvaluator,
)
from src.core.evals.ragas_eval import RAGASEvaluator
from src.core.exceptions import GenerationError

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_sample(
    question: str = "What is an appeal?",
    answer: str = "An appeal is a formal remedy.",
    contexts: list[str] | None = None,
    ground_truth: str = "A challenge to a legal decision.",
) -> EvalSample:
    return EvalSample(
        question=question,
        answer=answer,
        contexts=contexts if contexts is not None else ["An appeal is filed..."],
        ground_truth=ground_truth,
    )


VALID_JUDGE_JSON = json.dumps(
    {
        "legal_precision": 4,
        "completeness": 3,
        "clarity": 5,
        "factual_accuracy": 4,
        "justification": "Good answer with minor gaps.",
    }
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm() -> AsyncMock:
    """LLM mock returning valid judge JSON."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=VALID_JUDGE_JSON)
    return llm


@pytest.fixture
def oposiciones_evaluator(mock_llm) -> OposicionesEvaluator:
    """OposicionesEvaluator with mocked LLM client."""
    return OposicionesEvaluator(llm_client=mock_llm)


# ── TestEvalSample ────────────────────────────────────────────────────────────


class TestEvalSample:
    """Tests for EvalSample dataclass."""

    def test_fields_stored(self):
        """All fields are stored correctly."""
        s = make_sample()
        assert s.question == "What is an appeal?"
        assert s.answer == "An appeal is a formal remedy."
        assert s.ground_truth == "A challenge to a legal decision."

    def test_contexts_default(self):
        """contexts list is stored correctly."""
        s = make_sample(contexts=["ctx1", "ctx2"])
        assert s.contexts == ["ctx1", "ctx2"]


# ── TestEvalResult ────────────────────────────────────────────────────────────


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_scores_stored(self):
        """scores dict is stored correctly."""
        r = EvalResult(scores={"faithfulness": 0.9}, summary={"faithfulness_mean": 0.9})
        assert r.scores["faithfulness"] == pytest.approx(0.9)

    def test_summary_stored(self):
        """summary dict is stored correctly."""
        r = EvalResult(scores={}, summary={"clarity_mean": 4.0})
        assert r.summary["clarity_mean"] == pytest.approx(4.0)

    def test_default_metadata_empty(self):
        """metadata defaults to empty dict."""
        r1 = EvalResult(scores={}, summary={})
        r2 = EvalResult(scores={}, summary={})
        r1.metadata["key"] = "value"
        assert "key" not in r2.metadata


# ── TestRAGASEvaluator ────────────────────────────────────────────────────────


class TestRAGASEvaluator:
    """Tests for RAGASEvaluator."""

    async def test_raises_value_error_for_empty_samples(self):
        """ValueError raised when samples list is empty."""
        ev = RAGASEvaluator()
        with pytest.raises(ValueError, match="samples"):
            await ev.evaluate([])

    async def test_metric_names_constant(self):
        """METRIC_NAMES tuple contains expected metric identifiers."""
        assert "faithfulness" in RAGASEvaluator.METRIC_NAMES
        assert "answer_relevancy" in RAGASEvaluator.METRIC_NAMES
        assert "context_recall" in RAGASEvaluator.METRIC_NAMES
        assert "context_precision" in RAGASEvaluator.METRIC_NAMES

    async def test_is_evaluator_subclass(self):
        """RAGASEvaluator is a concrete Evaluator subclass."""
        assert issubclass(RAGASEvaluator, Evaluator)


# ── TestRAGASEvaluatorWithMocks ───────────────────────────────────────────────


class TestRAGASEvaluatorWithMocks:
    """Tests for RAGASEvaluator with fully mocked RAGAS library."""

    @staticmethod
    def _ragas_modules(metric_val: float = 0.8) -> dict:
        """Build a sys.modules patch dict for ragas/datasets lazy imports."""
        import pandas as pd

        df = pd.DataFrame({name: [metric_val] for name in RAGASEvaluator.METRIC_NAMES})
        mock_scores = MagicMock()
        mock_scores.to_pandas.return_value = df

        mock_metric = MagicMock()
        fake_ragas = MagicMock()
        fake_ragas.evaluate = MagicMock(return_value=mock_scores)
        fake_ragas_metrics = MagicMock()
        fake_ragas_metrics.faithfulness = mock_metric
        fake_ragas_metrics.answer_relevancy = mock_metric
        fake_ragas_metrics.context_recall = mock_metric
        fake_ragas_metrics.context_precision = mock_metric

        fake_datasets = MagicMock()
        fake_datasets.Dataset = MagicMock()
        fake_datasets.Dataset.from_list = MagicMock(return_value=MagicMock())

        return {
            "ragas": fake_ragas,
            "ragas.metrics": fake_ragas_metrics,
            "datasets": fake_datasets,
        }

    async def test_evaluate_returns_eval_result(self):
        """evaluate() returns EvalResult with scores and summary."""
        with patch.dict("sys.modules", self._ragas_modules(0.8)):
            ev = RAGASEvaluator()
            result = await ev.evaluate([make_sample()])
            assert isinstance(result, EvalResult)
            assert "faithfulness_mean" in result.summary

    async def test_summary_contains_means(self):
        """Summary contains _mean keys for all RAGAS metric names."""
        with patch.dict("sys.modules", self._ragas_modules(0.75)):
            ev = RAGASEvaluator()
            result = await ev.evaluate([make_sample()])
            for name in RAGASEvaluator.METRIC_NAMES:
                assert f"{name}_mean" in result.summary
                assert result.summary[f"{name}_mean"] == pytest.approx(0.75)


# ── TestOposicionesEvaluatorInit ──────────────────────────────────────────────


class TestOposicionesEvaluatorInit:
    """Tests for OposicionesEvaluator construction."""

    def test_llm_stored(self, mock_llm):
        """LLM client is stored correctly."""
        ev = OposicionesEvaluator(llm_client=mock_llm)
        assert ev._llm is mock_llm

    def test_is_evaluator_subclass(self):
        """OposicionesEvaluator is a concrete Evaluator subclass."""
        assert issubclass(OposicionesEvaluator, Evaluator)


# ── TestOposicionesEvaluatorJudgeSample ──────────────────────────────────────


class TestOposicionesEvaluatorJudgeSample:
    """Tests for OposicionesEvaluator._judge_sample."""

    async def test_returns_judgment_dict(self, oposiciones_evaluator):
        """_judge_sample returns a dict with dimension scores."""
        result = await oposiciones_evaluator._judge_sample(make_sample())
        assert isinstance(result, dict)
        assert "legal_precision" in result

    async def test_returns_empty_judge_on_generation_error(
        self, oposiciones_evaluator, mock_llm
    ):
        """GenerationError from LLM returns _EMPTY_JUDGE copy without raising."""
        mock_llm.complete.side_effect = GenerationError("llm down")
        result = await oposiciones_evaluator._judge_sample(make_sample())
        assert result["legal_precision"] == 0
        assert result["justification"] is not None

    async def test_returns_empty_judge_on_unexpected_error(
        self, oposiciones_evaluator, mock_llm
    ):
        """Unexpected exception from LLM returns _EMPTY_JUDGE copy without raising."""
        mock_llm.complete.side_effect = ConnectionError("timeout")
        result = await oposiciones_evaluator._judge_sample(make_sample())
        assert result == _EMPTY_JUDGE

    async def test_fallback_is_independent_copy(self, oposiciones_evaluator, mock_llm):
        """Empty judge fallback is a copy, not the module-level constant."""
        mock_llm.complete.side_effect = GenerationError("err")
        r1 = await oposiciones_evaluator._judge_sample(make_sample())
        r1["legal_precision"] = 99
        r2 = await oposiciones_evaluator._judge_sample(make_sample())
        assert r2["legal_precision"] == 0


# ── TestParseJudgment ─────────────────────────────────────────────────────────


class TestParseJudgment:
    """Tests for OposicionesEvaluator._parse_judgment."""

    def test_parses_clean_json(self, oposiciones_evaluator):
        """Direct JSON string is parsed correctly."""
        raw = '{"legal_precision": 4, "completeness": 3}'
        result = oposiciones_evaluator._parse_judgment(raw)
        assert result["legal_precision"] == 4

    def test_extracts_json_from_prose(self, oposiciones_evaluator):
        """JSON embedded in prose is extracted via incremental decoder."""
        raw = f"My evaluation:\n{VALID_JUDGE_JSON}\nDone."
        result = oposiciones_evaluator._parse_judgment(raw)
        assert result["clarity"] == 5

    def test_skips_invalid_and_finds_valid(self, oposiciones_evaluator):
        """First valid JSON object is returned when multiple blocks exist."""
        raw = 'Bad: {not valid} Good: {"legal_precision": 3}'
        result = oposiciones_evaluator._parse_judgment(raw)
        assert result["legal_precision"] == 3

    def test_returns_empty_judge_on_failure(self, oposiciones_evaluator):
        """Completely unparseable input returns _EMPTY_JUDGE copy."""
        result = oposiciones_evaluator._parse_judgment("No JSON at all.")
        assert result == _EMPTY_JUDGE


# ── TestSummarise ─────────────────────────────────────────────────────────────


class TestSummarise:
    """Tests for OposicionesEvaluator._summarise."""

    def test_mean_computed_per_dimension(self):
        """Mean is correctly computed for each dimension."""
        per_sample = [
            {
                "legal_precision": 4,
                "completeness": 2,
                "clarity": 5,
                "factual_accuracy": 3,
            },
            {
                "legal_precision": 2,
                "completeness": 4,
                "clarity": 3,
                "factual_accuracy": 5,
            },
        ]
        summary = OposicionesEvaluator._summarise(per_sample)
        assert summary["legal_precision_mean"] == pytest.approx(3.0)
        assert summary["completeness_mean"] == pytest.approx(3.0)
        assert summary["clarity_mean"] == pytest.approx(4.0)
        assert summary["factual_accuracy_mean"] == pytest.approx(4.0)

    def test_all_dimension_means_present(self):
        """Summary contains a mean key for every judge dimension."""
        per_sample = [
            dict.fromkeys(_JUDGE_DIMENSIONS, 3),
        ]
        summary = OposicionesEvaluator._summarise(per_sample)
        for dim in _JUDGE_DIMENSIONS:
            assert f"{dim}_mean" in summary

    def test_missing_dimension_defaults_to_zero(self):
        """Missing dimension in a sample defaults its mean to 0.0."""
        summary = OposicionesEvaluator._summarise([{"legal_precision": 4}])
        assert summary["completeness_mean"] == pytest.approx(0.0)

    def test_non_numeric_dimension_skipped(self):
        """Non-numeric dimension values are ignored in mean calculation."""
        per_sample = [
            {
                "legal_precision": "high",
                "completeness": 4,
                "clarity": 3,
                "factual_accuracy": 5,
            },
        ]
        summary = OposicionesEvaluator._summarise(per_sample)
        assert summary["legal_precision_mean"] == pytest.approx(0.0)
        assert summary["completeness_mean"] == pytest.approx(4.0)

    def test_empty_list_returns_zero_means(self):
        """Empty per_sample list returns 0.0 for all dimension means."""
        summary = OposicionesEvaluator._summarise([])
        for dim in _JUDGE_DIMENSIONS:
            assert summary[f"{dim}_mean"] == pytest.approx(0.0)


# ── TestOposicionesEvaluatorEvaluate ─────────────────────────────────────────


class TestOposicionesEvaluatorEvaluate:
    """End-to-end tests for OposicionesEvaluator.evaluate."""

    async def test_raises_value_error_for_empty_samples(self, oposiciones_evaluator):
        """ValueError raised when samples list is empty."""
        with pytest.raises(ValueError, match="samples"):
            await oposiciones_evaluator.evaluate([])

    async def test_returns_eval_result(self, oposiciones_evaluator):
        """evaluate() returns an EvalResult instance."""
        result = await oposiciones_evaluator.evaluate([make_sample()])
        assert isinstance(result, EvalResult)

    async def test_summary_contains_dimension_means(self, oposiciones_evaluator):
        """Summary dict contains mean keys for all judge dimensions."""
        result = await oposiciones_evaluator.evaluate([make_sample()])
        for dim in _JUDGE_DIMENSIONS:
            assert f"{dim}_mean" in result.summary

    async def test_llm_called_once_per_sample(self, oposiciones_evaluator, mock_llm):
        """LLM is called once per sample."""
        samples = [make_sample(), make_sample(question="Second?")]
        await oposiciones_evaluator.evaluate(samples)
        assert mock_llm.complete.call_count == 2

    async def test_handles_all_llm_failures_gracefully(
        self, oposiciones_evaluator, mock_llm
    ):
        """All LLM failures return zero means without raising."""
        mock_llm.complete.side_effect = GenerationError("down")
        result = await oposiciones_evaluator.evaluate([make_sample()])
        for dim in _JUDGE_DIMENSIONS:
            assert result.summary[f"{dim}_mean"] == pytest.approx(0.0)
