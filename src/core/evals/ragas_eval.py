"""RAGAS metrics wrapper for RAG pipeline evaluation."""

from __future__ import annotations

import structlog

from src.core.evals.base import EvalResult, EvalSample, Evaluator

__all__ = ["RAGASEvaluator"]

logger = structlog.get_logger(__name__)


class RAGASEvaluator(Evaluator):
    """Evaluates a RAG pipeline using the RAGAS framework.

    RAGAS measures four key metrics over a question-answer dataset:

    - **faithfulness** — Is the answer supported by the retrieved context?
    - **answer_relevancy** — Does the answer address the question?
    - **context_recall** — Did retrieval fetch all necessary context?
    - **context_precision** — Is the retrieved context relevant?

    Each metric is scored in [0, 1]; higher is better.

    .. note::
        RAGAS requires ``ragas>=0.2.0`` and ``datasets`` installed.  The
        library is imported lazily inside :meth:`evaluate` to avoid import
        errors in environments where it is not available (e.g. CI without
        the optional eval dependencies).

    Example:
        >>> evaluator = RAGASEvaluator()
        >>> samples = [
        ...     EvalSample(
        ...         question="What is an appeal?",
        ...         answer="An appeal is a remedy...",
        ...         contexts=["Appeals are filed within 30 days..."],
        ...         ground_truth="A formal challenge to a decision.",
        ...     )
        ... ]
        >>> result = await evaluator.evaluate(samples)
        >>> print(result.summary["faithfulness_mean"])
    """

    #: Metric names computed by this evaluator.
    METRIC_NAMES: tuple[str, ...] = (
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    )

    async def evaluate(self, samples: list[EvalSample]) -> EvalResult:
        """Run RAGAS evaluation over *samples*.

        Args:
            samples: List of :class:`~src.core.evals.base.EvalSample` with
                question, answer, contexts, and ground_truth.

        Returns:
            :class:`~src.core.evals.base.EvalResult` with raw metric scores
            and mean values in ``summary``.

        Raises:
            ValueError: If ``samples`` is empty.
            ImportError: If ``ragas`` or ``datasets`` are not installed.
        """
        if not samples:
            raise ValueError("samples must not be empty.")

        # Lazy imports to keep startup fast in non-eval environments
        from datasets import Dataset  # type: ignore[import-untyped]
        from ragas import evaluate as ragas_evaluate  # type: ignore[import-untyped]
        from ragas.metrics import (  # type: ignore[import-untyped]
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

        logger.info("ragas_eval_start", sample_count=len(samples))

        dataset = Dataset.from_list(
            [
                {
                    "question": s.question,
                    "answer": s.answer,
                    "contexts": s.contexts,
                    "ground_truth": s.ground_truth,
                }
                for s in samples
            ]
        )

        scores = ragas_evaluate(dataset, metrics=metrics)
        df = scores.to_pandas()

        raw_scores: dict[str, object] = df.to_dict()
        summary: dict[str, float] = {
            f"{name}_mean": float(df[name].mean())
            for name in self.METRIC_NAMES
            if name in df.columns
        }

        logger.info("ragas_eval_complete", summary=summary)
        return EvalResult(scores=raw_scores, summary=summary)
