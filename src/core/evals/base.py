"""Abstract base classes and data models for the evaluation layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

__all__ = ["EvalResult", "EvalSample", "Evaluator"]


@dataclass
class EvalSample:
    """A single question-answer pair for RAG evaluation.

    Attributes:
        question: The input question posed to the RAG system.
        answer: The RAG system's generated answer.
        contexts: List of retrieved text chunks used to generate the answer.
        ground_truth: Reference correct answer for the question.
    """

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


@dataclass
class EvalResult:
    """Aggregated evaluation results from an evaluator run.

    Attributes:
        scores: Raw per-metric score values (metric name → value or series).
        summary: High-level summary statistics (e.g. mean per metric).
        metadata: Optional additional context stored with the result.
    """

    scores: dict[str, object]
    summary: dict[str, float]
    metadata: dict[str, object] = field(default_factory=dict)


class Evaluator(ABC):
    """Abstract base class for RAG pipeline evaluators.

    Concrete subclasses implement :meth:`evaluate` to assess the quality
    of a RAG system over a dataset of :class:`EvalSample` instances.

    Example:
        >>> evaluator = RAGASEvaluator()
        >>> samples = [
        ...     EvalSample(question="Q?", answer="A.",
        ...                contexts=["ctx"], ground_truth="GT")
        ... ]
        >>> result = await evaluator.evaluate(samples)
        >>> print(result.summary)
    """

    @abstractmethod
    async def evaluate(self, samples: list[EvalSample]) -> EvalResult:
        """Evaluate the RAG system over a list of samples.

        Args:
            samples: List of :class:`EvalSample` instances to evaluate.

        Returns:
            :class:`EvalResult` with per-metric scores and summary statistics.

        Raises:
            ValueError: If ``samples`` is empty.
        """
