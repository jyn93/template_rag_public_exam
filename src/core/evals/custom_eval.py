"""LLM-as-judge evaluator for domain-specific oposiciones quality dimensions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import structlog

from src.core.evals.base import EvalResult, EvalSample, Evaluator
from src.core.exceptions import GenerationError

if TYPE_CHECKING:
    from src.infrastructure.llm.base import LLMClient

__all__ = ["OposicionesEvaluator"]

logger = structlog.get_logger(__name__)

_JUDGE_PROMPT = """\
You are a panel of Spanish public-exam (oposiciones) specialists assessing the
quality of a study-assistant RAG system.

QUESTION: {question}
SYSTEM ANSWER: {answer}
REFERENCE ANSWER: {ground_truth}

Rate each dimension on a scale of 1 (poor) to 5 (excellent):

1. legal_precision    — Are legal references, article numbers, and norms correct?
2. completeness       — Does the answer cover all aspects required by the question?
3. clarity            — Is the answer comprehensible for an exam candidate?
4. factual_accuracy   — Are dates, figures, and procedural steps accurate?

Respond ONLY with a valid JSON object — no prose, no markdown fences:
{{"legal_precision": N, "completeness": N, "clarity": N, \
"factual_accuracy": N, "justification": "..."}}
"""

_EMPTY_JUDGE: dict[str, object] = {
    "legal_precision": 0,
    "completeness": 0,
    "clarity": 0,
    "factual_accuracy": 0,
    "justification": "Could not parse LLM judge response.",
}

_JUDGE_DIMENSIONS: tuple[str, ...] = (
    "legal_precision",
    "completeness",
    "clarity",
    "factual_accuracy",
)


class OposicionesEvaluator(Evaluator):
    """LLM-as-judge evaluator for domain-specific oposiciones quality.

    Measures four dimensions beyond standard RAGAS metrics:

    - **legal_precision** — Correctness of legal references and article citations.
    - **completeness** — Coverage of all exam-relevant aspects.
    - **clarity** — Pedagogical clarity for an exam candidate.
    - **factual_accuracy** — Correctness of dates, figures, and procedures.

    Each dimension is scored 1–5 by an LLM judge.  The :meth:`evaluate`
    method returns the mean score per dimension across all samples.

    Args:
        llm_client: Any :class:`~src.infrastructure.llm.base.LLMClient`
            implementation used as the judge model.

    Example:
        >>> evaluator = OposicionesEvaluator(llm_client=client)
        >>> samples = [
        ...     EvalSample(
        ...         question="What is habeas corpus?",
        ...         answer="A writ requiring a person...",
        ...         contexts=["Legal remedy..."],
        ...         ground_truth="A fundamental right protecting liberty.",
        ...     )
        ... ]
        >>> result = await evaluator.evaluate(samples)
        >>> print(result.summary["clarity_mean"])
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialise the evaluator with a judge LLM.

        Args:
            llm_client: LLM adapter to use as the evaluation judge.
        """
        self._llm = llm_client

    async def evaluate(self, samples: list[EvalSample]) -> EvalResult:
        """Judge each sample and return aggregated dimension scores.

        Args:
            samples: List of :class:`~src.core.evals.base.EvalSample` to judge.

        Returns:
            :class:`~src.core.evals.base.EvalResult` where ``scores`` holds
            per-sample dicts and ``summary`` holds mean scores per dimension.

        Raises:
            ValueError: If ``samples`` is empty.
        """
        if not samples:
            raise ValueError("samples must not be empty.")

        logger.info("oposiciones_eval_start", sample_count=len(samples))

        per_sample: list[dict[str, object]] = []
        for sample in samples:
            judgment = await self._judge_sample(sample)
            per_sample.append(judgment)

        summary = self._summarise(per_sample)
        logger.info("oposiciones_eval_complete", summary=summary)

        scores: dict[str, object] = {"per_sample": per_sample}
        return EvalResult(scores=scores, summary=summary)

    async def _judge_sample(self, sample: EvalSample) -> dict[str, object]:
        """Call the LLM judge on a single sample.

        Args:
            sample: The evaluation sample to judge.

        Returns:
            Dict with dimension scores and justification.  Falls back to
            :data:`_EMPTY_JUDGE` on LLM or parse failure.
        """
        prompt = _JUDGE_PROMPT.format(
            question=sample.question,
            answer=sample.answer,
            ground_truth=sample.ground_truth,
        )
        try:
            raw = await self._llm.complete(prompt)
        except GenerationError:
            logger.warning(
                "oposiciones_judge_llm_failed", question=sample.question[:60]
            )
            return dict(_EMPTY_JUDGE)
        except Exception as exc:
            logger.warning(
                "oposiciones_judge_unexpected_error",
                question=sample.question[:60],
                error=str(exc),
            )
            return dict(_EMPTY_JUDGE)

        return self._parse_judgment(raw)

    def _parse_judgment(self, raw: str) -> dict[str, object]:
        """Parse the LLM judge response into a judgment dict.

        Args:
            raw: Raw string from the LLM judge.

        Returns:
            Parsed judgment dict, or a copy of :data:`_EMPTY_JUDGE` on failure.
        """
        # 1. Direct JSON parse
        try:
            return cast(dict[str, object], json.loads(raw))
        except json.JSONDecodeError:
            pass

        # 2. Incremental scan
        decoder = json.JSONDecoder()
        for i, ch in enumerate(raw):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(raw, i)
                return cast(dict[str, object], obj)
            except json.JSONDecodeError:
                continue

        logger.warning("oposiciones_judge_parse_failed", raw_preview=raw[:100])
        return dict(_EMPTY_JUDGE)

    @staticmethod
    def _summarise(per_sample: list[dict[str, object]]) -> dict[str, float]:
        """Compute per-dimension means across all judged samples.

        Args:
            per_sample: List of judgment dicts (one per sample).

        Returns:
            Dict mapping ``"<dimension>_mean"`` to the mean score.
            Dimensions with no numeric values default to 0.0.
        """
        totals: dict[str, float] = dict.fromkeys(_JUDGE_DIMENSIONS, 0.0)
        counts: dict[str, int] = dict.fromkeys(_JUDGE_DIMENSIONS, 0)

        for judgment in per_sample:
            for dim in _JUDGE_DIMENSIONS:
                value = judgment.get(dim)
                if isinstance(value, (int, float)):
                    totals[dim] += float(value)
                    counts[dim] += 1

        return {
            f"{dim}_mean": totals[dim] / counts[dim] if counts[dim] else 0.0
            for dim in _JUDGE_DIMENSIONS
        }
