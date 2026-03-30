"""AnswerEvaluator — LLM-based assessment of student exam answers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import structlog
from langfuse.decorators import langfuse_context, observe

from src.core.config.prompts import ANSWER_EVALUATION_PROMPT
from src.core.exceptions import GenerationError

if TYPE_CHECKING:
    from src.infrastructure.llm.base import LLMClient

__all__ = ["AnswerEvaluator", "EvaluationResult"]

logger = structlog.get_logger(__name__)

_EMPTY_EVALUATION: dict[str, object] = {
    "score": 0,
    "is_correct": False,
    "feedback": "Could not evaluate the answer — please try again.",
    "missing_points": [],
    "strengths": [],
}


@dataclass
class EvaluationResult:
    """Typed wrapper around the LLM evaluation dict.

    Attributes:
        score: Numeric score in the range 0–10.
        is_correct: Whether the answer is considered correct.
        feedback: Detailed explanatory feedback for the student.
        missing_points: Key points missing from the student answer.
        strengths: Points the student answered well.
        raw: Original parsed dict from the LLM, for full access.
    """

    score: int | float
    is_correct: bool
    feedback: str
    missing_points: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> EvaluationResult:
        """Build an EvaluationResult from a parsed LLM response dict.

        Args:
            data: Parsed dict with ``score``, ``is_correct``, ``feedback``,
                ``missing_points``, and ``strengths`` keys.

        Returns:
            Populated :class:`EvaluationResult` instance.
        """
        return cls(
            score=cast(int | float, data.get("score", 0)),
            is_correct=bool(data.get("is_correct", False)),
            feedback=str(data.get("feedback", "")),
            missing_points=cast(list[str], data.get("missing_points", [])),
            strengths=cast(list[str], data.get("strengths", [])),
            raw=data,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict suitable for JSON API responses.

        Returns:
            Dict with all evaluation fields.
        """
        return {
            "score": self.score,
            "is_correct": self.is_correct,
            "feedback": self.feedback,
            "missing_points": self.missing_points,
            "strengths": self.strengths,
        }


class AnswerEvaluator:
    """Evaluates student exam answers using an LLM judge.

    The evaluator formats the question, correct answer, and student answer
    into :data:`~src.core.config.prompts.ANSWER_EVALUATION_PROMPT`, calls the
    LLM, and parses the structured JSON feedback.

    Args:
        llm_client: Any :class:`~src.infrastructure.llm.base.LLMClient`
            implementation.

    Example:
        >>> evaluator = AnswerEvaluator(llm_client=client)
        >>> result = await evaluator.evaluate(
        ...     question="What is habeas corpus?",
        ...     correct_answer="A writ requiring a person to be brought...",
        ...     student_answer="It is a legal right protecting freedom.",
        ... )
        >>> print(result.score, result.feedback)
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialise the evaluator.

        Args:
            llm_client: LLM adapter used for evaluation completions.
        """
        self._llm = llm_client

    @observe(name="answer_evaluation")
    async def evaluate(
        self,
        question: str,
        correct_answer: str,
        student_answer: str,
    ) -> EvaluationResult:
        """Evaluate a student's answer against the correct answer.

        Wrapped with a Langfuse ``@observe`` span so the question, student
        answer, and resulting score are captured as a single trace, with the
        underlying LLM generation linked as a child span via the LiteLLM
        Langfuse callback.

        Args:
            question: The exam question text.
            correct_answer: The reference correct answer.
            student_answer: The student's submitted answer.

        Returns:
            :class:`EvaluationResult` with score, feedback, and
            missing / strong points.

        Raises:
            ValueError: If any of the three string arguments is empty.
            GenerationError: If the LLM call fails.

        Example:
            >>> result = await evaluator.evaluate(
            ...     question="Define due process.",
            ...     correct_answer="The legal requirement that the state...",
            ...     student_answer="It means fair treatment under the law.",
            ... )
        """
        self._validate(question, correct_answer, student_answer)

        langfuse_context.update_current_observation(
            input={
                "question": question[:300],
                "student_answer": student_answer[:300],
            },
        )

        prompt = ANSWER_EVALUATION_PROMPT.format(
            question=question,
            correct_answer=correct_answer,
            student_answer=student_answer,
        )

        logger.debug(
            "answer_evaluation_start",
            question_length=len(question),
            student_answer_length=len(student_answer),
        )

        raw = await self._call_llm(prompt)
        result = self._parse_result(raw)

        langfuse_context.update_current_observation(
            output={"score": result.score, "is_correct": result.is_correct},
        )

        logger.info(
            "answer_evaluation_complete",
            score=result.score,
            is_correct=result.is_correct,
        )
        return result

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _validate(
        self, question: str, correct_answer: str, student_answer: str
    ) -> None:
        """Guard against empty inputs.

        Args:
            question: Exam question text.
            correct_answer: Reference answer.
            student_answer: Student's submitted answer.

        Raises:
            ValueError: If any argument is empty or whitespace-only.
        """
        if not question.strip():
            raise ValueError("question must not be empty.")
        if not correct_answer.strip():
            raise ValueError("correct_answer must not be empty.")
        if not student_answer.strip():
            raise ValueError("student_answer must not be empty.")

    async def _call_llm(self, prompt: str) -> str:
        """Send the evaluation prompt to the LLM.

        Args:
            prompt: Formatted evaluation prompt.

        Returns:
            Raw LLM response string (expected JSON).

        Raises:
            GenerationError: On LLM failure.
        """
        try:
            return await self._llm.complete(prompt)
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(f"Answer evaluation LLM call failed: {exc}") from exc

    def _parse_result(self, raw: str) -> EvaluationResult:
        """Parse the LLM response into an :class:`EvaluationResult`.

        Uses incremental JSON decoding to find the first valid JSON object
        in the response, falling back to :data:`_EMPTY_EVALUATION` on
        complete parse failure.

        Args:
            raw: Raw response text from the LLM.

        Returns:
            :class:`EvaluationResult` populated from the parsed data.
        """
        data = self._extract_json(raw)
        return EvaluationResult.from_dict(data)

    @staticmethod
    def _extract_json(raw: str) -> dict[str, object]:
        """Extract the first JSON object from *raw*.

        Attempts direct ``json.loads`` first, then scans incrementally
        for the first well-formed JSON object starting at any ``{``
        position.  Returns :data:`_EMPTY_EVALUATION` as a safe fallback.

        Args:
            raw: Raw string that may contain embedded JSON.

        Returns:
            Parsed dict, or :data:`_EMPTY_EVALUATION` when no valid JSON
            is found.
        """
        # 1. Try direct parse
        try:
            return cast(dict[str, object], json.loads(raw))
        except json.JSONDecodeError:
            pass

        # 2. Incremental scan: find first well-formed JSON object
        decoder = json.JSONDecoder()
        for i, ch in enumerate(raw):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(raw, i)
                return cast(dict[str, object], obj)
            except json.JSONDecodeError:
                continue

        logger.warning("evaluation_json_parse_failed", raw_preview=raw[:100])
        return dict(_EMPTY_EVALUATION)
