"""ExamGenerator — structured exam generation from retrieved context."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import structlog
from langfuse.decorators import langfuse_context, observe

from src.core.config.prompts import EXAM_GENERATION_PROMPT
from src.core.exceptions import GenerationError
from src.core.generation.base import GenerationInput, GenerationOutput, Generator

if TYPE_CHECKING:
    from src.infrastructure.llm.base import LLMClient

__all__ = ["ExamGenerator"]

logger = structlog.get_logger(__name__)

_DEFAULT_NUM_QUESTIONS = 10
_DEFAULT_EXAM_TYPE = "test"
_DEFAULT_DIFFICULTY = "media"
# Maximum source chunks echoed back in the response (avoids large payloads)
_MAX_SOURCES = 3


class ExamGenerator(Generator):
    """Generates structured exams (test or open-ended) from retrieved context.

    Implements the :class:`~src.core.generation.base.Generator` Template Method
    to produce exams in a strict JSON format suitable for the frontend.

    Strategy:

    1. Format the retrieved chunks and exam parameters into
       :data:`~src.core.config.prompts.EXAM_GENERATION_PROMPT`.
    2. Call the LLM requesting JSON output.
    3. Parse the response, with a regex fallback for models that wrap
       JSON in prose (e.g. ``Here is the exam: ```json{...}```).

    Args:
        llm_client: Any :class:`~src.infrastructure.llm.base.LLMClient`
            implementation.
        num_questions: Number of questions to generate.
        exam_type: One of ``"test"``, ``"desarrollo"``, or ``"mixto"``.
        difficulty: One of ``"facil"``, ``"media"``, or ``"dificil"``.

    Example:
        >>> generator = ExamGenerator(llm_client=client, num_questions=5)
        >>> output = await generator.generate(
        ...     GenerationInput(
        ...         query="Administrative appeal",
        ...         context=["An appeal is filed within 30 days..."],
        ...         metadata={"subject": "Administrative Law"},
        ...     )
        ... )
        >>> questions = output.content["questions"]
    """

    def __init__(
        self,
        llm_client: LLMClient,
        num_questions: int = _DEFAULT_NUM_QUESTIONS,
        exam_type: str = _DEFAULT_EXAM_TYPE,
        difficulty: str = _DEFAULT_DIFFICULTY,
    ) -> None:
        """Initialise the exam generator.

        Args:
            llm_client: LLM adapter used for completions.
            num_questions: How many questions to generate.
            exam_type: Exam format (``"test"`` | ``"desarrollo"`` | ``"mixto"``).
            difficulty: Target difficulty level.
        """
        self._llm = llm_client
        self._num_questions = num_questions
        self._exam_type = exam_type
        self._difficulty = difficulty

    @observe(name="exam_generation")
    async def generate(self, input: GenerationInput) -> GenerationOutput:
        """Run the exam generation pipeline with Langfuse tracing.

        Wraps the base :meth:`~src.core.generation.base.Generator.generate`
        template method inside a Langfuse observation so the exam parameters
        and the nested LLM generation appear as a single structured trace.

        Args:
            input: Generation input with topic query, context chunks, and
                optional metadata (e.g. ``{"subject": "Administrative Law"}``).

        Returns:
            :class:`~src.core.generation.base.GenerationOutput` with the
            structured exam dict and source chunks.

        Raises:
            ValueError: If ``input.context`` is empty.
            GenerationError: If the LLM call fails.
        """
        langfuse_context.update_current_observation(
            input={"query": input.query, "context_chunks": len(input.context)},
            metadata={
                "exam_type": self._exam_type,
                "difficulty": self._difficulty,
                "num_questions": self._num_questions,
                **input.metadata,
            },
        )
        output = await super().generate(input)
        questions = (
            cast(list[object], output.content.get("questions", []))
            if isinstance(output.content, dict)
            else []
        )
        langfuse_context.update_current_observation(
            output={"questions_generated": len(questions)},
            metadata={"tokens_used": output.tokens_used},
        )
        return output

    def _build_prompt(self, input: GenerationInput) -> str:
        """Format the exam generation prompt with context and parameters.

        Args:
            input: Validated generation input.

        Returns:
            Formatted prompt string requesting JSON exam output.
        """
        context_str = "\n\n---\n\n".join(input.context)
        return EXAM_GENERATION_PROMPT.format(
            num_questions=self._num_questions,
            exam_type=self._exam_type,
            difficulty=self._difficulty,
            context=context_str,
        )

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM and request a JSON-formatted exam.

        Args:
            prompt: Assembled exam generation prompt.

        Returns:
            Raw LLM response string (expected to be JSON).

        Raises:
            GenerationError: If the LLM call fails.
        """
        logger.debug(
            "exam_generation_start",
            num_questions=self._num_questions,
            exam_type=self._exam_type,
            difficulty=self._difficulty,
        )
        try:
            response = await self._llm.complete(prompt)
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(f"Exam generation LLM call failed: {exc}") from exc

        logger.info(
            "exam_generation_llm_complete",
            response_length=len(response),
        )
        return response

    def _parse_response(self, raw: str, input: GenerationInput) -> GenerationOutput:
        """Parse the LLM JSON response into a structured :class:`GenerationOutput`.

        Attempts direct JSON parsing first.  Falls back to a regex search for
        a JSON object embedded in prose (e.g. when the model wraps the output
        in a code block or explanatory text).  If neither strategy succeeds,
        returns a safe empty structure ``{"questions": []}``.

        Args:
            raw: Raw response text from the LLM.
            input: Original generation input for source extraction.

        Returns:
            :class:`GenerationOutput` with a ``dict`` content holding the
            ``"questions"`` list.
        """
        data = self._extract_json(raw)
        sources = input.context[:_MAX_SOURCES]
        questions = cast(list[object], data.get("questions", []))
        logger.info(
            "exam_generation_complete",
            questions_parsed=len(questions),
        )
        return GenerationOutput(content=data, sources=sources)

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(raw: str) -> dict[str, object]:
        """Extract the first JSON object from *raw*, with fallback to empty.

        Args:
            raw: Raw string from the LLM that should contain JSON.

        Returns:
            Parsed dict, or ``{"questions": []}`` when no valid JSON is found.
        """
        # 1. Try direct parse
        try:
            return cast(dict[str, object], json.loads(raw))
        except json.JSONDecodeError:
            pass

        # 2. Incremental fallback: scan for the first valid JSON object starting
        #    at each '{' position.  This avoids greedy regex over-matching when
        #    multiple brace groups exist (e.g. "Error: {bad} Valid: {...}").
        decoder = json.JSONDecoder()
        for i, ch in enumerate(raw):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(raw, i)
                return cast(dict[str, object], obj)
            except json.JSONDecodeError:
                continue

        logger.warning("exam_json_parse_failed", raw_preview=raw[:100])
        return {"questions": []}
