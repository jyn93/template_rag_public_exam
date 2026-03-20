"""Abstract base classes and data models for the generation layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

__all__ = ["GenerationInput", "GenerationOutput", "Generator"]


@dataclass
class GenerationInput:
    """Input contract for all generators.

    Attributes:
        query: The user's question or generation directive.
        context: Ordered list of retrieved text chunks used as grounding
            context for the LLM.
        metadata: Arbitrary key-value pairs forwarded to the generator
            (e.g. ``{"subject": "Administrative Law"}``).
    """

    query: str
    context: list[str]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class GenerationOutput:
    """Output contract returned by all generators.

    Attributes:
        content: The generated content.  Plain ``str`` for chat responses;
            ``dict`` for structured outputs like exam JSON.
        sources: Excerpt or identifier of each source chunk used.
        tokens_used: Total tokens consumed by the LLM call.  May be 0
            when the underlying client does not expose usage stats.
    """

    content: str | dict[str, object]
    sources: list[str]
    tokens_used: int = 0


class Generator(ABC):
    """Template Method base for all generators.

    The fixed algorithm is:
    1. :meth:`_validate` — guard against bad inputs.
    2. :meth:`_build_prompt` — assemble the LLM prompt from the input.
    3. :meth:`_call_llm` — send the prompt to the model.
    4. :meth:`_parse_response` — convert the raw string into a structured
       :class:`GenerationOutput`.

    Concrete subclasses implement the abstract steps while inheriting
    the common orchestration logic from :meth:`generate`.

    Example:
        >>> generator = RAGGenerator(llm_client=client, retriever=retriever)
        >>> output = await generator.generate(
        ...     GenerationInput(query="What is habeas corpus?", context=chunks)
        ... )
        >>> print(output.content)
    """

    async def generate(self, input: GenerationInput) -> GenerationOutput:
        """Run the full generation pipeline.

        Args:
            input: Generation input with query, context, and metadata.

        Returns:
            A :class:`GenerationOutput` with the model's answer.

        Raises:
            ValueError: If ``input.context`` is empty.
            GenerationError: If the LLM call fails.
        """
        self._validate(input)
        prompt = self._build_prompt(input)
        raw = await self._call_llm(prompt)
        return self._parse_response(raw, input)

    def _validate(self, input: GenerationInput) -> None:
        """Validate the generation input.

        Args:
            input: The input to validate.

        Raises:
            ValueError: If ``context`` is empty.
        """
        if not input.context:
            raise ValueError("GenerationInput.context must not be empty.")

    @abstractmethod
    def _build_prompt(self, input: GenerationInput) -> str:
        """Assemble the user prompt from the generation input.

        Args:
            input: Validated generation input.

        Returns:
            Prompt string to pass to the LLM.
        """

    @abstractmethod
    async def _call_llm(self, prompt: str) -> str:
        """Call the underlying LLM with the assembled prompt.

        Args:
            prompt: The assembled user prompt.

        Returns:
            Raw text response from the model.
        """

    @abstractmethod
    def _parse_response(self, raw: str, input: GenerationInput) -> GenerationOutput:
        """Convert the raw LLM response into a :class:`GenerationOutput`.

        Args:
            raw: Raw text returned by :meth:`_call_llm`.
            input: The original generation input (for source extraction).

        Returns:
            Structured :class:`GenerationOutput`.
        """
