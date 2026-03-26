"""RAGGenerator — chat-style answer grounded in retrieved context."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from langfuse import get_client, observe

from src.core.config.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT_TEMPLATE
from src.core.exceptions import GenerationError
from src.core.generation.base import GenerationInput, GenerationOutput, Generator

if TYPE_CHECKING:
    from src.infrastructure.llm.base import LLMClient

__all__ = ["RAGGenerator"]

logger = structlog.get_logger(__name__)

# Maximum characters shown per source chunk in the response (keeps payloads small)
_SOURCE_PREVIEW_CHARS = 200


class RAGGenerator(Generator):
    """Generates grounded chat answers using RAG (Retrieval-Augmented Generation).

    Implements the :class:`~src.core.generation.base.Generator` Template Method
    with a straightforward prompt strategy:

    1. Concatenate retrieved chunks as numbered context blocks.
    2. Wrap with :data:`~src.core.config.prompts.RAG_SYSTEM_PROMPT` and
       :data:`~src.core.config.prompts.RAG_USER_PROMPT_TEMPLATE`.
    3. Call the LLM and return the raw text as the answer.

    Args:
        llm_client: Any :class:`~src.infrastructure.llm.base.LLMClient`
            implementation (e.g. ``LiteLLMClient``).

    Example:
        >>> generator = RAGGenerator(llm_client=client)
        >>> output = await generator.generate(
        ...     GenerationInput(
        ...         query="What is an administrative appeal?",
        ...         context=["An appeal is a remedy...", "Filed within 30 days..."],
        ...     )
        ... )
        >>> print(output.content)
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialise the RAG generator.

        Args:
            llm_client: LLM adapter used for completions.
        """
        self._llm = llm_client

    @observe(name="rag_generation")
    async def generate(self, input: GenerationInput) -> GenerationOutput:
        """Run the RAG generation pipeline with Langfuse tracing.

        Wraps the base :meth:`~src.core.generation.base.Generator.generate`
        template method inside a Langfuse observation so the full request —
        including the nested LLM generation recorded automatically by the
        LiteLLM callback — appears as a single trace in the Langfuse UI.

        Args:
            input: Generation input with query, retrieved context, and metadata.

        Returns:
            :class:`~src.core.generation.base.GenerationOutput` with the
            model's answer and source previews.

        Raises:
            ValueError: If ``input.context`` is empty.
            GenerationError: If the LLM call fails.
        """
        langfuse = get_client()
        langfuse.update_current_span(
            input={"query": input.query, "context_chunks": len(input.context)},
            metadata={"subject": input.metadata.get("subject", ""), **input.metadata},
        )
        output = await super().generate(input)
        answer_preview = (
            output.content[:500]
            if isinstance(output.content, str)
            else str(output.content)[:500]
        )
        langfuse.update_current_span(
            output={"answer": answer_preview},
            metadata={
                "tokens_used": output.tokens_used,
                "sources": len(output.sources),
            },
        )
        return output

    def _build_prompt(self, input: GenerationInput) -> str:
        """Format retrieved chunks and the query into the user prompt.

        Chunks are joined with a separator so the model can visually
        distinguish them.

        Args:
            input: Validated generation input.

        Returns:
            Formatted user prompt string.
        """
        context_str = "\n\n---\n\n".join(
            f"[{i + 1}] {chunk}" for i, chunk in enumerate(input.context)
        )
        return RAG_USER_PROMPT_TEMPLATE.format(
            context=context_str,
            question=input.query,
        )

    async def _call_llm(self, prompt: str) -> str:
        """Delegate to the injected LLM client with the RAG system prompt.

        Args:
            prompt: Assembled user prompt from :meth:`_build_prompt`.

        Returns:
            Raw model response text.

        Raises:
            GenerationError: Propagated from the LLM client on failure.
        """
        logger.debug("rag_generation_start", prompt_length=len(prompt))
        try:
            response = await self._llm.complete(
                prompt,
                system_prompt=RAG_SYSTEM_PROMPT,
            )
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(f"RAG generation failed: {exc}") from exc

        logger.info("rag_generation_complete", response_length=len(response))
        return response

    def _parse_response(self, raw: str, input: GenerationInput) -> GenerationOutput:
        """Wrap the raw LLM response in a :class:`GenerationOutput`.

        Sources are short previews of each context chunk to allow
        attribution without storing the full text.

        Args:
            raw: Raw model response text.
            input: Original generation input for source extraction.

        Returns:
            :class:`GenerationOutput` with the answer and source previews.
        """
        sources = [chunk[:_SOURCE_PREVIEW_CHARS] for chunk in input.context]
        return GenerationOutput(
            content=raw,
            sources=sources,
        )
