"""LiteLLM adapter — unified interface for multiple LLM providers."""

from __future__ import annotations

import litellm
import structlog

from src.core.exceptions import GenerationError
from src.infrastructure.llm.base import LLMClient

__all__ = ["LiteLLMClient"]

logger = structlog.get_logger(__name__)


class LiteLLMClient(LLMClient):
    """LLM client backed by LiteLLM to support multiple providers uniformly.

    LiteLLM normalises the API for OpenAI, Anthropic, Ollama, and many
    more providers under a single ``litellm.acompletion`` call.  The
    caller only needs to supply the provider prefix and model name in
    the format expected by LiteLLM (e.g. ``"anthropic/claude-3-5-sonnet-20241022"``
    or ``"gpt-4o"``).

    Args:
        model: LiteLLM model identifier (e.g. ``"gpt-4o-mini"``).
        temperature: Sampling temperature in [0, 2].
        max_tokens: Maximum tokens in the completion.
        timeout: Request timeout in seconds.

    Example:
        >>> client = LiteLLMClient(model="gpt-4o-mini", temperature=0.1)
        >>> answer = await client.complete("What is an appeal procedure?")
        >>> print(answer)
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> None:
        """Initialise the LiteLLM client.

        Args:
            model: LiteLLM model identifier.
            temperature: Sampling temperature.
            max_tokens: Token budget for each completion.
            timeout: HTTP timeout in seconds.
        """
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    async def complete(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Call the configured model and return the completion text.

        Args:
            user_prompt: The human turn of the conversation.
            system_prompt: Optional system instruction. When provided it
                is prepended as a ``system`` role message.

        Returns:
            The model's text response.

        Raises:
            GenerationError: If LiteLLM raises any exception or the
                response structure is unexpected.

        Example:
            >>> text = await client.complete(
            ...     "Explain habeas corpus",
            ...     system_prompt="Answer in formal Spanish.",
            ... )
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        logger.debug(
            "llm_request_start",
            model=self._model,
            messages_count=len(messages),
        )

        try:
            response = await litellm.acompletion(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout=self._timeout,
            )
            text: str = response.choices[0].message.content or ""
        except GenerationError:
            raise
        except Exception as exc:
            logger.error("llm_request_failed", model=self._model, error=str(exc))
            raise GenerationError(
                f"LiteLLM completion failed for model '{self._model}': {exc}"
            ) from exc

        usage = getattr(response, "usage", None)
        logger.info(
            "llm_request_complete",
            model=self._model,
            tokens_used=usage.total_tokens if usage else 0,
        )
        return text
