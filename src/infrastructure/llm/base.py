"""Abstract base class for LLM clients."""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["LLMClient"]


class LLMClient(ABC):
    """Abstract adapter for any LLM provider.

    Concrete subclasses wrap provider-specific SDKs (LiteLLM, OpenAI,
    Anthropic…) and expose a uniform async interface used by all
    generators in the application.
    """

    @abstractmethod
    async def complete(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Send a completion request and return the response text.

        Args:
            user_prompt: The human turn of the conversation.
            system_prompt: Optional system instruction prepended to the
                conversation. When *None* the provider default is used.

        Returns:
            The model's response as a plain string.

        Raises:
            GenerationError: If the provider returns an error or the
                response cannot be decoded.
        """
