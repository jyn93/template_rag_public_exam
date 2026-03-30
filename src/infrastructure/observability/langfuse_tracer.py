"""Langfuse observability service — configures LLM tracing via LiteLLM callbacks."""

from __future__ import annotations

import os

import litellm
import structlog

from src.core.config.settings import Settings

__all__ = ["LangfuseService"]

logger = structlog.get_logger(__name__)


class LangfuseService:
    """Configures Langfuse tracing for all LLM interactions in the application.

    Sets the ``LANGFUSE_*`` environment variables required by both the Langfuse
    SDK decorators (``@observe``) and the LiteLLM Langfuse callback, then
    registers that callback so every ``litellm.acompletion`` call is
    automatically recorded as a Langfuse *generation*.

    Usage::

        # Call once at application startup
        LangfuseService.configure(get_settings())

    When Langfuse is disabled (``langfuse_enabled=False``) or the API keys are
    missing the method is a no-op — the application continues normally without
    tracing.

    Example:
        >>> from src.core.config.settings import get_settings
        >>> LangfuseService.configure(get_settings())
    """

    @staticmethod
    def configure(settings: Settings) -> None:
        """Configure Langfuse environment variables and LiteLLM callbacks.

        Reads credentials from *settings* and:

        1. Sets ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``, and
           ``LANGFUSE_HOST`` in the process environment so the Langfuse SDK
           and LiteLLM callback can authenticate automatically.
        2. Appends ``"langfuse"`` to :attr:`litellm.success_callback` and
           :attr:`litellm.failure_callback` (idempotent — will not duplicate
           entries on repeated calls).

        Args:
            settings: Application settings containing Langfuse credentials.
        """
        if not settings.langfuse_enabled:
            logger.info("langfuse_disabled")
            return

        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            logger.warning(
                "langfuse_keys_missing",
                host=settings.langfuse_host,
                hint=(
                    "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable tracing."
                ),
            )
            return

        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
        os.environ["LANGFUSE_HOST"] = settings.langfuse_host

        # Register callbacks idempotently so repeated configure() calls are safe.
        if "langfuse" not in litellm.success_callback:
            litellm.success_callback.append("langfuse")
        if "langfuse" not in litellm.failure_callback:
            litellm.failure_callback.append("langfuse")

        logger.info("langfuse_configured", host=settings.langfuse_host)

    @staticmethod
    def is_configured() -> bool:
        """Return whether Langfuse credentials are present in the environment.

        Returns:
            ``True`` when both ``LANGFUSE_PUBLIC_KEY`` and
            ``LANGFUSE_SECRET_KEY`` are set to non-empty strings.
        """
        return bool(
            os.environ.get("LANGFUSE_PUBLIC_KEY")
            and os.environ.get("LANGFUSE_SECRET_KEY")
        )
