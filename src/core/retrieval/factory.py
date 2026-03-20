"""RetrieverFactory — creates the configured Retriever implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.config.settings import RetrieverType, get_settings
from src.core.exceptions import ConfigurationError
from src.core.retrieval.base import Retriever
from src.core.retrieval.dense_retriever import DenseRetriever
from src.core.retrieval.hybrid_retriever import HybridRetriever

if TYPE_CHECKING:
    from src.core.config.settings import Settings
    from src.core.protocols import VectorSearchProtocol

__all__ = ["RetrieverFactory"]


class RetrieverFactory:
    """Factory that instantiates the correct :class:`Retriever` from settings.

    The factory reads :attr:`~src.core.config.settings.Settings.retriever_type`
    and constructs the matching retriever, injecting the shared *vector_store*
    and *top_k* from settings.

    Example:
        >>> from src.core.retrieval.factory import RetrieverFactory
        >>> retriever = RetrieverFactory.create(vector_store=qdrant_store)
        >>> results = await retriever.retrieve("Administrative law")
    """

    @staticmethod
    def create(
        vector_store: VectorSearchProtocol,
        settings: Settings | None = None,
    ) -> Retriever:
        """Build and return the configured :class:`Retriever`.

        Args:
            vector_store: Any object satisfying
                :class:`~src.core.protocols.VectorSearchProtocol`.
                Injected into the chosen retriever.
            settings: Application settings. Defaults to the cached singleton
                from :func:`~src.core.config.settings.get_settings`.

        Returns:
            A concrete :class:`Retriever` instance ready to use.

        Raises:
            ConfigurationError: If
                :attr:`~src.core.config.settings.Settings.retriever_type`
                is not a recognised value.

        Example:
            >>> # Use default settings (reads from environment)
            >>> retriever = RetrieverFactory.create(vector_store=store)
            >>> # Override settings for testing
            >>> retriever = RetrieverFactory.create(store, settings=test_settings)
        """
        resolved = settings or get_settings()

        match resolved.retriever_type:
            case RetrieverType.DENSE:
                return DenseRetriever(
                    vector_store=vector_store,
                    top_k=resolved.top_k,
                )
            case RetrieverType.HYBRID:
                return HybridRetriever(
                    vector_store=vector_store,
                    top_k=resolved.top_k,
                )
            case _:  # pragma: no cover
                raise ConfigurationError(
                    f"Unknown retriever type: '{resolved.retriever_type}'. "
                    f"Valid options: {[t.value for t in RetrieverType]}"
                )
