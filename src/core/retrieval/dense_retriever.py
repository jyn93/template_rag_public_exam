"""Dense retriever — pure vector similarity search via a vector store."""

from __future__ import annotations

import structlog

from src.core.exceptions import RetrievalError
from src.core.protocols import VectorSearchProtocol
from src.core.retrieval.base import DEFAULT_TOP_K, RetrievalResult, Retriever

__all__ = ["DenseRetriever"]

logger = structlog.get_logger(__name__)


class DenseRetriever(Retriever):
    """Retriever that delegates to a vector store for similarity search.

    The retriever is intentionally thin: it validates the query, calls
    :meth:`VectorSearchProtocol.search`, wraps infrastructure errors, and
    emits a structured log entry on every call.

    Args:
        vector_store: Any object satisfying
            :class:`~src.core.protocols.VectorSearchProtocol`.
        top_k: Default number of results when *top_k* is not passed to
            :meth:`retrieve`.

    Example:
        >>> retriever = DenseRetriever(vector_store=qdrant_store, top_k=10)
        >>> results = await retriever.retrieve("Administrative appeal")
        >>> for r in results:
        ...     print(r.score, r.content[:60])
    """

    def __init__(
        self,
        vector_store: VectorSearchProtocol,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        """Initialise the retriever.

        Args:
            vector_store: Vector store that satisfies
                :class:`~src.core.protocols.VectorSearchProtocol`.
            top_k: Default number of results returned by :meth:`retrieve`.
        """
        self._vector_store = vector_store
        self._top_k = top_k

    async def retrieve(
        self, query: str, top_k: int = DEFAULT_TOP_K
    ) -> list[RetrievalResult]:
        """Return the most semantically similar chunks for *query*.

        Args:
            query: Natural language question or search string.
            top_k: Maximum number of results. Uses the instance default when
                equal to :data:`DEFAULT_TOP_K` and no override is desired;
                callers can pass an explicit value to override.

        Returns:
            List of :class:`RetrievalResult` sorted by descending score.

        Raises:
            ValueError: If *query* is empty or contains only whitespace.
            RetrievalError: If the vector store raises any exception.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty or blank")

        effective_top_k = top_k if top_k != DEFAULT_TOP_K else self._top_k

        logger.debug("dense_retrieval_start", query=query, top_k=effective_top_k)

        try:
            results = await self._vector_store.search(query, effective_top_k)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(
                f"Dense retrieval failed for query '{query}': {exc}"
            ) from exc

        logger.info(
            "dense_retrieval_complete",
            query=query,
            top_k=effective_top_k,
            results_count=len(results),
        )

        return results
