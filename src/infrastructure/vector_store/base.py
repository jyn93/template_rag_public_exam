"""Abstract base class for vector store implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.ingestion.base import Document
    from src.core.retrieval.base import RetrievalResult

__all__ = ["VectorStore"]


class VectorStore(ABC):
    """Abstract base for all vector store backends.

    Concrete implementations (e.g. :class:`~src.infrastructure.vector_store
    .qdrant_store.QdrantVectorStore`) must provide both write (:meth:`add_documents`)
    and read (:meth:`search`) capabilities, satisfying both
    :class:`~src.core.protocols.VectorIndexProtocol` and
    :class:`~src.core.protocols.VectorSearchProtocol` simultaneously.

    Example:
        >>> store = QdrantVectorStore(client=qdrant_client, embedding_model=embed)
        >>> await store.add_documents(chunks)
        {'inserted': 42, 'collection': 'oposiciones_temario'}
        >>> results = await store.search("appeal procedure", top_k=5)
        [RetrievalResult(content='...', score=0.92, ...)]
    """

    @abstractmethod
    async def add_documents(
        self, documents: list[Document]
    ) -> dict[str, object]:
        """Embed and persist documents in the vector store.

        Args:
            documents: Chunked documents to embed and index.

        Returns:
            Arbitrary result dict with at least an ``"inserted"`` key.

        Raises:
            VectorStoreError: If the backend is unavailable or upsert fails.
        """

    @abstractmethod
    async def search(
        self, query: str, top_k: int
    ) -> list[RetrievalResult]:
        """Return the most semantically similar chunks for *query*.

        Args:
            query: Natural language search string.
            top_k: Maximum number of results to return.

        Returns:
            List of :class:`~src.core.retrieval.base.RetrievalResult`
            sorted by descending relevance score.

        Raises:
            VectorStoreError: If the backend is unavailable or search fails.
        """
