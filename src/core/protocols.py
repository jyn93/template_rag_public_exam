"""Shared structural protocols for cross-layer dependency injection.

These protocols decouple the core layer from infrastructure implementations.
Any class that satisfies the required method signatures is compatible,
enabling easy substitution and unit testing with mocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

__all__ = ["DocStorageProtocol", "VectorIndexProtocol", "VectorSearchProtocol"]

if TYPE_CHECKING:
    from src.core.ingestion.base import Document
    from src.core.retrieval.base import RetrievalResult


class VectorIndexProtocol(Protocol):
    """Write interface for vector stores consumed by IngestionPipeline.

    Any object that implements :meth:`add_documents` satisfies this protocol.
    """

    async def add_documents(
        self, documents: list[Document]
    ) -> dict[str, object]:
        """Embed and persist a list of documents in the vector store.

        Args:
            documents: Chunked documents to index.

        Returns:
            Arbitrary result dict (implementation-defined).
        """
        ...


class VectorSearchProtocol(Protocol):
    """Read interface for vector stores consumed by retrievers.

    Any object that implements :meth:`search` satisfies this protocol.
    """

    async def search(
        self, query: str, top_k: int
    ) -> list[RetrievalResult]:
        """Search the vector store for chunks semantically similar to *query*.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return.

        Returns:
            List of :class:`~src.core.retrieval.base.RetrievalResult` sorted
            by descending relevance score.
        """
        ...


class DocStorageProtocol(Protocol):
    """Interface for object storage consumed by IngestionPipeline.

    Any object that implements :meth:`upload` satisfies this protocol.
    """

    async def upload(self, path: Path, subject: str) -> dict[str, object]:
        """Upload a raw document file to object storage.

        Args:
            path: Local path to the source file.
            subject: Subject / topic name used to organise the stored file.

        Returns:
            Arbitrary result dict (implementation-defined).
        """
        ...
