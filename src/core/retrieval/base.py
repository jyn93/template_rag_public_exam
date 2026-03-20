"""Abstract base classes for the retrieval layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

__all__ = ["RetrievalResult", "Retriever"]

DEFAULT_TOP_K = 5


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its relevance score and metadata.

    Attributes:
        content: Raw text of the retrieved chunk.
        score: Relevance score in the range [0, 1] (higher is more relevant).
        metadata: Arbitrary key-value pairs inherited from the source document
            (e.g. subject, source filename, page number, chunk_index).
        doc_id: Identifier of the original chunk in the vector store.
    """

    content: str
    score: float
    metadata: dict[str, object] = field(default_factory=dict)
    doc_id: str = ""


class Retriever(ABC):
    """Strategy interface for context retrieval.

    Concrete implementations decide how to rank and return chunks
    (e.g. dense vector search, BM25, or hybrid combination).

    Example:
        >>> retriever = DenseRetriever(vector_store=qdrant_store)
        >>> results = await retriever.retrieve("What is the appeal procedure?")
        >>> print(results[0].content)
    """

    @abstractmethod
    async def retrieve(
        self, query: str, top_k: int = DEFAULT_TOP_K
    ) -> list[RetrievalResult]:
        """Retrieve the most relevant chunks for a given query.

        Args:
            query: User question or search string in natural language.
            top_k: Maximum number of results to return.

        Returns:
            List of :class:`RetrievalResult` sorted by descending relevance.

        Raises:
            RetrievalError: If the underlying store is unavailable or fails.
            ValueError: If *query* is empty or blank.
        """
