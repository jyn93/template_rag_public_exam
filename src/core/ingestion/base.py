"""Abstract base classes for the document loading Strategy pattern."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Document", "DocumentLoader"]


@dataclass
class Document:
    """Represents a processed document ready for indexing.

    Attributes:
        content: Raw text extracted from the source file.
        metadata: Arbitrary key-value pairs describing the document
            (e.g. source filename, page number, subject name).
        doc_id: Unique identifier for this document fragment.
    """

    content: str
    metadata: dict[str, object] = field(default_factory=dict)
    doc_id: str = ""


class DocumentLoader(ABC):
    """Strategy interface for document loaders.

    Each concrete loader handles a specific file format and is
    responsible for reading the file and returning a list of
    :class:`Document` objects with extracted text and metadata.

    Chunking is intentionally *not* part of this interface — that
    concern belongs to a later stage of the ingestion pipeline.

    Example:
        >>> loader = TxtDocumentLoader()
        >>> if loader.supports(Path("temario.txt")):
        ...     docs = loader.load(Path("temario.txt"), subject="Civil Law")
    """

    @abstractmethod
    def load(self, path: Path, subject: str = "") -> list[Document]:
        """Load and pre-process a document file.

        Args:
            path: Absolute or relative path to the source file.
            subject: Human-readable subject / topic name used to
                populate the document metadata (e.g. "Administrative Law").

        Returns:
            Non-empty list of :class:`Document` objects extracted from
            the file.  Each loader determines the granularity (e.g. one
            Document per page, or one per file).

        Raises:
            IngestionError: If the file cannot be read or parsed.
            ValueError: If *path* does not exist.
        """

    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Return True if this loader can handle the given file.

        Args:
            path: Path to the file whose format is being checked.

        Returns:
            True when the loader recognises the file extension /
            format, False otherwise.
        """
