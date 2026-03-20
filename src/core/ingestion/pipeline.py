"""Ingestion pipeline — Facade that orchestrates load, store, chunk, and index."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import structlog
from llama_index.core.node_parser import SentenceSplitter

from src.core.config.settings import get_settings
from src.core.exceptions import IngestionError, StorageError, VectorStoreError
from src.core.ingestion.base import Document, DocumentLoader

__all__ = ["DocStorageProtocol", "IngestionPipeline", "VectorStoreProtocol"]

logger = structlog.get_logger(__name__)


class VectorStoreProtocol(Protocol):
    """Minimal interface for vector stores consumed by IngestionPipeline."""

    async def add_documents(self, documents: list[Document]) -> dict[str, object]:
        """Embed and persist a list of documents in the vector store.

        Args:
            documents: Chunked documents to index.

        Returns:
            Arbitrary result dict (implementation-defined).
        """
        ...


class DocStorageProtocol(Protocol):
    """Minimal interface for object storage consumed by IngestionPipeline."""

    async def upload(self, path: Path, subject: str) -> dict[str, object]:
        """Upload a raw document file to object storage.

        Args:
            path: Local path to the source file.
            subject: Subject / topic name used to organise the stored file.

        Returns:
            Arbitrary result dict (implementation-defined).
        """
        ...


class IngestionPipeline:
    """Facade that orchestrates the full document ingestion flow.

    The pipeline executes the following steps in order:

    1. **Validate** — ensure the source file exists.
    2. **Load** — select and run the appropriate :class:`DocumentLoader`.
    3. **Upload** — persist the original file to object storage *before* indexing,
       so the source is never lost even if the vector store step fails.
    4. **Chunk** — split documents into overlapping text chunks using
       :class:`~llama_index.core.node_parser.SentenceSplitter`.
    5. **Index** — embed chunks and store them in the vector store.

    Example:
        >>> pipeline = IngestionPipeline(
        ...     loaders=[PDFDocumentLoader(), TxtDocumentLoader()],
        ...     vector_store=qdrant_store,
        ...     doc_storage=minio_storage,
        ... )
        >>> result = await pipeline.ingest(
        ...     Path("temario.pdf"), subject_name="Civil Law"
        ... )
        >>> print(result)
        {'total_documents': 12, 'total_chunks': 48, 'subject': 'Civil Law'}
    """

    def __init__(
        self,
        loaders: list[DocumentLoader],
        vector_store: VectorStoreProtocol,
        doc_storage: DocStorageProtocol,
    ) -> None:
        """Initialise the pipeline with its collaborators.

        Args:
            loaders: Ordered list of :class:`DocumentLoader` strategies.
                The first loader whose :meth:`~DocumentLoader.supports` method
                returns ``True`` for a given file is used.
            vector_store: Vector store implementation that satisfies
                :class:`VectorStoreProtocol`.
            doc_storage: Object storage implementation that satisfies
                :class:`DocStorageProtocol`.
        """
        self._loaders = loaders
        self._vector_store = vector_store
        self._doc_storage = doc_storage
        settings = get_settings()
        self._splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def _get_loader(self, path: Path) -> DocumentLoader:
        """Return the first registered loader that supports *path*.

        Args:
            path: Path to the file whose loader should be selected.

        Returns:
            The matching :class:`DocumentLoader` instance.

        Raises:
            ValueError: If no registered loader supports the file extension.
        """
        for loader in self._loaders:
            if loader.supports(path):
                return loader
        raise ValueError(f"No loader registered for '{path.suffix}' files")

    async def ingest(self, path: Path, subject_name: str) -> dict[str, object]:
        """Run the full ingestion flow for a single document.

        Args:
            path: Path to the source document file.
            subject_name: Human-readable subject / topic name
                (e.g. "Administrative Law"). Injected into every document's
                metadata under the ``"subject"`` key.

        Returns:
            Summary dict with keys:
            - ``total_documents``: number of raw documents produced by the loader.
            - ``total_chunks``: number of text chunks sent to the vector store.
            - ``subject``: the *subject_name* argument, echoed for convenience.

        Raises:
            ValueError: If *path* does not exist or no loader supports the format.
            IngestionError: If the loader fails to read or parse the file.
            StorageError: If the original file cannot be uploaded to object storage.
            VectorStoreError: If the chunks cannot be indexed in the vector store.
        """
        if not path.exists():
            raise ValueError(f"File not found: {path}")

        loader = self._get_loader(path)

        try:
            documents = loader.load(path, subject=subject_name)
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(
                f"Unexpected error loading '{path.name}': {exc}"
            ) from exc

        try:
            await self._doc_storage.upload(path, subject_name)
        except Exception as exc:
            raise StorageError(
                f"Failed to upload '{path.name}' to object storage"
            ) from exc

        for doc in documents:
            doc.metadata["subject"] = subject_name

        chunks = self._chunk_documents(documents)

        try:
            await self._vector_store.add_documents(chunks)
        except Exception as exc:
            raise VectorStoreError(f"Failed to index chunks for '{path.name}'") from exc

        logger.info(
            "ingestion_complete",
            file=path.name,
            subject=subject_name,
            total_documents=len(documents),
            total_chunks=len(chunks),
        )

        return {
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "subject": subject_name,
        }

    def _chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split each document into overlapping text chunks.

        Uses the :class:`~llama_index.core.node_parser.SentenceSplitter`
        configured at construction time to respect sentence boundaries.

        Args:
            documents: Raw documents returned by a loader.

        Returns:
            Flat list of chunk :class:`Document` objects.  Each chunk
            inherits its parent's metadata and gains a ``"chunk_index"``
            key.  Chunk ``doc_id`` values follow the pattern
            ``"<parent_doc_id>_chunk_<index>"``.
        """
        chunks: list[Document] = []
        for doc in documents:
            texts = self._splitter.split_text(doc.content)
            for i, text in enumerate(texts):
                chunks.append(
                    Document(
                        content=text,
                        metadata={**doc.metadata, "chunk_index": i},
                        doc_id=f"{doc.doc_id}_chunk_{i}",
                    )
                )
        return chunks
