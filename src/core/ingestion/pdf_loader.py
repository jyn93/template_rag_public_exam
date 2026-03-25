"""PDF document loader implementing the DocumentLoader strategy."""

from __future__ import annotations

import uuid
from pathlib import Path

import structlog
from llama_index.readers.file import PDFReader

from src.core.exceptions import IngestionError
from src.core.ingestion.base import Document, DocumentLoader

__all__ = ["PDFDocumentLoader"]

logger = structlog.get_logger(__name__)


class PDFDocumentLoader(DocumentLoader):
    """Loader for PDF files using llama-index PDFReader.

    Produces one :class:`Document` per page, preserving page-level
    metadata.  Chunking is intentionally *not* performed here — that
    responsibility belongs to a downstream pipeline stage.

    Example:
        >>> loader = PDFDocumentLoader()
        >>> loader.supports(Path("temario.pdf"))
        True
        >>> loader.supports(Path("temario.txt"))
        False
        >>> docs = loader.load(Path("temario.pdf"), subject="Constitutional Law")
        >>> len(docs)  # one Document per page
        5
    """

    def supports(self, path: Path) -> bool:
        """Return True for files with a ``.pdf`` extension.

        Args:
            path: Path whose extension will be checked.

        Returns:
            True if the extension is ``.pdf`` (case-insensitive).
        """
        return path.suffix.lower() == ".pdf"

    def load(self, path: Path, subject: str = "") -> list[Document]:
        """Read a PDF file and return one Document per page.

        Args:
            path: Path to the ``.pdf`` file to load.
            subject: Subject / topic label stored in the document metadata.

        Returns:
            A list of :class:`Document` objects, one per page extracted
            from the PDF.

        Raises:
            ValueError: If *path* does not point to an existing file.
            IngestionError: If the PDF cannot be parsed.
        """
        if not path.exists():
            raise ValueError(f"File not found: {path}")

        logger.info("loading_pdf_document", file=path.name, subject=subject)

        try:
            reader = PDFReader()
            llama_docs = reader.load_data(file=path)
        except Exception as exc:
            logger.exception(
                "pdf_parse_failed", file=path.name, error=str(exc)
            )
            raise IngestionError(f"Cannot parse PDF '{path.name}': {exc}") from exc

        documents: list[Document] = []
        for i, llama_doc in enumerate(llama_docs):
            page_label = llama_doc.metadata.get("page_label", str(i))
            doc_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{path.resolve()}#page={i}",
                )
            )
            documents.append(
                Document(
                    content=llama_doc.text,
                    metadata={
                        "source": path.name,
                        "page": page_label,
                        "file_type": "pdf",
                        "subject": subject,
                    },
                    doc_id=doc_id,
                )
            )

        total_chars = sum(len(d.content) for d in documents)
        logger.info(
            "pdf_document_loaded",
            file=path.name,
            pages=len(documents),
            total_chars=total_chars,
            subject=subject,
        )
        return documents
