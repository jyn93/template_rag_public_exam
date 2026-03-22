"""Plain-text document loader implementing the DocumentLoader strategy."""

from __future__ import annotations

import uuid
from pathlib import Path

import structlog

from src.core.exceptions import IngestionError
from src.core.ingestion.base import Document, DocumentLoader

__all__ = ["TxtDocumentLoader"]

logger = structlog.get_logger(__name__)


class TxtDocumentLoader(DocumentLoader):
    """Loader for plain-text (.txt) files.

    Reads the entire file as a single :class:`Document`.  No chunking
    is performed here — that responsibility belongs to a downstream
    pipeline stage.

    Args:
        encoding: Character encoding used when reading the file.
            Defaults to ``"utf-8"``.

    Example:
        >>> loader = TxtDocumentLoader()
        >>> loader.supports(Path("temario.txt"))
        True
        >>> loader.supports(Path("temario.pdf"))
        False
        >>> docs = loader.load(Path("temario.txt"), subject="Constitutional Law")
        >>> len(docs)
        1
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self._encoding = encoding

    def supports(self, path: Path) -> bool:
        """Return True for files with a ``.txt`` extension.

        Args:
            path: Path whose extension will be checked.

        Returns:
            True if the extension is ``.txt`` (case-insensitive).
        """
        return path.suffix.lower() == ".txt"

    def load(self, path: Path, subject: str = "") -> list[Document]:
        """Read a plain-text file and return it as a single Document.

        Args:
            path: Path to the ``.txt`` file to load.
            subject: Subject / topic label stored in the document metadata.

        Returns:
            A list containing exactly one :class:`Document` with the
            full file content.

        Raises:
            ValueError: If *path* does not point to an existing file.
            IngestionError: If the file cannot be decoded with the
                configured encoding.
        """
        if not path.exists():
            raise ValueError(f"File not found: {path}")

        logger.info("loading_txt_document", file=path.name, subject=subject)

        try:
            content = path.read_text(encoding=self._encoding)
        except UnicodeDecodeError as exc:
            logger.exception(
                "txt_decode_failed",
                file=path.name,
                encoding=self._encoding,
                error=str(exc),
            )
            raise IngestionError(
                f"Cannot decode '{path.name}' with encoding '{self._encoding}'"
            ) from exc

        doc = Document(
            content=content,
            metadata={
                "source": path.name,
                "file_type": "txt",
                "subject": subject,
            },
            doc_id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve()))),
        )

        logger.info(
            "txt_document_loaded",
            file=path.name,
            doc_id=doc.doc_id,
            chars=len(content),
        )
        return [doc]
