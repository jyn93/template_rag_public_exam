"""Document ingestion package — Strategy pattern for document loaders."""

from src.core.ingestion.base import Document, DocumentLoader
from src.core.ingestion.pdf_loader import PDFDocumentLoader
from src.core.ingestion.pipeline import (
    DocStorageProtocol,
    IngestionPipeline,
    VectorStoreProtocol,
)
from src.core.ingestion.txt_loader import TxtDocumentLoader

__all__ = [
    "DocStorageProtocol",
    "Document",
    "DocumentLoader",
    "IngestionPipeline",
    "PDFDocumentLoader",
    "TxtDocumentLoader",
    "VectorStoreProtocol",
]
