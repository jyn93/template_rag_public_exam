"""Document ingestion package — Strategy pattern for document loaders."""

from src.core.ingestion.base import Document, DocumentLoader
from src.core.ingestion.pdf_loader import PDFDocumentLoader
from src.core.ingestion.pipeline import IngestionPipeline
from src.core.ingestion.txt_loader import TxtDocumentLoader
from src.core.protocols import DocStorageProtocol, VectorIndexProtocol

__all__ = [
    "DocStorageProtocol",
    "Document",
    "DocumentLoader",
    "IngestionPipeline",
    "PDFDocumentLoader",
    "TxtDocumentLoader",
    "VectorIndexProtocol",
]
