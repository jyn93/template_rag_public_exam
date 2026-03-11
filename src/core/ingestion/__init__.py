"""Document ingestion package — Strategy pattern for document loaders."""

from src.core.ingestion.base import Document, DocumentLoader
from src.core.ingestion.txt_loader import TxtDocumentLoader

__all__ = [
    "Document",
    "DocumentLoader",
    "TxtDocumentLoader",
]
