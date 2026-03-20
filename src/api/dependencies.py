"""FastAPI dependency injection container."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.config.settings import Settings, get_settings
from src.core.ingestion.pdf_loader import PDFDocumentLoader
from src.core.ingestion.pipeline import IngestionPipeline
from src.core.ingestion.txt_loader import TxtDocumentLoader
from src.infrastructure.storage.minio_storage import MinIOStorage
from src.infrastructure.vector_store.qdrant_store import QdrantVectorStore

__all__ = ["IngestionPipelineDep", "SettingsDep"]

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_ingestion_pipeline() -> IngestionPipeline:
    """Build and return a fully wired IngestionPipeline.

    Constructs concrete infrastructure adapters (QdrantVectorStore, MinIOStorage)
    from application settings and wires them into the pipeline facade.

    Returns:
        Ready-to-use :class:`~src.core.ingestion.pipeline.IngestionPipeline` instance.
    """
    vector_store = QdrantVectorStore()
    doc_storage = MinIOStorage()
    loaders = [PDFDocumentLoader(), TxtDocumentLoader()]
    return IngestionPipeline(
        loaders=loaders,
        vector_store=vector_store,
        doc_storage=doc_storage,
    )


IngestionPipelineDep = Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)]
