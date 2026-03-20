"""FastAPI dependency injection container."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.config.settings import Settings, get_settings
from src.core.generation.evaluator import AnswerEvaluator
from src.core.generation.rag_generator import RAGGenerator
from src.core.ingestion.pdf_loader import PDFDocumentLoader
from src.core.ingestion.pipeline import IngestionPipeline
from src.core.ingestion.txt_loader import TxtDocumentLoader
from src.core.retrieval.base import Retriever
from src.core.retrieval.factory import RetrieverFactory
from src.infrastructure.llm.base import LLMClient
from src.infrastructure.llm.litellm_client import LiteLLMClient
from src.infrastructure.storage.minio_storage import MinIOStorage
from src.infrastructure.vector_store.qdrant_store import QdrantVectorStore

__all__ = [
    "IngestionPipelineDep",
    "SettingsDep",
]

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_ingestion_pipeline() -> IngestionPipeline:
    """Build and return a fully wired IngestionPipeline.

    Constructs concrete infrastructure adapters (QdrantVectorStore, MinIOStorage)
    from application settings and wires them into the pipeline facade.

    Returns:
        Ready-to-use IngestionPipeline instance.
    """
    vector_store = QdrantVectorStore()
    doc_storage = MinIOStorage()
    loaders = [PDFDocumentLoader(), TxtDocumentLoader()]
    return IngestionPipeline(
        loaders=loaders,
        vector_store=vector_store,
        doc_storage=doc_storage,
    )


def get_llm_client() -> LLMClient:
    """Build and return a LiteLLMClient from application settings.

    Returns:
        Configured :class:`~src.infrastructure.llm.litellm_client.LiteLLMClient`.
    """
    settings = get_settings()
    return LiteLLMClient(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )


def get_retriever() -> Retriever:
    """Build and return the configured Retriever from settings.

    Returns:
        Concrete :class:`~src.core.retrieval.base.Retriever` instance
        (Dense or Hybrid depending on settings).
    """
    vector_store = QdrantVectorStore()
    return RetrieverFactory.create(vector_store=vector_store)


def get_rag_generator(
    llm_client: Annotated[LLMClient, Depends(get_llm_client)],
) -> RAGGenerator:
    """Build and return a RAGGenerator.

    Args:
        llm_client: Injected LLM client dependency.

    Returns:
        Configured :class:`~src.core.generation.rag_generator.RAGGenerator`.
    """
    return RAGGenerator(llm_client=llm_client)


def get_answer_evaluator(
    llm_client: Annotated[LLMClient, Depends(get_llm_client)],
) -> AnswerEvaluator:
    """Build and return an AnswerEvaluator.

    Args:
        llm_client: Injected LLM client dependency.

    Returns:
        Configured :class:`~src.core.generation.evaluator.AnswerEvaluator`.
    """
    return AnswerEvaluator(llm_client=llm_client)


IngestionPipelineDep = Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)]
