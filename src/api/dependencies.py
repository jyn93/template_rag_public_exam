"""FastAPI dependency injection container."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client import AsyncQdrantClient

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
    "get_vector_store",
]

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_vector_store() -> QdrantVectorStore:
    """Build a fully wired QdrantVectorStore from application settings.

    Declared as a FastAPI dependency so that any endpoint needing the vector
    store receives the same instance within a single request, avoiding the cost
    of constructing multiple Qdrant clients and embedding models.

    Returns:
        Configured :class:`~src.infrastructure.vector_store.qdrant_store\
            .QdrantVectorStore`.
    """
    settings = get_settings()
    client = AsyncQdrantClient(url=settings.qdrant_url)
    embedding_model = OpenAIEmbedding(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    return QdrantVectorStore(
        client=client,
        embedding_model=embedding_model,
        collection_name=settings.qdrant_collection,
        embedding_dim=settings.embedding_dim,
    )


def get_ingestion_pipeline(
    vector_store: Annotated[QdrantVectorStore, Depends(get_vector_store)],
) -> IngestionPipeline:
    """Build and return a fully wired IngestionPipeline.

    Constructs concrete infrastructure adapters (QdrantVectorStore, MinIOStorage)
    from application settings and wires them into the pipeline facade.

    Args:
        vector_store: Injected vector store dependency.

    Returns:
        Ready-to-use IngestionPipeline instance.
    """
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


def get_retriever(
    vector_store: Annotated[QdrantVectorStore, Depends(get_vector_store)],
) -> Retriever:
    """Build and return the configured Retriever from settings.

    Args:
        vector_store: Injected vector store dependency.

    Returns:
        Concrete :class:`~src.core.retrieval.base.Retriever` instance
        (Dense or Hybrid depending on settings).
    """
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
