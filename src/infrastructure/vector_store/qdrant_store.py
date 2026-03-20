"""Qdrant vector store implementation."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.core.exceptions import VectorStoreError
from src.core.ingestion.base import Document
from src.core.retrieval.base import RetrievalResult
from src.infrastructure.vector_store.base import VectorStore

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding

__all__ = ["QdrantVectorStore"]

logger = structlog.get_logger(__name__)

_CONTENT_PAYLOAD_KEY = "content"
_DOC_ID_PAYLOAD_KEY = "doc_id"


class QdrantVectorStore(VectorStore):
    """Vector store backed by Qdrant.

    Implements both :class:`~src.core.protocols.VectorIndexProtocol` (write)
    and :class:`~src.core.protocols.VectorSearchProtocol` (read), so a single
    instance can be injected into both :class:`~src.core.ingestion.pipeline
    .IngestionPipeline` and :class:`~src.core.retrieval.dense_retriever
    .DenseRetriever`.

    Args:
        client: Async Qdrant client pointed at the target cluster.
        embedding_model: LlamaIndex embedding model used to vectorise text.
        collection_name: Name of the Qdrant collection to read/write.
        embedding_dim: Dimensionality of the embedding vectors. Must match
            the model's output size.

    Example:
        >>> from qdrant_client import AsyncQdrantClient
        >>> from llama_index.embeddings.openai import OpenAIEmbedding
        >>> client = AsyncQdrantClient(url="http://localhost:6333")
        >>> embed = OpenAIEmbedding(model="text-embedding-3-small")
        >>> store = QdrantVectorStore(client, embed, "oposiciones_temario", 1536)
        >>> await store.add_documents(chunks)
        {'inserted': 10, 'collection': 'oposiciones_temario'}
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        embedding_model: BaseEmbedding,
        collection_name: str,
        embedding_dim: int,
    ) -> None:
        """Initialise the store.

        Args:
            client: Async Qdrant client.
            embedding_model: LlamaIndex :class:`BaseEmbedding` implementation.
            collection_name: Target Qdrant collection name.
            embedding_dim: Vector dimensionality (must match the embedding model).
        """
        self._client = client
        self._embedding_model = embedding_model
        self._collection_name = collection_name
        self._embedding_dim = embedding_dim

    async def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not exist yet.

        Raises:
            VectorStoreError: If the Qdrant API call fails.
        """
        try:
            exists = await self._client.collection_exists(self._collection_name)
            if not exists:
                await self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(
                        size=self._embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "qdrant_collection_created",
                    collection=self._collection_name,
                    dim=self._embedding_dim,
                )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to ensure collection '{self._collection_name}': {exc}"
            ) from exc

    async def add_documents(
        self, documents: list[Document]
    ) -> dict[str, object]:
        """Embed and upsert documents into the Qdrant collection.

        Each document is assigned a deterministic UUID derived from its
        :attr:`~src.core.ingestion.base.Document.doc_id`, so re-ingesting
        the same document overwrites the existing vector instead of
        creating a duplicate.

        Args:
            documents: Chunked documents to embed and index.

        Returns:
            Dict with keys:
            - ``"inserted"``: number of points upserted.
            - ``"collection"``: collection name.

        Raises:
            VectorStoreError: If embedding or upsert fails.
        """
        if not documents:
            return {"inserted": 0, "collection": self._collection_name}

        await self._ensure_collection()

        try:
            texts = [doc.content for doc in documents]
            vectors = await self._embedding_model.aget_text_embedding_batch(
                texts, show_progress=False
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Embedding batch failed for {len(documents)} documents: {exc}"
            ) from exc

        points = [
            PointStruct(
                id=str(
                    uuid.uuid5(uuid.NAMESPACE_DNS, doc.doc_id)
                    if doc.doc_id
                    else uuid.uuid4()
                ),
                vector=vector,
                payload={
                    _CONTENT_PAYLOAD_KEY: doc.content,
                    _DOC_ID_PAYLOAD_KEY: doc.doc_id,
                    **doc.metadata,
                },
            )
            for doc, vector in zip(documents, vectors, strict=True)
        ]

        try:
            await self._client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Qdrant upsert failed for collection '{self._collection_name}': {exc}"
            ) from exc

        logger.info(
            "qdrant_documents_indexed",
            collection=self._collection_name,
            inserted=len(points),
        )

        return {"inserted": len(points), "collection": self._collection_name}

    async def search(
        self, query: str, top_k: int
    ) -> list[RetrievalResult]:
        """Embed *query* and return the *top_k* most similar chunks.

        Args:
            query: Natural language search string.
            top_k: Maximum number of results to return.

        Returns:
            List of :class:`~src.core.retrieval.base.RetrievalResult` sorted
            by descending cosine similarity score.

        Raises:
            VectorStoreError: If embedding or Qdrant search fails.
        """
        try:
            query_vector = await self._embedding_model.aget_query_embedding(query)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to embed query '{query}': {exc}"
            ) from exc

        try:
            scored_points = await self._client.search(
                collection_name=self._collection_name,
                query_vector=query_vector,
                limit=top_k,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Qdrant search failed in collection '{self._collection_name}': {exc}"
            ) from exc

        results = [
            RetrievalResult(
                content=str(point.payload.get(_CONTENT_PAYLOAD_KEY, "")),
                score=point.score,
                metadata={
                    k: v
                    for k, v in (point.payload or {}).items()
                    if k not in (_CONTENT_PAYLOAD_KEY, _DOC_ID_PAYLOAD_KEY)
                },
                doc_id=str(
                    point.payload.get(_DOC_ID_PAYLOAD_KEY, str(point.id))
                ),
            )
            for point in scored_points
        ]

        logger.info(
            "qdrant_search_complete",
            collection=self._collection_name,
            query=query,
            top_k=top_k,
            results_count=len(results),
        )

        return results
