"""Hybrid retriever — BM25 + dense vector search with Reciprocal Rank Fusion."""

from __future__ import annotations

import structlog
from rank_bm25 import BM25Okapi

from src.core.exceptions import RetrievalError
from src.core.protocols import VectorSearchProtocol
from src.core.retrieval.base import DEFAULT_TOP_K, RetrievalResult, Retriever

__all__ = ["HybridRetriever"]

logger = structlog.get_logger(__name__)

_RRF_K = 60  # Standard constant for Reciprocal Rank Fusion
_DEFAULT_EXPANSION_FACTOR = 4
_DEFAULT_BM25_WEIGHT = 0.5
_DEFAULT_DENSE_WEIGHT = 0.5


class HybridRetriever(Retriever):
    """Hybrid retriever that combines dense vector search with BM25 re-ranking.

    Retrieval strategy:

    1. **Expand** — fetch ``top_k × expansion_factor`` candidates via dense
       vector search to build a rich candidate pool.
    2. **BM25 score** — score each candidate against the query using
       :class:`rank_bm25.BM25Okapi` over the candidate texts.
    3. **Reciprocal Rank Fusion** — combine the dense and BM25 rankings using
       the standard RRF formula ``1 / (k + rank)`` weighted by
       *dense_weight* and *bm25_weight*.
    4. **Select** — return the top *top_k* results sorted by fused score.

    Args:
        vector_store: Any object satisfying
            :class:`~src.core.protocols.VectorSearchProtocol`.
        top_k: Default number of final results to return.
        expansion_factor: Multiplier applied to *top_k* when fetching
            the initial candidate pool from the dense retriever.
        bm25_weight: Weight for the BM25 ranking component in RRF.
        dense_weight: Weight for the dense ranking component in RRF.

    Example:
        >>> retriever = HybridRetriever(vector_store=qdrant_store, top_k=5)
        >>> results = await retriever.retrieve("Administrative appeal procedure")
        >>> for r in results:
        ...     print(r.score, r.content[:60])
    """

    def __init__(
        self,
        vector_store: VectorSearchProtocol,
        top_k: int = DEFAULT_TOP_K,
        expansion_factor: int = _DEFAULT_EXPANSION_FACTOR,
        bm25_weight: float = _DEFAULT_BM25_WEIGHT,
        dense_weight: float = _DEFAULT_DENSE_WEIGHT,
    ) -> None:
        """Initialise the hybrid retriever.

        Args:
            vector_store: Vector store satisfying
                :class:`~src.core.protocols.VectorSearchProtocol`.
            top_k: Default number of final results.
            expansion_factor: Candidate pool size multiplier.
            bm25_weight: BM25 component weight in RRF fusion.
            dense_weight: Dense component weight in RRF fusion.
        """
        self._vector_store = vector_store
        self._top_k = top_k
        self._expansion_factor = expansion_factor
        self._bm25_weight = bm25_weight
        self._dense_weight = dense_weight

    async def retrieve(
        self, query: str, top_k: int = DEFAULT_TOP_K
    ) -> list[RetrievalResult]:
        """Retrieve and re-rank the most relevant chunks for *query*.

        Args:
            query: Natural language question or search string.
            top_k: Maximum number of results. Uses the instance default when
                equal to :data:`DEFAULT_TOP_K`.

        Returns:
            List of :class:`RetrievalResult` sorted by descending fused score.

        Raises:
            ValueError: If *query* is empty or contains only whitespace.
            RetrievalError: If the vector store raises any exception.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty or blank")

        effective_top_k = top_k if top_k != DEFAULT_TOP_K else self._top_k
        fetch_k = effective_top_k * self._expansion_factor

        logger.debug(
            "hybrid_retrieval_start",
            query=query,
            top_k=effective_top_k,
            fetch_k=fetch_k,
        )

        try:
            candidates = await self._vector_store.search(query, fetch_k)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(
                f"Hybrid retrieval failed for query '{query}': {exc}"
            ) from exc

        if not candidates:
            logger.info(
                "hybrid_retrieval_complete",
                query=query,
                top_k=effective_top_k,
                results_count=0,
            )
            return []

        results = self._fuse(query, candidates, effective_top_k)

        logger.info(
            "hybrid_retrieval_complete",
            query=query,
            top_k=effective_top_k,
            candidates_fetched=len(candidates),
            results_count=len(results),
        )

        return results

    def _fuse(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Apply BM25 scoring and Reciprocal Rank Fusion to *candidates*.

        Args:
            query: Original search query for BM25 scoring.
            candidates: Results from the dense retriever, ordered by
                descending cosine similarity (rank 0 = best).
            top_k: Number of results to return after fusion.

        Returns:
            Top *top_k* results sorted by descending fused RRF score.
        """
        # ── BM25 scoring ──────────────────────────────────────────────────────
        tokenized_corpus = [self._tokenize(r.content) for r in candidates]
        tokenized_query = self._tokenize(query)
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(tokenized_query)

        # ── Build rank maps ────────────────────────────────────────────────────
        # Dense: candidates already sorted desc by score → rank = position index
        dense_rank: dict[str, int] = {r.doc_id: i for i, r in enumerate(candidates)}

        # BM25: rank by descending BM25 score
        bm25_order = sorted(
            range(len(candidates)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )
        bm25_rank: dict[str, int] = {
            candidates[i].doc_id: rank for rank, i in enumerate(bm25_order)
        }

        # ── Reciprocal Rank Fusion ─────────────────────────────────────────────
        rrf_scores: dict[str, float] = {
            r.doc_id: (
                self._dense_weight / (_RRF_K + dense_rank[r.doc_id])
                + self._bm25_weight / (_RRF_K + bm25_rank[r.doc_id])
            )
            for r in candidates
        }

        # ── Sort and return top_k ──────────────────────────────────────────────
        return sorted(
            candidates,
            key=lambda r: rrf_scores.get(r.doc_id, 0.0),
            reverse=True,
        )[:top_k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenise *text* into lowercase whitespace-separated tokens.

        Args:
            text: Raw text to tokenise.

        Returns:
            List of lowercase word tokens.
        """
        return text.lower().split()
