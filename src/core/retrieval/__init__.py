"""Public API for the retrieval layer."""

from src.core.retrieval.base import DEFAULT_TOP_K, RetrievalResult, Retriever
from src.core.retrieval.dense_retriever import DenseRetriever

__all__ = ["DEFAULT_TOP_K", "DenseRetriever", "RetrievalResult", "Retriever"]
