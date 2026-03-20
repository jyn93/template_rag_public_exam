"""FastAPI router for RAG chat endpoint."""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.dependencies import get_rag_generator, get_retriever
from src.core.exceptions import GenerationError, RetrievalError
from src.core.generation.base import GenerationInput, GenerationOutput
from src.core.generation.rag_generator import RAGGenerator
from src.core.retrieval.base import Retriever

__all__ = ["router"]

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    query: str = Field(..., min_length=1, description="User question.")
    subject: str = Field(
        default="",
        description="Optional subject filter (e.g. 'Administrative Law').",
    )
    top_k: int = Field(
        default=5, ge=1, le=20, description="Context chunks to retrieve."
    )


class ChatResponse(BaseModel):
    """Response body for the chat endpoint."""

    answer: str
    sources: list[str]
    tokens_used: int


@router.post(
    "",
    summary="Ask a question to the RAG system",
    response_model=ChatResponse,
)
async def chat(
    body: ChatRequest,
    retriever: Annotated[Retriever, Depends(get_retriever)],
    generator: Annotated[RAGGenerator, Depends(get_rag_generator)],
) -> ChatResponse:
    """Retrieve relevant context and generate a grounded answer.

    Args:
        body: Chat request with the user's query and optional subject filter.
        retriever: Injected :class:`~src.core.retrieval.base.Retriever`.
        generator: Injected :class:`~src.core.generation.rag_generator.RAGGenerator`.

    Returns:
        :class:`ChatResponse` with the generated answer, source excerpts,
        and token usage.

    Raises:
        422: If the query is empty.
        500: If retrieval or generation fails.
    """
    logger.info("chat_request", query=body.query[:80], subject=body.subject)

    try:
        results = await retriever.retrieve(body.query, top_k=body.top_k)
    except (RetrievalError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {exc}",
        ) from exc

    if not results:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="No relevant context found for the query.",
        )

    gen_input = GenerationInput(
        query=body.query,
        context=[r.content for r in results],
        metadata={"subject": body.subject} if body.subject else {},
    )

    try:
        output: GenerationOutput = await generator.generate(gen_input)
    except (GenerationError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {exc}",
        ) from exc

    return ChatResponse(
        answer=str(output.content),
        sources=output.sources,
        tokens_used=output.tokens_used,
    )
