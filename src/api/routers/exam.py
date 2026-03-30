"""FastAPI router for exam generation and answer evaluation endpoints."""

from __future__ import annotations

from typing import Annotated, Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.api.dependencies import get_answer_evaluator, get_llm_client, get_retriever
from src.core.exceptions import GenerationError, RetrievalError
from src.core.export.pdf_generator import ExamPDFGenerator
from src.core.generation.base import GenerationInput
from src.core.generation.evaluator import AnswerEvaluator
from src.core.generation.exam_generator import ExamGenerator
from src.core.retrieval.base import Retriever
from src.infrastructure.llm.base import LLMClient

__all__ = ["router"]

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/exam", tags=["exam"])

_VALID_EXAM_TYPES = {"test", "desarrollo", "mixto"}
_VALID_DIFFICULTIES = {"facil", "media", "dificil"}


# ── Request / Response models ─────────────────────────────────────────────────


class ExamGenerateRequest(BaseModel):
    """Request body for exam generation."""

    query: str = Field(..., min_length=1, description="Topic or concept to examine.")
    subject: str = Field(default="", description="Subject / topic label.")
    num_questions: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of questions to generate.",
    )
    exam_type: Literal["test", "desarrollo", "mixto"] = Field(
        default="test",
        description="Exam format.",
    )
    difficulty: Literal["facil", "media", "dificil"] = Field(
        default="media",
        description="Difficulty level.",
    )
    top_k: int = Field(
        default=5, ge=1, le=20, description="Context chunks to retrieve."
    )


class ExamGenerateResponse(BaseModel):
    """Response body for exam generation."""

    exam: dict[str, object]
    sources: list[str]
    tokens_used: int


class EvaluateRequest(BaseModel):
    """Request body for answer evaluation."""

    question: str = Field(..., min_length=1, description="The exam question.")
    correct_answer: str = Field(
        ..., min_length=1, description="Reference correct answer."
    )
    student_answer: str = Field(
        ..., min_length=1, description="The student's answer to evaluate."
    )


class EvaluateResponse(BaseModel):
    """Response body for answer evaluation."""

    score: float
    is_correct: bool
    feedback: str
    missing_points: list[str]
    strengths: list[str]


class ExamExportRequest(BaseModel):
    """Request body for exam PDF export."""

    exam: dict[str, object] = Field(
        ..., description="Structured exam dict produced by /exam/generate."
    )
    include_answers: bool = Field(
        default=False,
        description="If true, append an answer key page to the PDF.",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post(
    "/generate",
    summary="Generate a practice exam from retrieved context",
    response_model=ExamGenerateResponse,
    status_code=status.HTTP_200_OK,
)
async def generate_exam(
    body: ExamGenerateRequest,
    retriever: Annotated[Retriever, Depends(get_retriever)],
    llm_client: Annotated[LLMClient, Depends(get_llm_client)],
) -> ExamGenerateResponse:
    """Retrieve study context and generate a structured practice exam.

    Args:
        body: Exam generation parameters including topic, format, and difficulty.
        retriever: Injected context retriever.
        llm_client: Injected LLM client.

    Returns:
        :class:`ExamGenerateResponse` with the JSON exam, source excerpts,
        and token usage.

    Raises:
        422: If no context is found for the query.
        500: If retrieval or generation fails.
    """
    logger.info(
        "exam_generate_request",
        query=body.query[:80],
        exam_type=body.exam_type,
        num_questions=body.num_questions,
    )

    try:
        results = await retriever.retrieve(body.query, top_k=body.top_k)
    except (RetrievalError, ValueError) as exc:
        logger.exception(
            "exam_retrieval_failed", query=body.query[:80], error=str(exc)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {exc}",
        ) from exc

    if not results:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="No relevant context found for the query.",
        )

    generator = ExamGenerator(
        llm_client=llm_client,
        num_questions=body.num_questions,
        exam_type=body.exam_type,
        difficulty=body.difficulty,
    )

    gen_input = GenerationInput(
        query=body.query,
        context=[r.content for r in results],
        metadata={"subject": body.subject} if body.subject else {},
    )

    try:
        output = await generator.generate(gen_input)
    except ValueError as exc:
        logger.exception(
            "exam_generation_invalid_input", query=body.query[:80], error=str(exc)
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid generation input: {exc}",
        ) from exc
    except GenerationError as exc:
        logger.exception(
            "exam_generation_failed", query=body.query[:80], error=str(exc)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Exam generation failed: {exc}",
        ) from exc

    exam_content = output.content if isinstance(output.content, dict) else {}
    return ExamGenerateResponse(
        exam=exam_content,
        sources=output.sources,
        tokens_used=output.tokens_used,
    )


@router.post(
    "/evaluate",
    summary="Evaluate a student answer against the reference answer",
    response_model=EvaluateResponse,
    status_code=status.HTTP_200_OK,
)
async def evaluate_answer(
    body: EvaluateRequest,
    evaluator: Annotated[AnswerEvaluator, Depends(get_answer_evaluator)],
) -> EvaluateResponse:
    """Score a student's answer using an LLM judge.

    Args:
        body: The question, reference answer, and student answer.
        evaluator: Injected :class:`~src.core.generation.evaluator.AnswerEvaluator`.

    Returns:
        :class:`EvaluateResponse` with score, feedback, and missing points.

    Raises:
        500: If the LLM evaluation call fails.
    """
    logger.info("exam_evaluate_request", question=body.question[:80])

    try:
        result = await evaluator.evaluate(
            question=body.question,
            correct_answer=body.correct_answer,
            student_answer=body.student_answer,
        )
    except GenerationError as exc:
        logger.exception(
            "exam_evaluation_failed", question=body.question[:80], error=str(exc)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {exc}",
        ) from exc

    return EvaluateResponse(
        score=float(result.score),
        is_correct=result.is_correct,
        feedback=result.feedback,
        missing_points=result.missing_points,
        strengths=result.strengths,
    )


@router.post(
    "/export",
    summary="Export a generated exam as a PDF file",
    response_class=Response,
    responses={
        200: {
            "content": {"application/pdf": {}},
            "description": "PDF file download.",
        }
    },
    status_code=status.HTTP_200_OK,
)
async def export_exam_pdf(body: ExamExportRequest) -> Response:
    """Render a structured exam dict to a downloadable PDF.

    Args:
        body: The exam dict (from ``/exam/generate``) and optional flag to
            include the answer key page.

    Returns:
        A ``application/pdf`` response with the PDF bytes and a
        ``Content-Disposition: attachment`` header.

    Raises:
        500: If PDF generation fails unexpectedly.
    """
    logger.info(
        "exam_export_request",
        num_questions=len(list(body.exam.get("questions", []))),  # type: ignore[call-overload]
        include_answers=body.include_answers,
    )

    try:
        generator = ExamPDFGenerator()
        pdf_bytes = generator.generate(body.exam, include_answers=body.include_answers)
    except Exception as exc:
        logger.exception("exam_export_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF generation failed: {exc}",
        ) from exc

    title = str(body.exam.get("title", "exam")).replace(" ", "_")
    filename = f"{title}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
