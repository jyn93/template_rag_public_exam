"""FastAPI router for document ingestion endpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from starlette import status

from src.api.dependencies import get_ingestion_pipeline
from src.core.exceptions import IngestionError, StorageError, VectorStoreError
from src.core.ingestion.pipeline import IngestionPipeline

__all__ = ["router"]

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])

_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt"})
_MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a study document",
    response_description="Ingestion summary with chunk and document counts.",
)
async def ingest_document(
    file: UploadFile,
    subject: Annotated[str, Form(description="Exam subject / topic name.")],
    pipeline: Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)],
) -> dict[str, object]:
    """Upload and ingest a PDF or TXT study document into the RAG pipeline.

    The file is validated, temporarily saved to disk, and then processed by
    :class:`~src.core.ingestion.pipeline.IngestionPipeline` which loads,
    chunks, uploads to object storage, and indexes the content into Qdrant.

    Args:
        file: Multipart file upload. Supported formats: ``.pdf``, ``.txt``.
            Maximum size: 50 MB.
        subject: Human-readable subject or topic name (e.g. "Administrative Law").
            Used to organise the document in storage and tag all chunks.
        pipeline: Injected :class:`~src.core.ingestion.pipeline.IngestionPipeline`
            dependency.

    Returns:
        Dict with keys:
        - ``total_documents``: raw documents extracted from the file.
        - ``total_chunks``: text chunks indexed into the vector store.
        - ``subject``: the subject name echoed back.
        - ``filename``: original file name.

    Raises:
        422: If the file extension is not supported or the subject is empty.
        413: If the file exceeds the 50 MB size limit.
        500: If ingestion, storage, or vector store operations fail.
    """
    _validate_subject(subject)
    _validate_filename(file.filename or "")

    content = await file.read()
    _validate_size(len(content))

    suffix = Path(file.filename or "").suffix.lower()
    original_name = Path(file.filename or f"upload{suffix}").name

    logger.info(
        "ingestion_request",
        filename=original_name,
        subject=subject,
        size_bytes=len(content),
    )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Rename to preserve the original filename for metadata
        named_path = tmp_path.parent / original_name
        tmp_path.rename(named_path)

        result = await pipeline.ingest(named_path, subject_name=subject)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except IngestionError as exc:
        logger.error("ingestion_failed", filename=original_name, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc
    except StorageError as exc:
        logger.error("storage_failed", filename=original_name, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Storage upload failed: {exc}",
        ) from exc
    except VectorStoreError as exc:
        logger.error("vector_store_failed", filename=original_name, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector store indexing failed: {exc}",
        ) from exc
    finally:
        named_path.unlink(missing_ok=True)

    return {**result, "filename": original_name}


def _validate_subject(subject: str) -> None:
    """Raise HTTPException if subject is blank."""
    if not subject.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="subject must not be empty.",
        )


def _validate_filename(filename: str) -> None:
    """Raise HTTPException if file extension is not supported."""
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Allowed: {sorted(_ALLOWED_EXTENSIONS)}"
            ),
        )


def _validate_size(size: int) -> None:
    """Raise HTTPException if content exceeds the size limit."""
    if size > _MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"File exceeds maximum size of "
                f"{_MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
            ),
        )
