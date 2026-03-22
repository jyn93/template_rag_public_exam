"""MinIO object storage adapter implementing DocStorageProtocol."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from minio import Minio
from minio.error import S3Error

from src.core.config.settings import get_settings
from src.core.exceptions import StorageError

if TYPE_CHECKING:
    pass

__all__ = ["MinIOStorage"]

logger = structlog.get_logger(__name__)


class MinIOStorage:
    """MinIO-backed document storage satisfying DocStorageProtocol.

    Uploads source files to a MinIO bucket organised by subject name.
    All blocking MinIO SDK calls are dispatched to a thread pool via
    :func:`asyncio.to_thread` to keep the async event loop unblocked.

    Object keys follow the pattern ``<subject>/<filename>`` so files are
    naturally grouped by exam subject inside the bucket.

    Args:
        client: Pre-configured :class:`~minio.Minio` client instance.
            Defaults to a client built from application settings.
        bucket: Target bucket name. Defaults to ``settings.minio_bucket``.

    Example:
        >>> storage = MinIOStorage()
        >>> result = await storage.upload(Path("temario.pdf"), subject="Civil Law")
        >>> print(result["key"])
        Civil Law/temario.pdf
    """

    def __init__(
        self,
        client: Minio | None = None,
        bucket: str | None = None,
    ) -> None:
        """Initialise the storage adapter.

        Args:
            client: Optional Minio client.  When ``None`` a client is
                constructed from application settings.
            bucket: Target bucket name.  Defaults to ``settings.minio_bucket``.
        """
        settings = get_settings()
        self._bucket = bucket or settings.minio_bucket
        if client is not None:
            self._client = client
        else:
            self._client = Minio(
                endpoint=settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=False,
            )

    async def upload(self, path: Path, subject: str) -> dict[str, object]:
        """Upload *path* to MinIO under ``<subject>/<filename>``.

        The bucket is created if it does not already exist.  The original
        file is read in its entirety and uploaded as a single object.

        Args:
            path: Local path to the file to upload.
            subject: Subject / topic name used as the object key prefix.

        Returns:
            Dict with keys:
            - ``bucket``: target bucket name.
            - ``key``: full object key (``<subject>/<filename>``).
            - ``size``: number of bytes uploaded.

        Raises:
            StorageError: If the upload fails for any reason.
        """
        key = f"{subject}/{path.name}"
        data = path.read_bytes()
        size = len(data)

        try:
            await asyncio.to_thread(self._ensure_bucket)
            await asyncio.to_thread(
                self._put_object,
                key,
                data,
                size,
                path.suffix,
            )
        except S3Error as exc:
            logger.exception(
                "minio_upload_s3_error", bucket=self._bucket, key=key, error=str(exc)
            )
            raise StorageError(f"MinIO upload failed for '{path.name}': {exc}") from exc
        except Exception as exc:
            logger.exception(
                "minio_upload_unexpected_error",
                bucket=self._bucket,
                key=key,
                error=str(exc),
            )
            raise StorageError(
                f"Unexpected error uploading '{path.name}' to MinIO"
            ) from exc

        logger.info("minio_upload_complete", bucket=self._bucket, key=key, size=size)
        return {"bucket": self._bucket, "key": key, "size": size}

    def _ensure_bucket(self) -> None:
        """Create the bucket if it does not exist (synchronous helper)."""
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    def _put_object(
        self,
        key: str,
        data: bytes,
        size: int,
        suffix: str,
    ) -> None:
        """Upload bytes as an object (synchronous helper).

        Args:
            key: Object key in the bucket.
            data: Raw file bytes.
            size: Number of bytes.
            suffix: File extension used to determine content type.
        """
        content_type = _CONTENT_TYPES.get(suffix.lower(), "application/octet-stream")
        self._client.put_object(
            bucket_name=self._bucket,
            object_name=key,
            data=io.BytesIO(data),
            length=size,
            content_type=content_type,
        )


_CONTENT_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
}
