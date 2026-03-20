"""Unit tests for MinIOStorage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from minio.error import S3Error

from src.core.exceptions import StorageError
from src.infrastructure.storage.minio_storage import MinIOStorage

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_storage(bucket_exists: bool = True) -> tuple[MinIOStorage, MagicMock]:
    """Return a MinIOStorage instance with a mocked Minio client."""
    mock_client = MagicMock()
    mock_client.bucket_exists.return_value = bucket_exists
    mock_client.make_bucket.return_value = None
    mock_client.put_object.return_value = None
    storage = MinIOStorage(client=mock_client, bucket="test-bucket")
    return storage, mock_client


# ── TestMinIOStorageInit ──────────────────────────────────────────────────────


class TestMinIOStorageInit:
    """Tests for MinIOStorage construction."""

    def test_custom_client_stored(self):
        """Provided client is stored without creating a new one."""
        mock_client = MagicMock()
        storage = MinIOStorage(client=mock_client, bucket="b")
        assert storage._client is mock_client

    def test_custom_bucket_stored(self):
        """Custom bucket name is used over settings default."""
        mock_client = MagicMock()
        storage = MinIOStorage(client=mock_client, bucket="my-bucket")
        assert storage._bucket == "my-bucket"

    def test_default_bucket_from_settings(self):
        """When bucket is None, bucket is read from settings."""
        mock_client = MagicMock()
        patch_target = "src.infrastructure.storage.minio_storage.get_settings"
        with patch(patch_target) as mock_settings:
            mock_settings.return_value.minio_bucket = "default-bucket"
            mock_settings.return_value.minio_endpoint = "localhost:9000"
            mock_settings.return_value.minio_access_key = "key"
            mock_settings.return_value.minio_secret_key = "secret"
            storage = MinIOStorage(client=mock_client)
        assert storage._bucket == "default-bucket"


# ── TestMinIOStorageUpload ────────────────────────────────────────────────────


class TestMinIOStorageUpload:
    """Tests for MinIOStorage.upload."""

    async def test_upload_returns_expected_keys(self, tmp_path):
        """upload() returns bucket, key, and size."""
        storage, _ = make_storage()
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"pdf content")

        result = await storage.upload(f, "Civil Law")

        assert result["bucket"] == "test-bucket"
        assert result["key"] == "Civil Law/doc.pdf"
        assert result["size"] == len(b"pdf content")

    async def test_upload_key_includes_subject(self, tmp_path):
        """Object key is <subject>/<filename>."""
        storage, _ = make_storage()
        f = tmp_path / "notes.txt"
        f.write_bytes(b"notes")

        result = await storage.upload(f, "Administrative Law")

        assert result["key"] == "Administrative Law/notes.txt"

    async def test_bucket_created_when_missing(self, tmp_path):
        """make_bucket is called when the bucket does not exist."""
        storage, mock_client = make_storage(bucket_exists=False)
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"data")

        await storage.upload(f, "subject")

        mock_client.make_bucket.assert_called_once_with("test-bucket")

    async def test_bucket_not_created_when_exists(self, tmp_path):
        """make_bucket is NOT called when the bucket already exists."""
        storage, mock_client = make_storage(bucket_exists=True)
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"data")

        await storage.upload(f, "subject")

        mock_client.make_bucket.assert_not_called()

    async def test_put_object_called_once(self, tmp_path):
        """put_object is called exactly once per upload."""
        storage, mock_client = make_storage()
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"hello")

        await storage.upload(f, "subject")

        mock_client.put_object.assert_called_once()

    async def test_s3_error_raises_storage_error(self, tmp_path):
        """S3Error from put_object is wrapped in StorageError."""
        storage, mock_client = make_storage()
        mock_client.put_object.side_effect = S3Error(
            "NoSuchBucket", "bucket not found", "", "", "", None
        )
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"data")

        with pytest.raises(StorageError, match="MinIO upload failed"):
            await storage.upload(f, "subject")

    async def test_unexpected_error_raises_storage_error(self, tmp_path):
        """Any unexpected exception is wrapped in StorageError."""
        storage, mock_client = make_storage()
        mock_client.put_object.side_effect = ConnectionError("timeout")
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"data")

        with pytest.raises(StorageError, match="Unexpected error"):
            await storage.upload(f, "subject")

    async def test_pdf_content_type(self, tmp_path):
        """PDF files use application/pdf content type."""
        storage, mock_client = make_storage()
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"pdf")

        await storage.upload(f, "subj")

        call_kwargs = mock_client.put_object.call_args.kwargs
        assert call_kwargs["content_type"] == "application/pdf"

    async def test_txt_content_type(self, tmp_path):
        """TXT files use text/plain content type."""
        storage, mock_client = make_storage()
        f = tmp_path / "notes.txt"
        f.write_bytes(b"text")

        await storage.upload(f, "subj")

        call_kwargs = mock_client.put_object.call_args.kwargs
        assert call_kwargs["content_type"] == "text/plain"

    async def test_unknown_extension_uses_octet_stream(self, tmp_path):
        """Unknown extensions fall back to application/octet-stream."""
        storage, mock_client = make_storage()
        f = tmp_path / "data.bin"
        f.write_bytes(b"binary")

        await storage.upload(f, "subj")

        call_kwargs = mock_client.put_object.call_args.kwargs
        assert call_kwargs["content_type"] == "application/octet-stream"
