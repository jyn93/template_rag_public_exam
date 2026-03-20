"""Unit tests for IngestionPipeline (Facade Pattern)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import IngestionError, StorageError, VectorStoreError
from src.core.ingestion.base import Document, DocumentLoader
from src.core.ingestion.pipeline import IngestionPipeline
from src.core.ingestion.txt_loader import TxtDocumentLoader

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_loader():
    """DocumentLoader mock that supports any file and returns two documents."""
    loader = MagicMock(spec=DocumentLoader)
    loader.supports.return_value = True
    loader.load.return_value = [
        Document(
            content="First document text for testing purposes.",
            metadata={"source": "test.txt", "file_type": "txt"},
            doc_id="doc-1",
        ),
        Document(
            content="Second document text for testing purposes.",
            metadata={"source": "test.txt", "file_type": "txt"},
            doc_id="doc-2",
        ),
    ]
    return loader


@pytest.fixture
def mock_splitter():
    """SentenceSplitter mock returning two fixed chunks per document."""
    splitter = MagicMock()
    splitter.split_text.side_effect = lambda text: (
        [text[:20], text[20:]] if len(text) > 20 else [text]
    )
    return splitter


@pytest.fixture
def pipeline(mock_loader, mock_vector_store, mock_doc_storage, mock_splitter):
    """IngestionPipeline with SentenceSplitter patched and all deps mocked."""
    with patch(
        "src.core.ingestion.pipeline.SentenceSplitter",
        return_value=mock_splitter,
    ):
        p = IngestionPipeline(
            loaders=[mock_loader],
            vector_store=mock_vector_store,
            doc_storage=mock_doc_storage,
        )
    return p


@pytest.fixture
def real_txt_file(tmp_path):
    """A real .txt file on disk for integration-style unit tests."""
    content = (
        "This is a real text document used for integration testing. "
        "It contains multiple sentences to exercise the chunking pipeline. "
        "The SentenceSplitter will split this text into smaller pieces."
    )
    path = tmp_path / "document.txt"
    path.write_text(content, encoding="utf-8")
    return path


# ── TestGetLoader ─────────────────────────────────────────────────────────────


class TestGetLoader:
    """Tests for IngestionPipeline._get_loader."""

    def test_returns_matching_loader(self, pipeline, mock_loader):
        """Returns the loader whose supports() returns True."""
        result = pipeline._get_loader(Path("file.txt"))

        assert result is mock_loader

    def test_raises_value_error_when_no_loader_matches(
        self, mock_vector_store, mock_doc_storage, mock_splitter
    ):
        """Raises ValueError containing the file extension when nothing matches."""
        non_supporting = MagicMock(spec=DocumentLoader)
        non_supporting.supports.return_value = False

        with patch(
            "src.core.ingestion.pipeline.SentenceSplitter",
            return_value=mock_splitter,
        ):
            p = IngestionPipeline(
                loaders=[non_supporting],
                vector_store=mock_vector_store,
                doc_storage=mock_doc_storage,
            )

        with pytest.raises(ValueError, match=r"\.xyz"):
            p._get_loader(Path("file.xyz"))

    def test_first_matching_loader_wins(
        self, mock_vector_store, mock_doc_storage, mock_splitter
    ):
        """First loader that supports the file is returned; second never checked."""
        first = MagicMock(spec=DocumentLoader)
        first.supports.return_value = True
        second = MagicMock(spec=DocumentLoader)
        second.supports.return_value = True

        with patch(
            "src.core.ingestion.pipeline.SentenceSplitter",
            return_value=mock_splitter,
        ):
            p = IngestionPipeline(
                loaders=[first, second],
                vector_store=mock_vector_store,
                doc_storage=mock_doc_storage,
            )

        result = p._get_loader(Path("file.txt"))

        assert result is first
        second.supports.assert_not_called()


# ── TestChunkDocuments ────────────────────────────────────────────────────────


class TestChunkDocuments:
    """Tests for IngestionPipeline._chunk_documents."""

    def test_chunks_preserve_parent_metadata(self, pipeline):
        """Each chunk carries the source metadata from its parent document."""
        doc = Document(
            content="Some text for chunking test.",
            metadata={"source": "doc.txt", "file_type": "txt", "subject": "Law"},
            doc_id="parent-1",
        )

        chunks = pipeline._chunk_documents([doc])

        for chunk in chunks:
            assert chunk.metadata["source"] == "doc.txt"
            assert chunk.metadata["file_type"] == "txt"
            assert chunk.metadata["subject"] == "Law"

    def test_chunk_index_added_to_metadata(self, pipeline):
        """Each chunk metadata contains a sequential 'chunk_index' key."""
        doc = Document(content="A" * 50, metadata={}, doc_id="d1")

        chunks = pipeline._chunk_documents([doc])

        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_doc_id_includes_parent_id(self, pipeline):
        """Chunk doc_ids are prefixed with the parent doc_id."""
        doc = Document(content="Some content here.", metadata={}, doc_id="parent-42")

        chunks = pipeline._chunk_documents([doc])

        for i, chunk in enumerate(chunks):
            assert chunk.doc_id == f"parent-42_chunk_{i}"

    def test_multiple_documents_all_chunked(self, pipeline, mock_splitter):
        """Two documents each producing 2 chunks yields 4 total chunks."""
        mock_splitter.split_text.side_effect = lambda text: ["part1", "part2"]
        docs = [
            Document(content="Doc one text.", metadata={}, doc_id="d1"),
            Document(content="Doc two text.", metadata={}, doc_id="d2"),
        ]

        chunks = pipeline._chunk_documents(docs)

        assert len(chunks) == 4

    def test_empty_content_produces_one_chunk(self, pipeline, mock_splitter):
        """Empty document content results in exactly one (empty) chunk."""
        mock_splitter.split_text.return_value = [""]
        doc = Document(content="", metadata={}, doc_id="empty-doc")

        chunks = pipeline._chunk_documents([doc])

        assert len(chunks) == 1
        assert chunks[0].content == ""

    def test_chunk_indices_are_sequential(self, pipeline, mock_splitter):
        """Chunk indices within a single document are 0, 1, 2, …"""
        mock_splitter.split_text.side_effect = None
        mock_splitter.split_text.return_value = ["a", "b", "c"]
        doc = Document(content="abc", metadata={}, doc_id="seq-doc")

        chunks = pipeline._chunk_documents([doc])

        assert [c.metadata["chunk_index"] for c in chunks] == [0, 1, 2]


# ── TestIngest ────────────────────────────────────────────────────────────────


class TestIngest:
    """Tests for IngestionPipeline.ingest (async)."""

    async def test_ingest_returns_summary_dict(
        self, pipeline, tmp_path, mock_loader, mock_splitter
    ):
        """Happy path: ingest returns a well-formed summary dict."""
        path = tmp_path / "doc.txt"
        path.write_text("content", encoding="utf-8")
        mock_splitter.split_text.return_value = ["chunk1", "chunk2"]

        result = await pipeline.ingest(path, subject_name="Administrative Law")

        assert result["subject"] == "Administrative Law"
        assert result["total_documents"] == len(mock_loader.load.return_value)
        assert isinstance(result["total_chunks"], int)
        assert result["total_chunks"] > 0

    async def test_ingest_calls_loader_load_with_subject(
        self, pipeline, tmp_path, mock_loader
    ):
        """loader.load is called once with the path and subject kwarg."""
        path = tmp_path / "doc.txt"
        path.write_text("content", encoding="utf-8")

        await pipeline.ingest(path, subject_name="Civil Law")

        mock_loader.load.assert_called_once_with(path, subject="Civil Law")

    async def test_ingest_calls_storage_upload(
        self, pipeline, tmp_path, mock_doc_storage
    ):
        """doc_storage.upload is called once with the path and subject_name."""
        path = tmp_path / "doc.txt"
        path.write_text("content", encoding="utf-8")

        await pipeline.ingest(path, subject_name="Civil Law")

        mock_doc_storage.upload.assert_called_once_with(path, "Civil Law")

    async def test_ingest_calls_vector_store_add_documents(
        self, pipeline, tmp_path, mock_vector_store, mock_splitter
    ):
        """vector_store.add_documents is called once with a list of Documents."""
        path = tmp_path / "doc.txt"
        path.write_text("content", encoding="utf-8")
        mock_splitter.split_text.return_value = ["chunk"]

        await pipeline.ingest(path, subject_name="Civil Law")

        mock_vector_store.add_documents.assert_called_once()
        call_args = mock_vector_store.add_documents.call_args[0][0]
        assert isinstance(call_args, list)
        assert all(isinstance(d, Document) for d in call_args)

    async def test_ingest_injects_subject_into_all_doc_metadata(
        self, pipeline, tmp_path, mock_vector_store, mock_splitter
    ):
        """All chunks passed to the vector store carry the subject in metadata."""
        path = tmp_path / "doc.txt"
        path.write_text("content", encoding="utf-8")
        mock_splitter.split_text.return_value = ["chunk"]

        await pipeline.ingest(path, subject_name="Penal Law")

        chunks = mock_vector_store.add_documents.call_args[0][0]
        for chunk in chunks:
            assert chunk.metadata["subject"] == "Penal Law"

    async def test_ingest_raises_value_error_for_missing_file(self, pipeline, tmp_path):
        """ValueError raised when the file does not exist."""
        missing = tmp_path / "nonexistent.txt"

        with pytest.raises(ValueError, match="File not found"):
            await pipeline.ingest(missing, subject_name="Law")

    async def test_ingest_raises_value_error_for_unsupported_format(
        self, mock_vector_store, mock_doc_storage, mock_splitter, tmp_path
    ):
        """ValueError raised when no loader supports the file extension."""
        non_supporting = MagicMock(spec=DocumentLoader)
        non_supporting.supports.return_value = False
        path = tmp_path / "file.xyz"
        path.write_text("data", encoding="utf-8")

        with patch(
            "src.core.ingestion.pipeline.SentenceSplitter",
            return_value=mock_splitter,
        ):
            p = IngestionPipeline(
                loaders=[non_supporting],
                vector_store=mock_vector_store,
                doc_storage=mock_doc_storage,
            )

        with pytest.raises(ValueError, match=r"\.xyz"):
            await p.ingest(path, subject_name="Law")

    async def test_ingest_raises_ingestion_error_when_loader_raises(
        self, pipeline, tmp_path, mock_loader
    ):
        """IngestionError from loader propagates unchanged."""
        path = tmp_path / "bad.txt"
        path.write_text("data", encoding="utf-8")
        mock_loader.load.side_effect = IngestionError("Parse failure")

        with pytest.raises(IngestionError, match="Parse failure"):
            await pipeline.ingest(path, subject_name="Law")

    async def test_ingest_wraps_unexpected_loader_exception(
        self, pipeline, tmp_path, mock_loader
    ):
        """Unexpected exception from loader is wrapped as IngestionError."""
        path = tmp_path / "bad.txt"
        path.write_text("data", encoding="utf-8")
        mock_loader.load.side_effect = RuntimeError("unexpected crash")

        with pytest.raises(IngestionError, match="Unexpected error loading"):
            await pipeline.ingest(path, subject_name="Law")

    async def test_ingest_raises_storage_error_when_upload_fails(
        self, pipeline, tmp_path, mock_doc_storage
    ):
        """StorageError raised when doc_storage.upload fails; original is chained."""
        path = tmp_path / "doc.txt"
        path.write_text("content", encoding="utf-8")
        original = OSError("connection refused")
        mock_doc_storage.upload.side_effect = original

        with pytest.raises(StorageError, match="Failed to upload") as exc_info:
            await pipeline.ingest(path, subject_name="Law")

        assert exc_info.value.__cause__ is original

    async def test_ingest_raises_vector_store_error_when_indexing_fails(
        self, pipeline, tmp_path, mock_vector_store
    ):
        """VectorStoreError raised when add_documents fails; original is chained."""
        path = tmp_path / "doc.txt"
        path.write_text("content", encoding="utf-8")
        original = ConnectionError("qdrant unavailable")
        mock_vector_store.add_documents.side_effect = original

        with pytest.raises(
            VectorStoreError, match="Failed to index"
        ) as exc_info:
            await pipeline.ingest(path, subject_name="Law")

        assert exc_info.value.__cause__ is original

    async def test_ingest_total_documents_count(self, pipeline, tmp_path, mock_loader):
        """total_documents matches the number of documents returned by the loader."""
        path = tmp_path / "doc.txt"
        path.write_text("content", encoding="utf-8")
        mock_loader.load.return_value = [
            Document(content="Doc 1", metadata={}, doc_id="d1"),
            Document(content="Doc 2", metadata={}, doc_id="d2"),
            Document(content="Doc 3", metadata={}, doc_id="d3"),
        ]

        result = await pipeline.ingest(path, subject_name="Law")

        assert result["total_documents"] == 3

    async def test_ingest_total_chunks_count(self, pipeline, tmp_path, mock_splitter):
        """total_chunks matches the chunks produced by _chunk_documents."""
        path = tmp_path / "doc.txt"
        path.write_text("content", encoding="utf-8")
        # mock_loader returns 2 docs; splitter returns 2 chunks each → 4 total
        mock_splitter.split_text.return_value = ["chunk_a", "chunk_b"]

        result = await pipeline.ingest(path, subject_name="Law")

        assert result["total_chunks"] == 4  # 2 docs × 2 chunks


# ── TestIngestWithRealTxtFile ─────────────────────────────────────────────────


class TestIngestWithRealTxtFile:
    """End-to-end tests using a real TxtDocumentLoader and real SentenceSplitter."""

    async def test_ingest_real_file_returns_valid_summary(
        self, real_txt_file, mock_vector_store, mock_doc_storage
    ):
        """Ingest a real .txt file end-to-end; summary dict is well-formed."""
        p = IngestionPipeline(
            loaders=[TxtDocumentLoader()],
            vector_store=mock_vector_store,
            doc_storage=mock_doc_storage,
        )

        result = await p.ingest(real_txt_file, subject_name="Test Subject")

        assert result["subject"] == "Test Subject"
        assert result["total_documents"] >= 1
        assert result["total_chunks"] >= 1

    async def test_ingest_real_file_vector_store_receives_documents(
        self, real_txt_file, mock_vector_store, mock_doc_storage
    ):
        """vector_store.add_documents is called with non-empty Document list."""
        p = IngestionPipeline(
            loaders=[TxtDocumentLoader()],
            vector_store=mock_vector_store,
            doc_storage=mock_doc_storage,
        )

        await p.ingest(real_txt_file, subject_name="Test Subject")

        mock_vector_store.add_documents.assert_called_once()
        docs_sent = mock_vector_store.add_documents.call_args[0][0]
        assert len(docs_sent) >= 1
        assert all(isinstance(d, Document) for d in docs_sent)

    async def test_ingest_real_file_subject_in_all_chunks(
        self, real_txt_file, mock_vector_store, mock_doc_storage
    ):
        """All chunks sent to the vector store have the correct subject metadata."""
        p = IngestionPipeline(
            loaders=[TxtDocumentLoader()],
            vector_store=mock_vector_store,
            doc_storage=mock_doc_storage,
        )

        await p.ingest(real_txt_file, subject_name="Real Subject")

        chunks = mock_vector_store.add_documents.call_args[0][0]
        for chunk in chunks:
            assert chunk.metadata["subject"] == "Real Subject"
