"""Unit tests for the document loading Strategy pattern."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.exceptions import IngestionError
from src.core.ingestion.base import Document, DocumentLoader
from src.core.ingestion.txt_loader import TxtDocumentLoader

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def txt_loader() -> TxtDocumentLoader:
    """Return a default TxtDocumentLoader instance."""
    return TxtDocumentLoader()


@pytest.fixture
def sample_txt_file(tmp_path: Path) -> Path:
    """Create a temporary .txt file with known content."""
    file = tmp_path / "temario.txt"
    file.write_text("This is a test document.\nSecond line.", encoding="utf-8")
    return file


# ---------------------------------------------------------------------------
# Document dataclass
# ---------------------------------------------------------------------------


class TestDocument:
    """Tests for the Document dataclass."""

    def test_fields_stored_correctly(self) -> None:
        """Verify all fields are accessible after construction."""
        doc = Document(
            content="Hello world",
            metadata={"source": "test.txt"},
            doc_id="abc-123",
        )
        assert doc.content == "Hello world"
        assert doc.metadata["source"] == "test.txt"
        assert doc.doc_id == "abc-123"

    def test_default_metadata_is_empty_dict(self) -> None:
        """Verify metadata defaults to an empty dict, not a shared mutable."""
        doc1 = Document(content="a")
        doc2 = Document(content="b")
        doc1.metadata["key"] = "value"
        assert "key" not in doc2.metadata

    def test_default_doc_id_is_empty_string(self) -> None:
        """Verify doc_id defaults to an empty string."""
        doc = Document(content="x")
        assert doc.doc_id == ""


# ---------------------------------------------------------------------------
# DocumentLoader ABC
# ---------------------------------------------------------------------------


class TestDocumentLoaderInterface:
    """Verify the ABC contract is enforced correctly."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Verify DocumentLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DocumentLoader()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_load(self) -> None:
        """Verify a subclass that skips load() cannot be instantiated."""

        class IncompleteLoader(DocumentLoader):
            def supports(self, path: Path) -> bool:
                return True

        with pytest.raises(TypeError):
            IncompleteLoader()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_supports(self) -> None:
        """Verify a subclass that skips supports() cannot be instantiated."""

        class IncompleteLoader(DocumentLoader):
            def load(self, path: Path, subject: str = "") -> list[Document]:
                return []

        with pytest.raises(TypeError):
            IncompleteLoader()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# TxtDocumentLoader.supports()
# ---------------------------------------------------------------------------


class TestTxtDocumentLoaderSupports:
    """Tests for TxtDocumentLoader.supports()."""

    @pytest.mark.parametrize("filename", ["file.txt", "FILE.TXT", "temario.TxT"])
    def test_supports_txt_extensions(
        self, txt_loader: TxtDocumentLoader, filename: str
    ) -> None:
        """Verify .txt is accepted regardless of case."""
        assert txt_loader.supports(Path(filename)) is True

    @pytest.mark.parametrize("filename", ["file.pdf", "file.docx", "file.md", "file"])
    def test_rejects_non_txt_extensions(
        self, txt_loader: TxtDocumentLoader, filename: str
    ) -> None:
        """Verify non-.txt files are rejected."""
        assert txt_loader.supports(Path(filename)) is False


# ---------------------------------------------------------------------------
# TxtDocumentLoader.load()
# ---------------------------------------------------------------------------


class TestTxtDocumentLoaderLoad:
    """Tests for TxtDocumentLoader.load()."""

    def test_returns_single_document(
        self, txt_loader: TxtDocumentLoader, sample_txt_file: Path
    ) -> None:
        """Verify load() returns exactly one Document per file."""
        docs = txt_loader.load(sample_txt_file)
        assert len(docs) == 1

    def test_document_content_matches_file(
        self, txt_loader: TxtDocumentLoader, sample_txt_file: Path
    ) -> None:
        """Verify the returned Document contains the full file text."""
        docs = txt_loader.load(sample_txt_file)
        assert "This is a test document." in docs[0].content
        assert "Second line." in docs[0].content

    def test_metadata_includes_source_filename(
        self, txt_loader: TxtDocumentLoader, sample_txt_file: Path
    ) -> None:
        """Verify metadata['source'] is set to the file name."""
        docs = txt_loader.load(sample_txt_file)
        assert docs[0].metadata["source"] == sample_txt_file.name

    def test_metadata_file_type_is_txt(
        self, txt_loader: TxtDocumentLoader, sample_txt_file: Path
    ) -> None:
        """Verify metadata['file_type'] is 'txt'."""
        docs = txt_loader.load(sample_txt_file)
        assert docs[0].metadata["file_type"] == "txt"

    def test_metadata_subject_stored(
        self, txt_loader: TxtDocumentLoader, sample_txt_file: Path
    ) -> None:
        """Verify the subject argument is persisted in metadata."""
        docs = txt_loader.load(sample_txt_file, subject="Constitutional Law")
        assert docs[0].metadata["subject"] == "Constitutional Law"

    def test_doc_id_is_non_empty_string(
        self, txt_loader: TxtDocumentLoader, sample_txt_file: Path
    ) -> None:
        """Verify doc_id is a non-empty string (UUID5)."""
        docs = txt_loader.load(sample_txt_file)
        assert isinstance(docs[0].doc_id, str)
        assert len(docs[0].doc_id) > 0

    def test_same_file_produces_same_doc_id(
        self, txt_loader: TxtDocumentLoader, sample_txt_file: Path
    ) -> None:
        """Verify the same file always yields the same deterministic doc_id."""
        docs1 = txt_loader.load(sample_txt_file)
        docs2 = txt_loader.load(sample_txt_file)
        assert docs1[0].doc_id == docs2[0].doc_id

    def test_raises_value_error_for_missing_file(
        self, txt_loader: TxtDocumentLoader, tmp_path: Path
    ) -> None:
        """Verify ValueError is raised when the file does not exist."""
        missing = tmp_path / "nonexistent.txt"
        with pytest.raises(ValueError, match="File not found"):
            txt_loader.load(missing)

    def test_raises_ingestion_error_for_bad_encoding(
        self, txt_loader: TxtDocumentLoader, tmp_path: Path
    ) -> None:
        """Verify IngestionError is raised when file encoding is incompatible."""
        bad_file = tmp_path / "bad_encoding.txt"
        # Write raw bytes that are invalid UTF-8
        bad_file.write_bytes(b"\xff\xfe invalid utf-8 \x80\x81")
        with pytest.raises(IngestionError, match="Cannot decode"):
            txt_loader.load(bad_file)

    def test_empty_file_returns_document_with_empty_content(
        self, txt_loader: TxtDocumentLoader, tmp_path: Path
    ) -> None:
        """Verify an empty .txt file yields a Document with empty content."""
        empty = tmp_path / "empty.txt"
        empty.write_text("", encoding="utf-8")
        docs = txt_loader.load(empty)
        assert len(docs) == 1
        assert docs[0].content == ""

    def test_custom_encoding_accepted(self, tmp_path: Path) -> None:
        """Verify a loader with a custom encoding reads the file correctly."""
        file = tmp_path / "latin.txt"
        text = "Árboles y cañones"
        file.write_text(text, encoding="latin-1")
        loader = TxtDocumentLoader(encoding="latin-1")
        docs = loader.load(file)
        assert docs[0].content == text
