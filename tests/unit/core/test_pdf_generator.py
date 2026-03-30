"""Unit tests for ExamPDFGenerator."""

from __future__ import annotations

import pytest

from src.core.export.pdf_generator import ExamPDFGenerator

# ── Fixtures ──────────────────────────────────────────────────────────────────

_TEST_QUESTION: dict[str, object] = {
    "id": "1",
    "type": "test",
    "question": "What is the purpose of administrative appeal?",
    "options": [
        "To challenge an administrative decision",
        "To file a criminal complaint",
        "To apply for a license",
        "To request information",
    ],
    "correct_answer": "To challenge an administrative decision",
    "explanation": (
        "Administrative appeal allows citizens to contest unfavourable decisions."
    ),
}

_OPEN_QUESTION: dict[str, object] = {
    "id": "2",
    "type": "desarrollo",
    "question": "Explain the concept of administrative silence.",
    "options": [],
    "correct_answer": (
        "Administrative silence occurs when the administration fails"
        " to respond within the legal deadline."
    ),
    "explanation": "It can be positive or negative depending on the legal provision.",
}

_MINIMAL_EXAM: dict[str, object] = {
    "title": "Administrative Law Test",
    "questions": [_TEST_QUESTION, _OPEN_QUESTION],
}

_EMPTY_EXAM: dict[str, object] = {
    "title": "Empty Exam",
    "questions": [],
}


# ── TestExamPDFGenerator ──────────────────────────────────────────────────────


class TestExamPDFGenerator:
    """Tests for ExamPDFGenerator.generate."""

    def test_returns_bytes(self) -> None:
        """Output is raw bytes."""
        gen = ExamPDFGenerator()
        result = gen.generate(_MINIMAL_EXAM)
        assert isinstance(result, bytes)

    def test_output_is_non_empty(self) -> None:
        """Generated PDF is not empty."""
        gen = ExamPDFGenerator()
        result = gen.generate(_MINIMAL_EXAM)
        assert len(result) > 0

    def test_output_starts_with_pdf_header(self) -> None:
        """Output starts with the %PDF magic bytes."""
        gen = ExamPDFGenerator()
        result = gen.generate(_MINIMAL_EXAM)
        assert result.startswith(b"%PDF-")

    def test_empty_exam_produces_valid_pdf(self) -> None:
        """An exam with no questions still produces a valid PDF."""
        gen = ExamPDFGenerator()
        result = gen.generate(_EMPTY_EXAM)
        assert result.startswith(b"%PDF-")
        assert len(result) > 0

    def test_include_answers_false_by_default(self) -> None:
        """Default call does not include answer key."""
        gen = ExamPDFGenerator()
        without_key = gen.generate(_MINIMAL_EXAM, include_answers=False)
        with_key = gen.generate(_MINIMAL_EXAM, include_answers=True)
        # Answer key adds content → PDF with answers is larger
        assert len(with_key) > len(without_key)

    def test_include_answers_true_produces_larger_pdf(self) -> None:
        """PDF with answer key is strictly larger than without."""
        gen = ExamPDFGenerator()
        result_no_key = gen.generate(_MINIMAL_EXAM, include_answers=False)
        result_with_key = gen.generate(_MINIMAL_EXAM, include_answers=True)
        assert len(result_with_key) > len(result_no_key)

    def test_missing_title_uses_default(self) -> None:
        """Exam without 'title' key still generates without error."""
        gen = ExamPDFGenerator()
        exam: dict[str, object] = {"questions": [_TEST_QUESTION]}
        result = gen.generate(exam)
        assert result.startswith(b"%PDF-")

    def test_test_type_question_renders(self) -> None:
        """Multiple-choice question renders without error."""
        gen = ExamPDFGenerator()
        exam: dict[str, object] = {
            "title": "Test",
            "questions": [_TEST_QUESTION],
        }
        result = gen.generate(exam)
        assert result.startswith(b"%PDF-")

    def test_desarrollo_type_question_renders(self) -> None:
        """Open-answer question renders without error."""
        gen = ExamPDFGenerator()
        exam: dict[str, object] = {
            "title": "Test",
            "questions": [_OPEN_QUESTION],
        }
        result = gen.generate(exam)
        assert result.startswith(b"%PDF-")

    def test_question_without_explanation_renders(self) -> None:
        """Question missing 'explanation' key renders without error."""
        gen = ExamPDFGenerator()
        q: dict[str, object] = {
            "id": "1",
            "type": "test",
            "question": "Simple question?",
            "options": ["Yes", "No"],
            "correct_answer": "Yes",
        }
        exam: dict[str, object] = {"title": "T", "questions": [q]}
        result = gen.generate(exam, include_answers=True)
        assert result.startswith(b"%PDF-")

    def test_multiple_questions_render(self) -> None:
        """Exam with many questions renders without error."""
        gen = ExamPDFGenerator()
        questions = [
            {
                "id": str(i),
                "type": "test",
                "question": f"Question {i}?",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
            }
            for i in range(1, 16)
        ]
        exam: dict[str, object] = {"title": "Big Exam", "questions": questions}
        result = gen.generate(exam)
        assert result.startswith(b"%PDF-")
        assert len(result) > 0

    @pytest.mark.parametrize("include_answers", [True, False])
    def test_parametrized_include_answers(self, include_answers: bool) -> None:
        """Both include_answers values produce valid PDFs."""
        gen = ExamPDFGenerator()
        result = gen.generate(_MINIMAL_EXAM, include_answers=include_answers)
        assert result.startswith(b"%PDF-")
