"""PDF generation for practice exams."""

from __future__ import annotations

from fpdf import FPDF

__all__ = ["ExamPDFGenerator"]

_MARGIN = 15.0
_LINE_HEIGHT = 7.0
_OPTION_INDENT = 8.0


class ExamPDFGenerator:
    """Generate a printable PDF document from a structured exam dict.

    The exam dict is expected to have the shape produced by
    :class:`~src.core.generation.exam_generator.ExamGenerator`::

        {
          "title": "...",
          "questions": [
            {
              "id": 1,
              "type": "test" | "desarrollo",
              "question": "...",
              "options": ["A", "B", "C", "D"],   # test only
              "correct_answer": "...",
              "explanation": "...",               # optional
            },
            ...
          ]
        }

    Example:
        >>> gen = ExamPDFGenerator()
        >>> pdf_bytes = gen.generate(exam_dict, include_answers=False)
        >>> with open("exam.pdf", "wb") as f:
        ...     f.write(pdf_bytes)
    """

    def generate(
        self,
        exam: dict[str, object],
        include_answers: bool = False,
    ) -> bytes:
        """Render an exam to PDF bytes.

        Args:
            exam: Structured exam dict with ``title`` and ``questions`` keys.
            include_answers: If ``True``, append the answer key after all
                questions.

        Returns:
            Raw PDF bytes suitable for writing to a file or HTTP response.
        """
        pdf = FPDF()
        pdf.set_margins(_MARGIN, _MARGIN, _MARGIN)
        pdf.set_auto_page_break(auto=True, margin=_MARGIN)
        pdf.add_page()

        title = str(exam.get("title", "Practice Exam"))
        questions: list[dict[str, object]] = list(
            exam.get("questions", [])  # type: ignore[call-overload]
        )

        self._render_title(pdf, title)
        self._render_questions(pdf, questions)

        if include_answers:
            pdf.add_page()
            self._render_answer_key(pdf, questions)

        return bytes(pdf.output())

    # ── Private helpers ───────────────────────────────────────────────────────

    def _render_title(self, pdf: FPDF, title: str) -> None:
        """Render the exam title and a horizontal rule."""
        pdf.set_font("Helvetica", style="B", size=18)
        pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(2)
        pdf.set_line_width(0.5)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(6)

    def _render_questions(
        self, pdf: FPDF, questions: list[dict[str, object]]
    ) -> None:
        """Render all questions with their answer inputs."""
        for q in questions:
            q_id = str(q.get("id", ""))
            q_text = str(q.get("question", ""))
            q_type = str(q.get("type", "test"))
            options: list[str] = list(q.get("options", []))  # type: ignore[call-overload]

            # Question text
            pdf.set_font("Helvetica", style="B", size=11)
            pdf.multi_cell(
                0,
                _LINE_HEIGHT,
                f"Q{q_id}. {q_text}",
                new_x="LMARGIN",
                new_y="NEXT",
            )
            pdf.ln(1)

            if q_type == "test" and options:
                self._render_options(pdf, options)
            else:
                self._render_open_answer_box(pdf)

            pdf.ln(4)

    def _render_options(self, pdf: FPDF, options: list[str]) -> None:
        """Render multiple-choice options with a selection circle."""
        pdf.set_font("Helvetica", size=10)
        labels = ["A", "B", "C", "D", "E", "F"]
        for i, option in enumerate(options):
            label = labels[i] if i < len(labels) else str(i + 1)
            pdf.set_x(pdf.l_margin + _OPTION_INDENT)
            pdf.cell(8, _LINE_HEIGHT, f"{label})", new_x="RIGHT", new_y="TOP")
            pdf.multi_cell(
                0,
                _LINE_HEIGHT,
                option,
                new_x="LMARGIN",
                new_y="NEXT",
            )

    def _render_open_answer_box(self, pdf: FPDF) -> None:
        """Render a blank answer area for open-ended questions."""
        pdf.set_font("Helvetica", size=10)
        pdf.set_x(pdf.l_margin + _OPTION_INDENT)
        pdf.cell(0, _LINE_HEIGHT, "Answer:", new_x="LMARGIN", new_y="NEXT")
        # Draw a dotted answer box
        box_height = 30.0
        x = pdf.l_margin + _OPTION_INDENT
        y = pdf.get_y()
        w = pdf.w - pdf.l_margin - pdf.r_margin - _OPTION_INDENT
        pdf.set_draw_color(180, 180, 180)
        pdf.rect(x, y, w, box_height)
        pdf.set_draw_color(0, 0, 0)
        pdf.ln(box_height + 2)

    def _render_answer_key(
        self, pdf: FPDF, questions: list[dict[str, object]]
    ) -> None:
        """Render the answer key page."""
        pdf.set_font("Helvetica", style="B", size=16)
        pdf.cell(0, 12, "Answer Key", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(4)

        for q in questions:
            q_id = str(q.get("id", ""))
            correct = str(q.get("correct_answer", ""))
            explanation = str(q.get("explanation", ""))

            pdf.set_font("Helvetica", style="B", size=10)
            pdf.cell(0, _LINE_HEIGHT, f"Q{q_id}:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=10)
            pdf.set_x(pdf.l_margin + _OPTION_INDENT)
            pdf.multi_cell(
                0,
                _LINE_HEIGHT,
                correct,
                new_x="LMARGIN",
                new_y="NEXT",
            )
            if explanation:
                pdf.set_font("Helvetica", style="I", size=9)
                pdf.set_x(pdf.l_margin + _OPTION_INDENT)
                pdf.multi_cell(
                    0,
                    _LINE_HEIGHT,
                    explanation,
                    new_x="LMARGIN",
                    new_y="NEXT",
                )
            pdf.ln(3)
