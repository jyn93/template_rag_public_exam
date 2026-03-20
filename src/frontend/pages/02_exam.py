"""Streamlit exam generation page — RAG-grounded exam builder."""

from __future__ import annotations

from typing import cast

import streamlit as st

from src.frontend.api_client import call_exam_evaluate_api, call_exam_generate_api

_DEFAULT_NUM_QUESTIONS = 5
_DEFAULT_TOP_K = 5

st.set_page_config(
    page_title="Generar Examen — Oposiciones",
    page_icon="📝",
    layout="wide",
)

st.title("📝 Generar Examen")
st.caption(
    "Generate exam questions from your study material and evaluate your answers."
)

# ── Session state ─────────────────────────────────────────────────────────────

if "exam_data" not in st.session_state:
    st.session_state.exam_data = None  # dict[str, object] | None
if "exam_sources" not in st.session_state:
    st.session_state.exam_sources = []  # list[str]
if "student_answers" not in st.session_state:
    st.session_state.student_answers = {}  # dict[str, str]
if "evaluations" not in st.session_state:
    st.session_state.evaluations = {}  # dict[str, dict[str, object]]

# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Exam Settings")
    subject = st.text_input(
        "Subject filter",
        value="",
        placeholder="e.g. Administrative Law",
        help="Optional: filter retrieved context to a specific subject.",
    )
    exam_type = st.selectbox(
        "Exam type",
        options=["test", "desarrollo", "mixto"],
        index=0,
        help="test = multiple choice, desarrollo = open answer, mixto = both.",
    )
    difficulty = st.selectbox(
        "Difficulty",
        options=["facil", "media", "dificil"],
        index=1,
        help="Difficulty level of the generated questions.",
    )
    num_questions = st.slider(
        "Number of questions",
        min_value=1,
        max_value=15,
        value=_DEFAULT_NUM_QUESTIONS,
    )
    top_k = st.slider(
        "Context chunks (top-k)",
        min_value=1,
        max_value=20,
        value=_DEFAULT_TOP_K,
        help="Number of document chunks retrieved per query.",
    )
    if st.button("🗑️ Clear exam", use_container_width=True):
        st.session_state.exam_data = None
        st.session_state.exam_sources = []
        st.session_state.student_answers = {}
        st.session_state.evaluations = {}
        st.rerun()

# ── Exam generation form ──────────────────────────────────────────────────────

with st.form("generate_form"):
    topic = st.text_input(
        "Topic or concept to examine",
        placeholder="e.g. Administrative appeal procedure",
    )
    submitted = st.form_submit_button("🚀 Generate Exam", use_container_width=True)

if submitted:
    if not topic.strip():
        st.warning("Please enter a topic to generate an exam.")
    else:
        with st.spinner("Generating exam questions…"):
            exam, sources, error = call_exam_generate_api(
                query=topic,
                subject=subject,
                num_questions=num_questions,
                exam_type=str(exam_type),
                difficulty=str(difficulty),
                top_k=top_k,
            )
        if error:
            st.error(f"❌ {error}")
        else:
            st.session_state.exam_data = exam
            st.session_state.exam_sources = sources
            st.session_state.student_answers = {}
            st.session_state.evaluations = {}
            st.rerun()

# ── Display exam and collect answers ─────────────────────────────────────────

if st.session_state.exam_data:
    exam = st.session_state.exam_data
    questions = cast(list[dict[str, object]], exam.get("questions", []))

    if not questions:
        st.warning("The exam was generated but contains no questions.")
    else:
        st.subheader(f"Exam — {len(questions)} questions")

        if st.session_state.exam_sources:
            with st.expander("📎 Sources used", expanded=False):
                for i, src in enumerate(st.session_state.exam_sources, 1):
                    st.markdown(f"**[{i}]** {src}")

        for q in questions:
            q_id = str(q.get("id", ""))
            q_text = str(q.get("question", ""))
            q_type = str(q.get("type", ""))
            options = cast(list[str], q.get("options", []))
            correct = str(q.get("correct_answer", ""))

            with st.container(border=True):
                st.markdown(f"**Q{q_id}.** {q_text}")

                if q_type == "test" and options:
                    choice = st.radio(
                        "Your answer",
                        options=options,
                        key=f"radio_{q_id}",
                        index=None,
                        label_visibility="collapsed",
                    )
                    if choice:
                        st.session_state.student_answers[q_id] = str(choice)
                else:
                    answer = st.text_area(
                        "Your answer",
                        key=f"textarea_{q_id}",
                        height=100,
                        label_visibility="collapsed",
                        placeholder="Write your answer here…",
                    )
                    if answer.strip():
                        st.session_state.student_answers[q_id] = answer.strip()

                # Show evaluation result if available
                if q_id in st.session_state.evaluations:
                    ev = st.session_state.evaluations[q_id]
                    score = ev.get("score", 0)
                    is_correct = ev.get("is_correct", False)
                    feedback = ev.get("feedback", "")
                    missing = list(ev.get("missing_points", []))
                    strengths = list(ev.get("strengths", []))

                    if is_correct:
                        st.success(f"✅ Score: {score}/10 — {feedback}")
                    else:
                        st.error(f"❌ Score: {score}/10 — {feedback}")
                    if strengths:
                        st.markdown(
                            "**Strengths:** " + ", ".join(str(s) for s in strengths)
                        )
                    if missing:
                        st.markdown(
                            "**Missing points:** " + ", ".join(str(m) for m in missing)
                        )

                    with st.expander("Correct answer", expanded=False):
                        st.markdown(correct)

        # ── Evaluate button ───────────────────────────────────────────────────

        if st.session_state.student_answers:
            if st.button("📊 Evaluate My Answers", use_container_width=True):
                evaluations: dict[str, dict[str, object]] = {}
                eval_errors: list[str] = []

                with st.spinner("Evaluating answers…"):
                    for q in questions:
                        q_id = str(q.get("id", ""))
                        if q_id not in st.session_state.student_answers:
                            continue
                        result, err = call_exam_evaluate_api(
                            question=str(q.get("question", "")),
                            correct_answer=str(q.get("correct_answer", "")),
                            student_answer=st.session_state.student_answers[q_id],
                        )
                        if err:
                            eval_errors.append(f"Q{q_id}: {err}")
                        else:
                            evaluations[q_id] = result

                st.session_state.evaluations = evaluations
                if eval_errors:
                    st.error("Some evaluations failed:\n" + "\n".join(eval_errors))
                st.rerun()
