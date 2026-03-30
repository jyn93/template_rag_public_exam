"""Streamlit exam simulation page — timed full exam session."""

from __future__ import annotations

import time
from typing import cast

import streamlit as st

from src.frontend.api_client import call_exam_evaluate_api, call_exam_generate_api
from src.frontend.pages._simulacro_helpers import (
    compute_score_summary,
    format_remaining,
)

_DEFAULT_NUM_QUESTIONS = 10
_DEFAULT_TOP_K = 5
_TIMER_OPTIONS_MINUTES = [15, 30, 45, 60, 90, 120]

# Simulation states
_STATE_IDLE = "idle"
_STATE_ACTIVE = "active"
_STATE_RESULTS = "results"

st.set_page_config(
    page_title="Simulacro de Examen — Oposiciones",
    page_icon="⏱️",
    layout="wide",
)

st.title("⏱️ Simulacro de Examen")
st.caption("Full timed exam session. Answers are locked when the timer runs out.")


# ── Session state initialisation ──────────────────────────────────────────────

_DEFAULTS: dict[str, object] = {
    "sim_state": _STATE_IDLE,
    "sim_exam_data": None,
    "sim_exam_sources": [],
    "sim_answers": {},
    "sim_evaluations": {},
    "sim_start_time": 0.0,
    "sim_duration_secs": 0,
    "sim_current_q": 0,
}

for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


def _reset_simulation() -> None:
    """Reset all simulation state to initial values."""
    for key, val in _DEFAULTS.items():
        if isinstance(val, (dict, list)):
            st.session_state[key] = type(val)()
        else:
            st.session_state[key] = val


# ── Timer fragment (auto-refreshes every second during active exam) ───────────


@st.fragment(run_every="1s")
def _timer_widget() -> None:
    """Display the countdown timer and auto-submit when time expires."""
    elapsed = time.time() - float(st.session_state.sim_start_time)
    remaining = max(0, int(st.session_state.sim_duration_secs) - int(elapsed))

    minutes_left = remaining // 60
    col_timer, col_progress = st.columns([1, 3])

    with col_timer:
        label = "⏰ Time remaining" if remaining > 60 else "🚨 Time remaining"
        st.metric(label, format_remaining(remaining))

    with col_progress:
        total = int(st.session_state.sim_duration_secs)
        fraction = remaining / total if total > 0 else 0.0
        st.progress(fraction)

    if remaining == 0 and st.session_state.sim_state == _STATE_ACTIVE:
        st.warning("⏰ Time is up! Submitting your exam…")
        st.session_state.sim_state = _STATE_RESULTS
        st.rerun()

    if minutes_left <= 5 and remaining > 0:
        st.warning(f"⚠️ Less than {minutes_left + 1} minute(s) left!")


# ── IDLE: configuration form ──────────────────────────────────────────────────

if st.session_state.sim_state == _STATE_IDLE:
    st.subheader("Configure your simulation")

    with st.form("sim_config_form"):
        col1, col2 = st.columns(2)
        with col1:
            topic = st.text_input(
                "Topic / concept",
                placeholder="e.g. Administrative appeal procedure",
            )
            subject = st.text_input(
                "Subject filter",
                value="",
                placeholder="e.g. Administrative Law (optional)",
            )
            timer_minutes = st.selectbox(
                "Timer duration (minutes)",
                options=_TIMER_OPTIONS_MINUTES,
                index=1,
                help="Exam will auto-submit when the timer reaches zero.",
            )
        with col2:
            exam_type = st.selectbox(
                "Exam type",
                options=["test", "desarrollo", "mixto"],
                index=0,
            )
            difficulty = st.selectbox(
                "Difficulty",
                options=["facil", "media", "dificil"],
                index=1,
            )
            num_questions = st.slider(
                "Number of questions",
                min_value=3,
                max_value=20,
                value=_DEFAULT_NUM_QUESTIONS,
            )
            top_k = st.slider(
                "Context chunks (top-k)",
                min_value=1,
                max_value=20,
                value=_DEFAULT_TOP_K,
            )

        start_btn = st.form_submit_button(
            "🚀 Start Simulation", type="primary", use_container_width=True
        )

    if start_btn:
        if not topic.strip():
            st.warning("Please enter a topic to generate the exam.")
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
                st.error(f"❌ Failed to generate exam: {error}")
            elif not cast("list[object]", exam.get("questions", [])):
                st.error(
                    "The API returned an exam with no questions. Please try again."
                )
            else:
                st.session_state.sim_exam_data = exam
                st.session_state.sim_exam_sources = sources
                st.session_state.sim_answers = {}
                st.session_state.sim_evaluations = {}
                st.session_state.sim_start_time = time.time()
                st.session_state.sim_duration_secs = int(timer_minutes) * 60
                st.session_state.sim_current_q = 0
                st.session_state.sim_state = _STATE_ACTIVE
                st.rerun()

# ── ACTIVE: exam in progress ──────────────────────────────────────────────────

elif st.session_state.sim_state == _STATE_ACTIVE:
    exam = cast(dict[str, object], st.session_state.sim_exam_data)
    questions = cast(list[dict[str, object]], exam.get("questions", []))
    n = len(questions)

    _timer_widget()
    st.divider()

    # Question navigation
    current = int(st.session_state.sim_current_q)
    current = max(0, min(current, n - 1))

    nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
    with nav_col1:
        if st.button("← Previous", disabled=current == 0):
            st.session_state.sim_current_q = current - 1
            st.rerun()
    with nav_col2:
        st.markdown(
            f"<div style='text-align:center; font-weight:bold;'>"
            f"Question {current + 1} / {n}"
            f"</div>",
            unsafe_allow_html=True,
        )
        answered_count = len(st.session_state.sim_answers)
        st.progress(answered_count / n if n > 0 else 0.0,
                    text=f"{answered_count}/{n} answered")
    with nav_col3:
        if st.button("Next →", disabled=current == n - 1):
            st.session_state.sim_current_q = current + 1
            st.rerun()

    st.divider()

    # Current question
    q = questions[current]
    q_id = str(q.get("id", current))
    q_text = str(q.get("question", ""))
    q_type = str(q.get("type", "test"))
    options = cast(list[str], q.get("options", []))

    with st.container(border=True):
        st.markdown(f"**Q{q_id}.** {q_text}")

        saved_answer = str(st.session_state.sim_answers.get(q_id, ""))

        if q_type == "test" and options:
            saved_index = (
                options.index(saved_answer) if saved_answer in options else None
            )
            choice = st.radio(
                "Select your answer",
                options=options,
                index=saved_index,
                key=f"sim_radio_{q_id}",
                label_visibility="collapsed",
            )
            if choice:
                st.session_state.sim_answers[q_id] = str(choice)
        else:
            text_ans = st.text_area(
                "Your answer",
                value=saved_answer,
                key=f"sim_text_{q_id}",
                height=120,
                label_visibility="collapsed",
                placeholder="Write your answer here…",
            )
            if text_ans.strip():
                st.session_state.sim_answers[q_id] = text_ans.strip()

    st.divider()

    # Submit / End exam
    col_submit, col_clear = st.columns([3, 1])
    with col_submit:
        answered = len(st.session_state.sim_answers)
        if st.button(
            f"📊 Submit Exam ({answered}/{n} answered)",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.sim_state = _STATE_RESULTS
            st.rerun()
    with col_clear:
        if st.button("🗑️ Abandon", use_container_width=True):
            _reset_simulation()
            st.rerun()

# ── RESULTS: scoring and review ───────────────────────────────────────────────

elif st.session_state.sim_state == _STATE_RESULTS:
    exam = cast(dict[str, object], st.session_state.sim_exam_data)
    questions = cast(list[dict[str, object]], exam.get("questions", []))

    # Evaluate answers if not yet done
    if not st.session_state.sim_evaluations and st.session_state.sim_answers:
        with st.spinner("Evaluating your answers…"):
            evaluations: dict[str, dict[str, object]] = {}
            errors: list[str] = []
            for q in questions:
                q_id = str(q.get("id", ""))
                if q_id not in st.session_state.sim_answers:
                    continue
                result, err = call_exam_evaluate_api(
                    question=str(q.get("question", "")),
                    correct_answer=str(q.get("correct_answer", "")),
                    student_answer=st.session_state.sim_answers[q_id],
                )
                if err:
                    errors.append(f"Q{q_id}: {err}")
                else:
                    evaluations[q_id] = result
            st.session_state.sim_evaluations = evaluations
            if errors:
                st.warning("Some evaluations failed: " + "; ".join(errors))
        st.rerun()

    # Score summary
    summary = compute_score_summary(
        questions, dict(st.session_state.sim_evaluations)
    )

    elapsed_secs = int(time.time() - float(st.session_state.sim_start_time))
    time_taken = format_remaining(
        min(elapsed_secs, int(st.session_state.sim_duration_secs))
    )

    st.subheader("📊 Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Score (avg)", f"{summary['score_avg']}/10")
    m2.metric("Correct", f"{summary['correct']}/{summary['total']}")
    m3.metric("Percentage", f"{summary['percentage']}%")
    m4.metric("Time used", time_taken)

    passed = int(summary["percentage"]) >= 50  # type: ignore[call-overload]
    if passed:
        st.success("🎉 You passed the simulation!")
    else:
        st.error("❌ You did not pass this simulation. Keep studying!")

    st.divider()
    st.subheader("Question review")

    for q in questions:
        q_id = str(q.get("id", ""))
        q_text = str(q.get("question", ""))
        correct = str(q.get("correct_answer", ""))
        student_ans = st.session_state.sim_answers.get(q_id, "*(not answered)*")
        ev = dict(st.session_state.sim_evaluations).get(q_id, {})

        is_correct = bool(ev.get("is_correct", False))
        score = ev.get("score", "—")
        icon = "✅" if is_correct else "❌"

        with st.expander(f"{icon} Q{q_id}. {q_text[:80]}…", expanded=False):
            st.markdown(f"**Your answer:** {student_ans}")
            st.markdown(f"**Correct answer:** {correct}")
            if ev:
                st.markdown(f"**Score:** {score}/10")
                st.markdown(f"**Feedback:** {ev.get('feedback', '')}")
                strengths = list(ev.get("strengths", []))
                missing = list(ev.get("missing_points", []))
                if strengths:
                    st.markdown(
                        "**Strengths:** " + ", ".join(str(s) for s in strengths)
                    )
                if missing:
                    st.markdown(
                        "**Missing points:** " + ", ".join(str(m) for m in missing)
                    )

    st.divider()
    if st.button("🔄 New Simulation", type="primary", use_container_width=True):
        _reset_simulation()
        st.rerun()
