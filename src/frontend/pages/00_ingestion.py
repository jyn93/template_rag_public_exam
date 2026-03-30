"""Streamlit document ingestion page — admin-only document upload."""

from __future__ import annotations

import os

import streamlit as st

from src.frontend.api_client import call_ingest_api

_ALLOWED_TYPES = ["pdf", "txt"]
_PASSWORD_ENV_VAR = "INGESTION_ADMIN_PASSWORD"  # noqa: S105

st.set_page_config(
    page_title="Document Ingestion — Oposiciones",
    page_icon="📥",
    layout="wide",
)

st.title("📥 Document Ingestion")
st.caption("Upload study documents (PDF / TXT) to index them into the RAG pipeline.")

# ── Password gate ─────────────────────────────────────────────────────────────

_admin_password = os.getenv(_PASSWORD_ENV_VAR, "")

if _admin_password:
    if "ingestion_authenticated" not in st.session_state:
        st.session_state.ingestion_authenticated = False

    if not st.session_state.ingestion_authenticated:
        st.subheader("🔐 Admin access required")
        entered = st.text_input("Password", type="password", key="ingestion_password")
        if st.button("Unlock", type="primary"):
            if entered == _admin_password:
                st.session_state.ingestion_authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.stop()

# ── Upload form ───────────────────────────────────────────────────────────────

with st.form("ingestion_form", clear_on_submit=True):
    st.subheader("Upload a document")

    uploaded_file = st.file_uploader(
        "Select file",
        type=_ALLOWED_TYPES,
        help="Supported formats: PDF, TXT. Maximum size: 50 MB.",
    )
    subject = st.text_input(
        "Subject / topic name",
        placeholder="e.g. Administrative Law, Constitutional Law…",
        help="Used to tag all chunks in the vector store for filtered retrieval.",
    )
    submitted = st.form_submit_button("📤 Ingest document", type="primary")

if submitted:
    if uploaded_file is None:
        st.warning("Please select a file before submitting.")
    elif not subject.strip():
        st.warning("Please enter a subject name.")
    else:
        file_bytes = uploaded_file.read()
        size_mb = len(file_bytes) / (1024 * 1024)

        with st.spinner(
            f"Ingesting **{uploaded_file.name}** ({size_mb:.1f} MB)…"
        ):
            result, error = call_ingest_api(
                file_bytes=file_bytes,
                filename=uploaded_file.name,
                subject=subject.strip(),
            )

        if error:
            st.error(f"Ingestion failed: {error}")
        else:
            st.success(
                f"**{result.get('filename', uploaded_file.name)}** ingested "
                f"successfully into subject **{result.get('subject', subject)}**."
            )
            col1, col2 = st.columns(2)
            col1.metric(
                "Documents extracted",
                str(result.get("total_documents", "—")),
            )
            col2.metric(
                "Chunks indexed",
                str(result.get("total_chunks", "—")),
            )

# ── Info sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("ℹ️ How it works")
    st.markdown(
        """
        1. **Select** a PDF or TXT file from your computer.
        2. **Enter** a subject name to organise the content.
        3. **Click Ingest** — the backend will:
           - Extract and chunk the text.
           - Store the original file in MinIO.
           - Index all chunks into Qdrant.
        4. The document is now **searchable** from the Chat and Exam pages.
        """
    )
    st.divider()
    st.markdown(
        "**Supported formats:** PDF, TXT  \n"
        "**Max file size:** 50 MB  \n"
        "**Encoding:** UTF-8 for TXT files"
    )
    if _admin_password and st.session_state.get("ingestion_authenticated"):
        st.divider()
        if st.button("🔒 Lock page", use_container_width=True):
            st.session_state.ingestion_authenticated = False
            st.rerun()
