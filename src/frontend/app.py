"""Streamlit frontend entry point."""

from __future__ import annotations

import os

import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Oposiciones",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG Oposiciones")
st.markdown(
    f"""
    Sistema RAG para preparación de oposiciones.

    **API Backend:** `{API_URL}`

    Usa el menú lateral para navegar entre secciones:
    - **Chat RAG** — consulta el temario en lenguaje natural
    - **Generar Examen** — genera exámenes tipo test o desarrollo
    - **Evaluar Respuestas** — corrige y puntúa tus respuestas
    """
)
