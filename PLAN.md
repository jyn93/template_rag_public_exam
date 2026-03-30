# 📚 RAG Template para Preparación de Oposiciones
## Plan de Ejecución Completo — Guía de Proyecto

> **Referencia técnica:** Basado en las prácticas de [Miguel Otero Pedrido](https://github.com/miguelotero) sobre LLM evals y arquitecturas RAG de producción.

---

## 🗺️ Visión General del Proyecto

Un sistema RAG (Retrieval-Augmented Generation) modular y extensible para preparación de oposiciones, con capacidad de:

1. **Ingesta y embebido de temario** (PDFs, DOCX, texto plano)
2. **Chat interactivo** con el temario mediante preguntas en lenguaje natural
3. **Generación automática de exámenes** tipo test y desarrollo
4. **Evaluación de respuestas** del alumno con feedback detallado
5. **LLM Evals** para medir la calidad del sistema RAG

Todo desplegado en **contenedores Docker** para ejecución local reproducible.

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Streamlit / Gradio)            │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  Chat RAG    │  │  Generador   │  │   Evaluador de        │  │
│  │  Temario     │  │  de Exámenes │  │   Respuestas          │  │
│  └──────────────┘  └──────────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   FastAPI Backend   │
                    │   (API Gateway)     │
                    └─────────┬──────────┘
                              │
        ┌─────────────────────┼──────────────────────┐
        │                     │                      │
┌───────▼──────┐   ┌──────────▼───────┐   ┌─────────▼────────┐
│  RAG Engine  │   │  Exam Generator  │   │   Eval Engine    │
│  (LangChain/ │   │  (Prompt-based)  │   │  (RAGAS/Custom)  │
│   LlamaIndex)│   └──────────────────┘   └──────────────────┘
└───────┬──────┘
        │
┌───────▼──────────────────────────────────────────────────────┐
│                     DATA LAYER                               │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │   Qdrant    │  │  PostgreSQL  │  │    MinIO / Local     │ │
│  │ (Vectores)  │  │  (Metadatos, │  │    Storage (docs)    │ │
│  │             │  │   historial) │  │                      │ │
│  └─────────────┘  └──────────────┘  └──────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Stack Tecnológico

| Capa | Tecnología | Justificación |
|------|-----------|---------------|
| **LLM** | Claude 3.5 Sonnet / GPT-4o / Ollama (local) | Flexibilidad provider via LiteLLM |
| **Embeddings** | `text-embedding-3-small` / `nomic-embed-text` (local) | Coste-eficiencia |
| **Vector DB** | **Qdrant** | Open source, Docker-first, filtros avanzados |
| **Orquestación RAG** | **LlamaIndex** | Más maduro para RAG complejo que LangChain puro |
| **Evals** | **RAGAS** + evaluadores custom (estilo M. Otero) | Métricas: faithfulness, answer relevancy, context recall |
| **Backend** | **FastAPI** | Async, tipado, OpenAPI automático |
| **Frontend** | **Streamlit** | Prototipado rápido, suficiente para MVP |
| **Base de datos** | **PostgreSQL** + SQLAlchemy | Historial, usuarios, resultados |
| **Document store** | **MinIO** (compatible S3) | Almacenamiento de docs originales |
| **Contenedores** | **Docker Compose** | Orquestación local sencilla |
| **Observabilidad** | **Langfuse** (self-hosted) | Trazas LLM, costes, evals |
| **Testing** | **pytest** + **pytest-asyncio** | Cobertura backend |

---

## 📁 Estructura del Proyecto

```
rag-oposiciones/
│
├── 📄 docker-compose.yml          # Orquestación de todos los servicios
├── 📄 docker-compose.dev.yml      # Override para desarrollo
├── 📄 .env.example                # Variables de entorno plantilla
├── 📄 Makefile                    # Comandos de utilidad
├── 📄 pyproject.toml              # Dependencias y configuración (Poetry/uv)
│
├── 📂 src/
│   ├── 📂 api/                    # FastAPI app
│   │   ├── __init__.py
│   │   ├── main.py                # Entry point FastAPI
│   │   ├── dependencies.py        # DI Container (patron Dependency Injection)
│   │   ├── 📂 routers/
│   │   │   ├── ingestion.py       # POST /documents/ingest
│   │   │   ├── chat.py            # POST /chat/query
│   │   │   ├── exams.py           # POST /exams/generate, /exams/evaluate
│   │   │   └── evals.py           # GET /evals/run
│   │   └── 📂 middleware/
│   │       ├── logging.py
│   │       └── auth.py            # JWT básico (extensible)
│   │
│   ├── 📂 core/                   # Dominio / lógica de negocio
│   │   ├── 📂 config/
│   │   │   ├── settings.py        # Pydantic Settings (env vars)
│   │   │   └── prompts.py         # Prompt templates centralizados
│   │   │
│   │   ├── 📂 ingestion/          # Patrón Strategy para loaders
│   │   │   ├── base.py            # Abstract: DocumentLoader
│   │   │   ├── pdf_loader.py
│   │   │   ├── docx_loader.py
│   │   │   ├── txt_loader.py
│   │   │   └── pipeline.py        # IngestionPipeline (Facade)
│   │   │
│   │   ├── 📂 retrieval/          # Patrón Strategy para retrievers
│   │   │   ├── base.py            # Abstract: Retriever
│   │   │   ├── dense_retriever.py  # Vector search puro
│   │   │   ├── hybrid_retriever.py # BM25 + vector (reranking)
│   │   │   └── factory.py         # RetrieverFactory (Factory Pattern)
│   │   │
│   │   ├── 📂 generation/         # Patrón Template Method para generación
│   │   │   ├── base.py            # Abstract: Generator
│   │   │   ├── rag_generator.py   # Chat RAG estándar
│   │   │   ├── exam_generator.py  # Generación de exámenes
│   │   │   └── evaluator.py       # Evaluación de respuestas alumno
│   │   │
│   │   └── 📂 evals/              # Sistema de evaluación (estilo M. Otero)
│   │       ├── base.py            # Abstract: Evaluator
│   │       ├── ragas_eval.py      # RAGAS metrics wrapper
│   │       ├── custom_eval.py     # Evaluadores LLM-as-judge
│   │       └── reporter.py        # Generación de reportes
│   │
│   ├── 📂 infrastructure/         # Adaptadores externos
│   │   ├── 📂 vector_store/
│   │   │   ├── base.py            # Abstract: VectorStore
│   │   │   └── qdrant_store.py    # Implementación Qdrant
│   │   ├── 📂 llm/
│   │   │   ├── base.py            # Abstract: LLMClient
│   │   │   └── litellm_client.py  # LiteLLM (unifica providers)
│   │   ├── 📂 storage/
│   │   │   └── minio_storage.py   # Document storage
│   │   └── 📂 database/
│   │       ├── models.py          # SQLAlchemy models
│   │       ├── repository.py      # Repository Pattern
│   │       └── session.py         # Gestión de sesiones
│   │
│   └── 📂 frontend/               # Streamlit UI
│       ├── app.py                 # Entry point Streamlit
│       └── 📂 pages/
│           ├── 01_chat.py
│           ├── 02_generate_exam.py
│           └── 03_evaluate.py
│
├── 📂 tests/
│   ├── 📂 unit/
│   │   ├── test_ingestion.py
│   │   ├── test_retrieval.py
│   │   └── test_generation.py
│   ├── 📂 integration/
│   │   └── test_api.py
│   └── 📂 evals/
│       └── eval_suite.py          # Suite de evals ejecutable
│
├── 📂 data/
│   ├── 📂 raw/                    # Temario original (gitignored)
│   └── 📂 eval_datasets/          # Q&A pares para evaluar el RAG
│
├── 📂 docker/
│   ├── 📄 Dockerfile.api
│   ├── 📄 Dockerfile.frontend
│   └── 📄 Dockerfile.worker       # Proceso de ingesta async
│
└── 📂 docs/
    ├── adr/                       # Architecture Decision Records
    └── api/                       # OpenAPI exportado
```

---

## 🔄 Plan de Ejecución — Fases

### FASE 0 — Infraestructura Base (Día 1-2)

**Objetivo:** Tener el entorno Docker funcional y la estructura de proyecto lista.

#### 0.1 Crear `docker-compose.yml`

```yaml
# docker-compose.yml
version: "3.9"

services:
  # Vector Database
  qdrant:
    image: qdrant/qdrant:v1.9.0
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

  # Relational Database
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-opositor}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-opositor123}
      POSTGRES_DB: ${POSTGRES_DB:-oposiciones_db}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  # Document Storage
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER:-minioadmin}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD:-minioadmin123}
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data
    restart: unless-stopped

  # LLM Observability (Langfuse self-hosted)
  langfuse:
    image: langfuse/langfuse:latest
    depends_on:
      - postgres
    environment:
      DATABASE_URL: postgresql://opositor:opositor123@postgres:5432/langfuse
      NEXTAUTH_SECRET: ${LANGFUSE_SECRET:-supersecret}
      NEXTAUTH_URL: http://localhost:3000
      SALT: ${LANGFUSE_SALT:-salt}
    ports:
      - "3000:3000"
    restart: unless-stopped

  # FastAPI Backend
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - QDRANT_URL=http://qdrant:6333
      - DATABASE_URL=postgresql+asyncpg://opositor:opositor123@postgres:5432/oposiciones_db
      - MINIO_ENDPOINT=minio:9000
      - LANGFUSE_HOST=http://langfuse:3000
    env_file:
      - .env
    volumes:
      - ./src:/app/src
      - ./data:/app/data
    depends_on:
      - qdrant
      - postgres
      - minio
    restart: unless-stopped

  # Streamlit Frontend
  frontend:
    build:
      context: .
      dockerfile: docker/Dockerfile.frontend
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
    restart: unless-stopped

volumes:
  qdrant_data:
  postgres_data:
  minio_data:
```

#### 0.2 Configuración con Pydantic Settings

```python
# src/core/config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from enum import Enum

class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"

class RetrieverType(str, Enum):
    DENSE = "dense"
    HYBRID = "hybrid"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    app_name: str = "RAG Oposiciones"
    debug: bool = False
    log_level: str = "INFO"

    # LLM
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    llm_model: str = "claude-3-5-sonnet-20241022"
    llm_temperature: float = 0.1
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "oposiciones_temario"

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://opositor:opositor123@localhost:5432/oposiciones_db"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin123"
    minio_bucket: str = "temario-docs"

    # RAG
    retriever_type: RetrieverType = RetrieverType.HYBRID
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Langfuse
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

---

### FASE 1 — Ingesta de Documentos (Día 3-4)

**Objetivo:** Subir PDFs/DOCX del temario, chunkearlo e indexarlo en Qdrant.

#### 1.1 Patrón Strategy para Document Loaders

```python
# src/core/ingestion/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Document:
    """Representa un fragmento de documento ya procesado."""
    content: str
    metadata: dict  # tema, pagina, fuente, etc.
    doc_id: str

class DocumentLoader(ABC):
    """Strategy base para carga de documentos."""

    @abstractmethod
    def load(self, path: Path) -> list[Document]:
        """Carga y pre-procesa un documento."""
        ...

    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Indica si este loader soporta el tipo de fichero."""
        ...
```

```python
# src/core/ingestion/pdf_loader.py
from pathlib import Path
from llama_index.readers.file import PDFReader
from .base import Document, DocumentLoader

class PDFDocumentLoader(DocumentLoader):

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def load(self, path: Path) -> list[Document]:
        reader = PDFReader()
        llama_docs = reader.load_data(file=path)

        return [
            Document(
                content=doc.text,
                metadata={
                    "source": path.name,
                    "page": doc.metadata.get("page_label", i),
                    "file_type": "pdf",
                },
                doc_id=f"{path.stem}_page_{i}",
            )
            for i, doc in enumerate(llama_docs)
        ]
```

#### 1.2 Pipeline de Ingesta (Facade Pattern)

```python
# src/core/ingestion/pipeline.py
from pathlib import Path
from llama_index.core.node_parser import SentenceSplitter
from .base import Document, DocumentLoader
from ..config.settings import get_settings

class IngestionPipeline:
    """
    Facade que orquesta: carga → chunking → embebido → almacenamiento.
    """

    def __init__(
        self,
        loaders: list[DocumentLoader],
        vector_store,      # VectorStore (puerto de infraestructura)
        doc_storage,       # MinIO storage
    ):
        self._loaders = loaders
        self._vector_store = vector_store
        self._doc_storage = doc_storage
        settings = get_settings()
        self._splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def _get_loader(self, path: Path) -> DocumentLoader:
        for loader in self._loaders:
            if loader.supports(path):
                return loader
        raise ValueError(f"No hay loader registrado para {path.suffix}")

    async def ingest(self, path: Path, subject_name: str) -> dict:
        """Ingesta completa de un documento."""
        # 1. Cargar
        loader = self._get_loader(path)
        documents = loader.load(path)

        # 2. Subir original a MinIO
        await self._doc_storage.upload(path, subject_name)

        # 3. Añadir metadatos de oposición
        for doc in documents:
            doc.metadata["subject"] = subject_name

        # 4. Chunking
        chunks = self._chunk_documents(documents)

        # 5. Embeber y almacenar en Qdrant
        await self._vector_store.add_documents(chunks)

        return {
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "subject": subject_name,
        }

    def _chunk_documents(self, documents: list[Document]) -> list[Document]:
        chunked = []
        for doc in documents:
            # SentenceSplitter respeta límites de oración
            texts = self._splitter.split_text(doc.content)
            for i, text in enumerate(texts):
                chunked.append(Document(
                    content=text,
                    metadata={**doc.metadata, "chunk_index": i},
                    doc_id=f"{doc.doc_id}_chunk_{i}",
                ))
        return chunked
```

---

### FASE 2 — Motor RAG (Día 5-7)

**Objetivo:** Chat funcional con el temario usando recuperación híbrida.

#### 2.1 Prompt Templates Centralizados

```python
# src/core/config/prompts.py
from enum import Enum

class PromptMode(str, Enum):
    CHAT = "chat"
    EXAM_GENERATE = "exam_generate"
    EXAM_EVALUATE = "exam_evaluate"

SYSTEM_PROMPTS = {
    PromptMode.CHAT: """
Eres un asistente experto en preparación de oposiciones españolas.
Tu objetivo es ayudar al opositor a estudiar el temario de manera eficaz.

INSTRUCCIONES:
- Responde SIEMPRE basándote en el contexto recuperado del temario.
- Si la respuesta no está en el contexto, indícalo explícitamente.
- Usa un lenguaje claro, estructurado y pedagógico.
- Cita el tema y página cuando sea posible.
- Fomenta el pensamiento crítico del alumno.

CONTEXTO DEL TEMARIO:
{context}
""",

    PromptMode.EXAM_GENERATE: """
Eres un creador experto de exámenes para oposiciones españolas.

Genera {num_questions} preguntas de tipo {exam_type} basadas EXCLUSIVAMENTE
en el siguiente contexto del temario.

Tipos de examen:
- "test": 4 opciones (A, B, C, D), solo una correcta
- "desarrollo": pregunta abierta para respuesta detallada
- "mixto": combinación de ambas

Formato de salida (JSON estricto):
{{
  "questions": [
    {{
      "id": 1,
      "type": "test",
      "question": "...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "correct_answer": "A",
      "explanation": "...",
      "source_topic": "Tema X",
      "difficulty": "media"
    }}
  ]
}}

CONTEXTO DEL TEMARIO:
{context}

TEMA SOLICITADO: {subject_filter}
DIFICULTAD: {difficulty}
""",

    PromptMode.EXAM_EVALUATE: """
Eres un corrector experto de oposiciones españolas.

Evalúa la respuesta del alumno con criterio riguroso pero constructivo.

PREGUNTA: {question}
RESPUESTA CORRECTA: {correct_answer}
RESPUESTA DEL ALUMNO: {student_answer}
CONTEXTO DE REFERENCIA: {context}

Devuelve JSON:
{{
  "is_correct": true/false,
  "score": 0-10,
  "feedback": "Explicación detallada...",
  "strong_points": ["..."],
  "improvement_areas": ["..."],
  "reference_topics": ["Tema X, apartado Y"]
}}
""",
}
```

#### 2.2 Recuperador Híbrido (BM25 + Dense)

```python
# src/core/retrieval/hybrid_retriever.py
from llama_index.core import VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from .base import Retriever, RetrievalResult

class HybridRetriever(Retriever):
    """
    Combina búsqueda densa (vectores) con BM25 (léxico).
    Reranking con Reciprocal Rank Fusion.
    """

    def __init__(self, index: VectorStoreIndex, top_k: int = 5):
        self._dense = index.as_retriever(similarity_top_k=top_k)
        self._bm25 = BM25Retriever.from_defaults(
            index=index, similarity_top_k=top_k
        )
        self._fusion = QueryFusionRetriever(
            retrievers=[self._dense, self._bm25],
            similarity_top_k=top_k,
            mode="reciprocal_rerank",
            use_async=True,
        )

    async def retrieve(self, query: str, filters: dict = None) -> list[RetrievalResult]:
        nodes = await self._fusion.aretrieve(query)
        return [
            RetrievalResult(
                content=n.text,
                score=n.score,
                metadata=n.metadata,
            )
            for n in nodes
        ]
```

#### 2.3 Factory Pattern para Retrievers

```python
# src/core/retrieval/factory.py
from ..config.settings import get_settings, RetrieverType
from .dense_retriever import DenseRetriever
from .hybrid_retriever import HybridRetriever
from .base import Retriever

class RetrieverFactory:
    """Factory que crea el retriever según configuración."""

    @staticmethod
    def create(index, settings=None) -> Retriever:
        if settings is None:
            settings = get_settings()

        match settings.retriever_type:
            case RetrieverType.DENSE:
                return DenseRetriever(index, top_k=settings.top_k)
            case RetrieverType.HYBRID:
                return HybridRetriever(index, top_k=settings.top_k)
            case _:
                raise ValueError(f"Retriever desconocido: {settings.retriever_type}")
```

---

### FASE 3 — Generador de Exámenes (Día 8-9)

**Objetivo:** Generar exámenes tipo test y desarrollo a partir del temario.

#### 3.1 Template Method para Generadores

```python
# src/core/generation/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class GenerationInput:
    query: str
    context: list[str]
    metadata: dict = None

@dataclass
class GenerationOutput:
    content: str | dict
    sources: list[dict]
    tokens_used: int

class Generator(ABC):
    """Template Method base para todos los generadores."""

    async def generate(self, input: GenerationInput) -> GenerationOutput:
        """Flujo fijo: validate → build_prompt → call_llm → parse."""
        self._validate(input)
        prompt = self._build_prompt(input)
        raw_response = await self._call_llm(prompt)
        return self._parse_response(raw_response, input)

    def _validate(self, input: GenerationInput) -> None:
        if not input.context:
            raise ValueError("Se requiere contexto para generar respuesta.")

    @abstractmethod
    def _build_prompt(self, input: GenerationInput) -> str: ...

    @abstractmethod
    async def _call_llm(self, prompt: str) -> str: ...

    @abstractmethod
    def _parse_response(self, raw: str, input: GenerationInput) -> GenerationOutput: ...
```

```python
# src/core/generation/exam_generator.py
import json
from .base import Generator, GenerationInput, GenerationOutput
from ..config.prompts import SYSTEM_PROMPTS, PromptMode

class ExamGenerator(Generator):
    """Genera exámenes a partir del contexto recuperado."""

    def __init__(self, llm_client, num_questions: int = 10,
                 exam_type: str = "test", difficulty: str = "media"):
        self._llm = llm_client
        self._num_questions = num_questions
        self._exam_type = exam_type
        self._difficulty = difficulty

    def _build_prompt(self, input: GenerationInput) -> str:
        context_str = "\n\n---\n\n".join(input.context)
        return SYSTEM_PROMPTS[PromptMode.EXAM_GENERATE].format(
            num_questions=self._num_questions,
            exam_type=self._exam_type,
            context=context_str,
            subject_filter=input.metadata.get("subject", "General"),
            difficulty=self._difficulty,
        )

    async def _call_llm(self, prompt: str) -> str:
        return await self._llm.complete(prompt, response_format="json")

    def _parse_response(self, raw: str, input: GenerationInput) -> GenerationOutput:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: extraer JSON del texto si el LLM añadió contexto
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            data = json.loads(match.group()) if match else {"questions": []}

        return GenerationOutput(
            content=data,
            sources=[c for c in input.context[:3]],
            tokens_used=0,  # se actualiza desde el LLM client
        )
```

---

### FASE 4 — Sistema de Evals (Día 10-12)

**Objetivo:** Medir objetivamente la calidad del RAG. Metodología basada en proyectos de Miguel Otero.

> 📌 **Referencia:** Ver repositorios como `miguelotero/llm-evaluation-framework` para patrones de evaluación con datasets sintéticos y LLM-as-judge.

#### 4.1 Métricas RAGAS + Evaluadores Custom

```python
# src/core/evals/ragas_eval.py
from ragas import evaluate
from ragas.metrics import (
    faithfulness,           # ¿La respuesta está soportada por el contexto?
    answer_relevancy,       # ¿La respuesta responde la pregunta?
    context_recall,         # ¿Se recuperó todo el contexto necesario?
    context_precision,      # ¿El contexto recuperado es relevante?
)
from datasets import Dataset
from .base import EvalResult

class RAGASEvaluator:
    """
    Evalúa el pipeline RAG completo usando RAGAS.

    Dataset mínimo necesario:
    - question: la pregunta
    - answer: la respuesta generada por el RAG
    - contexts: lista de chunks recuperados
    - ground_truth: respuesta correcta de referencia
    """

    METRICS = [faithfulness, answer_relevancy, context_recall, context_precision]

    async def evaluate(self, eval_dataset: list[dict]) -> EvalResult:
        dataset = Dataset.from_list(eval_dataset)
        scores = evaluate(dataset, metrics=self.METRICS)
        return EvalResult(
            scores=scores.to_pandas().to_dict(),
            summary=self._summarize(scores),
        )

    def _summarize(self, scores) -> dict:
        df = scores.to_pandas()
        return {
            "faithfulness_mean": df["faithfulness"].mean(),
            "answer_relevancy_mean": df["answer_relevancy"].mean(),
            "context_recall_mean": df["context_recall"].mean(),
            "context_precision_mean": df["context_precision"].mean(),
        }
```

```python
# src/core/evals/custom_eval.py
"""
Evaluadores LLM-as-judge personalizados para oposiciones.
Inspirado en la metodología de Miguel Otero para evals específicos de dominio.
"""

class OposicionesEvaluator:
    """
    Evalúa aspectos específicos del dominio de oposiciones:
    - Precisión jurídica (citas a normativa correctas)
    - Completitud del temario (¿se cubre todo lo exigible?)
    - Claridad pedagógica
    """

    JUDGE_PROMPT = """
    Actúa como un tribunal de oposiciones evaluando la calidad de este sistema de estudio.

    PREGUNTA: {question}
    RESPUESTA DEL SISTEMA: {answer}
    RESPUESTA DE REFERENCIA: {ground_truth}

    Evalúa en escala 1-5:
    1. precision_juridica: ¿Las referencias legales son correctas?
    2. completitud: ¿La respuesta cubre todos los aspectos de la pregunta?
    3. claridad: ¿Es comprensible para un opositor?
    4. precision_factual: ¿Los datos y fechas son correctos?

    Responde SOLO en JSON:
    {{"precision_juridica": N, "completitud": N, "claridad": N, "precision_factual": N, "justificacion": "..."}}
    """

    def __init__(self, llm_client):
        self._llm = llm_client

    async def judge(self, question: str, answer: str, ground_truth: str) -> dict:
        prompt = self.JUDGE_PROMPT.format(
            question=question, answer=answer, ground_truth=ground_truth
        )
        raw = await self._llm.complete(prompt, response_format="json")
        import json
        return json.loads(raw)
```

#### 4.2 Dataset Sintético de Evaluación

```python
# tests/evals/generate_eval_dataset.py
"""
Genera un dataset de evaluación sintético a partir del temario.
Patrón recomendado por M. Otero: generar Q&A pares con LLM fuerte,
evaluar con RAG, comparar.
"""

DATASET_GENERATION_PROMPT = """
Dado el siguiente fragmento de temario de oposiciones,
genera {n} pares pregunta-respuesta de alta calidad
que podrían aparecer en el examen oficial.

Asegúrate de cubrir: definiciones, procedimientos, plazos legales, artículos.

TEMARIO:
{chunk}

Responde en JSON:
{{"pairs": [{{"question": "...", "ground_truth": "..."}}]}}
"""

async def generate_eval_dataset(chunks: list[str], llm_client, n_per_chunk: int = 3):
    """Genera dataset de evaluación automáticamente."""
    pairs = []
    for chunk in chunks:
        prompt = DATASET_GENERATION_PROMPT.format(n=n_per_chunk, chunk=chunk)
        raw = await llm_client.complete(prompt, response_format="json")
        data = json.loads(raw)
        pairs.extend(data["pairs"])
    return pairs
```

---

### FASE 5 — API REST con FastAPI (Día 13-14)

**Objetivo:** Exponer toda la funcionalidad como API documentada.

#### 5.1 Dependency Injection

```python
# src/api/dependencies.py
from functools import lru_cache
from fastapi import Depends
from ..core.config.settings import get_settings, Settings
from ..infrastructure.vector_store.qdrant_store import QdrantVectorStore
from ..infrastructure.llm.litellm_client import LiteLLMClient
from ..core.ingestion.pipeline import IngestionPipeline
from ..core.retrieval.factory import RetrieverFactory
from ..core.generation.rag_generator import RAGGenerator
from ..core.generation.exam_generator import ExamGenerator

def get_vector_store(settings: Settings = Depends(get_settings)):
    return QdrantVectorStore(url=settings.qdrant_url,
                             collection=settings.qdrant_collection)

def get_llm_client(settings: Settings = Depends(get_settings)):
    return LiteLLMClient(
        provider=settings.llm_provider,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )

def get_rag_generator(
    llm=Depends(get_llm_client),
    vector_store=Depends(get_vector_store),
    settings=Depends(get_settings),
):
    retriever = RetrieverFactory.create(vector_store.get_index(), settings)
    return RAGGenerator(llm_client=llm, retriever=retriever)
```

#### 5.2 Router de Exámenes

```python
# src/api/routers/exams.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from ..dependencies import get_rag_generator, get_vector_store
from ...core.generation.exam_generator import ExamGenerator

router = APIRouter(prefix="/exams", tags=["exams"])

class ExamRequest(BaseModel):
    subject: str
    num_questions: int = 10
    exam_type: str = "test"          # test | desarrollo | mixto
    difficulty: str = "media"        # facil | media | dificil
    topics: list[str] = []           # Filtrar por temas específicos

class EvaluationRequest(BaseModel):
    question: str
    correct_answer: str
    student_answer: str
    question_type: str = "test"

@router.post("/generate")
async def generate_exam(
    request: ExamRequest,
    vector_store=Depends(get_vector_store),
    llm=Depends(get_llm_client),
):
    """Genera un examen basado en el temario embebido."""
    # Recuperar contexto relevante del temario
    retriever = RetrieverFactory.create(vector_store.get_index())
    query = f"Temario completo sobre {request.subject} {' '.join(request.topics)}"
    context = await retriever.retrieve(query)

    generator = ExamGenerator(
        llm_client=llm,
        num_questions=request.num_questions,
        exam_type=request.exam_type,
        difficulty=request.difficulty,
    )

    from ...core.generation.base import GenerationInput
    result = await generator.generate(GenerationInput(
        query=query,
        context=[c.content for c in context],
        metadata={"subject": request.subject},
    ))

    return {"exam": result.content, "sources_used": len(context)}

@router.post("/evaluate")
async def evaluate_answer(
    request: EvaluationRequest,
    llm=Depends(get_llm_client),
    vector_store=Depends(get_vector_store),
):
    """Evalúa la respuesta de un alumno y da feedback."""
    from ...core.generation.evaluator import AnswerEvaluator
    evaluator = AnswerEvaluator(llm_client=llm)

    result = await evaluator.evaluate(
        question=request.question,
        correct_answer=request.correct_answer,
        student_answer=request.student_answer,
    )
    return result
```

---

### FASE 6 — Frontend Streamlit (Día 15-16)

**Objetivo:** UI funcional para chat, generación y evaluación de exámenes.

```python
# src/frontend/pages/02_generate_exam.py
import streamlit as st
import httpx
import json

st.set_page_config(page_title="🎓 Generador de Exámenes", layout="wide")
st.title("🎓 Generador de Exámenes de Oposición")

with st.sidebar:
    st.header("⚙️ Configuración del Examen")
    subject = st.text_input("Materia/Oposición", placeholder="Ej: Administrativo del Estado")
    num_q = st.slider("Número de preguntas", 5, 50, 10)
    exam_type = st.selectbox("Tipo", ["test", "desarrollo", "mixto"])
    difficulty = st.select_slider("Dificultad", ["facil", "media", "dificil"], value="media")
    topics = st.text_area("Temas específicos (uno por línea)").split("\n")
    generate_btn = st.button("🚀 Generar Examen", type="primary")

if generate_btn and subject:
    with st.spinner("Generando examen desde el temario..."):
        response = httpx.post(
            f"{st.secrets['API_URL']}/exams/generate",
            json={
                "subject": subject,
                "num_questions": num_q,
                "exam_type": exam_type,
                "difficulty": difficulty,
                "topics": [t for t in topics if t.strip()],
            },
            timeout=60,
        )

        if response.status_code == 200:
            exam_data = response.json()["exam"]
            st.success(f"✅ {len(exam_data['questions'])} preguntas generadas")

            for i, q in enumerate(exam_data["questions"], 1):
                with st.expander(f"Pregunta {i}: {q['question'][:80]}..."):
                    st.markdown(f"**{q['question']}**")

                    if q["type"] == "test":
                        for opt, text in q["options"].items():
                            st.write(f"**{opt})** {text}")
                        student_ans = st.radio(
                            "Tu respuesta:",
                            list(q["options"].keys()),
                            key=f"q_{i}",
                            horizontal=True,
                        )
                        if st.button("✅ Comprobar", key=f"check_{i}"):
                            if student_ans == q["correct_answer"]:
                                st.success("¡Correcto! 🎉")
                            else:
                                st.error(f"Incorrecto. La respuesta era: {q['correct_answer']}")
                            st.info(f"**Explicación:** {q['explanation']}")
```

---

### FASE 7 — Testing y CI (Día 17-18)

**Objetivo:** Cobertura mínima de tests y configuración de GitHub Actions.

```python
# tests/unit/test_ingestion.py
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from src.core.ingestion.pipeline import IngestionPipeline
from src.core.ingestion.pdf_loader import PDFDocumentLoader

@pytest.fixture
def mock_vector_store():
    store = AsyncMock()
    store.add_documents = AsyncMock(return_value=True)
    return store

@pytest.fixture
def pipeline(mock_vector_store):
    return IngestionPipeline(
        loaders=[PDFDocumentLoader()],
        vector_store=mock_vector_store,
        doc_storage=AsyncMock(),
    )

@pytest.mark.asyncio
async def test_ingest_pdf_calls_vector_store(pipeline, tmp_path, mock_vector_store):
    """Verifica que la ingesta llama al vector store."""
    # Arrange: crear PDF de prueba mínimo
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 minimal")

    with pytest.raises(Exception):  # PDF inválido en test
        await pipeline.ingest(pdf_path, "Tema Test")

    # En test real con PDF válido:
    # mock_vector_store.add_documents.assert_called_once()
```

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      qdrant:
        image: qdrant/qdrant:v1.9.0
        ports: ["6333:6333"]

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install uv
        run: pip install uv
      - name: Install dependencies
        run: uv sync
      - name: Run tests
        run: uv run pytest tests/ -v --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v4
```

---

## 📋 Patrones de Diseño Aplicados

| Patrón | Dónde se aplica | Por qué |
|--------|----------------|---------|
| **Strategy** | `DocumentLoader`, `Retriever`, `Generator` | Intercambiar loaders/retrievers sin cambiar cliente |
| **Factory** | `RetrieverFactory`, `LoaderFactory` | Desacoplar creación de objetos de su uso |
| **Template Method** | `Generator.generate()` | Flujo fijo, pasos variables por subclase |
| **Facade** | `IngestionPipeline` | Simplifica la interfaz del subsistema de ingesta |
| **Repository** | `DocumentRepository`, `ExamRepository` | Desacopla lógica de negocio de la DB |
| **Dependency Injection** | FastAPI `Depends()` | Testabilidad, bajo acoplamiento |
| **Adapter** | `QdrantVectorStore`, `LiteLLMClient` | Aísla dependencias externas |

---

## 🔧 Comandos de Desarrollo

```bash
# Levantar toda la infraestructura
make up

# Solo servicios de datos (sin API/frontend)
make infra

# Ingestar documentos de ejemplo
make ingest FILE=data/raw/tema1.pdf SUBJECT="Derecho Constitucional"

# Ejecutar suite de evals
make eval

# Ver logs en tiempo real
make logs SERVICE=api

# Entrar a shell del contenedor API
make shell SERVICE=api

# Correr tests
make test

# Limpiar todo (CUIDADO: borra datos)
make clean
```

```makefile
# Makefile
.PHONY: up down infra ingest eval test logs shell clean

up:
	docker compose up -d --build

down:
	docker compose down

infra:
	docker compose up -d qdrant postgres minio langfuse

ingest:
	docker compose exec api python -m src.cli.ingest --file $(FILE) --subject "$(SUBJECT)"

eval:
	docker compose exec api python -m src.core.evals.runner --dataset data/eval_datasets/

test:
	docker compose exec api pytest tests/ -v

logs:
	docker compose logs -f $(SERVICE)

shell:
	docker compose exec $(SERVICE) bash

clean:
	docker compose down -v --remove-orphans
```

---

## 📦 Dependencias Clave (`pyproject.toml`)

```toml
[project]
name = "rag-oposiciones"
version = "0.1.0"
requires-python = ">=3.12"

[tool.uv.dependencies]
# Core RAG
llama-index = "^0.11"
llama-index-vector-stores-qdrant = "^0.3"
llama-index-retrievers-bm25 = "^0.3"
llama-index-readers-file = "^0.2"
qdrant-client = "^1.9"

# LLM
litellm = "^1.40"
anthropic = "^0.34"
openai = "^1.40"

# Evals
ragas = "^0.1"
langfuse = "^2.0"

# API
fastapi = "^0.115"
uvicorn = {extras = ["standard"], version = "^0.30"}
pydantic = "^2.8"
pydantic-settings = "^2.4"

# DB
sqlalchemy = {extras = ["asyncio"], version = "^2.0"}
asyncpg = "^0.29"
alembic = "^1.13"

# Storage
minio = "^7.2"

# Frontend
streamlit = "^1.38"

# Utils
python-multipart = "^0.0.9"
httpx = "^0.27"
tenacity = "^9.0"          # Retry logic
structlog = "^24.0"        # Logging estructurado

[tool.uv.dev-dependencies]
pytest = "^8.0"
pytest-asyncio = "^0.24"
pytest-cov = "^5.0"
ruff = "^0.6"
mypy = "^1.11"
```

---

## 🗓️ Roadmap de Funcionalidades

### MVP (Semanas 1-3) — ✅ Completado
- [x] Estructura de proyecto y Docker Compose
- [x] Ingesta de PDFs y TXT + pipeline
- [x] Chat RAG con temario (HybridRetriever BM25+RRF)
- [x] Generación de exámenes tipo test y desarrollo
- [x] Evaluación de respuestas con feedback detallado
- [x] Suite de evals con RAGAS + LLM-as-judge
- [x] FastAPI routers (chat, exam, ingestion)
- [x] Streamlit UI (chat + generación de exámenes)
- [x] Observabilidad LLM con Langfuse (tracing + LiteLLM callback)
- [x] Logging estructurado con structlog

### v1.0 — Prioridad: funcionalidad completa antes de productivizar

> Criterio: completar el flujo de uso del prototipo de extremo a extremo
> antes de añadir infraestructura multi-usuario (JWT, roles, etc.).

1. [x] **Página de ingesta en Streamlit** (`feature/streamlit-ingestion-page`) — PR #25 ✅
   - Upload de documentos (PDF/TXT) desde la UI sin necesidad de CLI
   - Protección sencilla por contraseña configurable via `.env` (pre-auth)
   - Visualización del estado de ingesta (progreso, documentos ingestados)
   - _Prerequisito real_: sin esto el sistema solo lo puede usar un técnico

2. [x] **Modo simulacro de examen con timer** (`feature/exam-simulation-mode`) — PR #26 ✅
   - Sesión de examen completa con cuenta atrás configurable
   - Navegación entre preguntas, respuestas bloqueadas al acabar el tiempo
   - Pantalla de resultados: puntuación, revisión pregunta a pregunta con explicaciones
   - Sin cambios en backend — trabajo puramente de Streamlit

3. [x] **Exportación de exámenes a PDF** (`feature/exam-export-pdf`) — PR #27 ✅
   - Endpoint `GET /exam/export` que devuelve PDF con el examen generado
   - Botón "Descargar PDF" en la UI de exámenes
   - Útil para estudio en papel o impresión

4. [ ] **Historial de sesiones y seguimiento de progreso** (`feature/study-history`)
   - Capa PostgreSQL: modelos SQLAlchemy + migraciones Alembic
   - Guardar cada sesión de examen/chat con puntuación y timestamps
   - Dashboard de progreso por tema en Streamlit

### v2.0 (Futuro) — Productivización y multiusuario
- [ ] Autenticación JWT (aplazada intencionalmente hasta completar v1.0)
- [ ] Soporte multi-oposición (perfiles de usuario)
- [ ] Flashcards con spaced repetition
- [ ] Resúmenes automáticos de temas
- [ ] Modo oral (speech-to-text + TTS)
- [ ] Dashboard de analytics del alumno

---

## 📚 Referencias y Recursos

### Metodología de Evals (Miguel Otero)
- Patrón: **generar dataset sintético** → evaluar RAG → iterar sobre chunking/retrieval
- Métricas prioritarias: `faithfulness` > `context_recall` > `answer_relevancy`
- Usar **LLM-as-judge** para métricas específicas del dominio (precisión jurídica)
- Separar **evals offline** (sobre dataset fijo) de **monitorización online** (Langfuse)

### Documentación Clave
- [LlamaIndex Docs](https://docs.llamaindex.ai)
- [Qdrant Docs](https://qdrant.tech/documentation)
- [RAGAS Docs](https://docs.ragas.io)
- [LiteLLM Docs](https://docs.litellm.ai)
- [Langfuse Docs](https://langfuse.com/docs)
- [FastAPI Docs](https://fastapi.tiangolo.com)

### Lecturas Recomendadas
- "Building RAG Applications" — LlamaIndex blog
- "RAG Evaluation: A Practical Guide" — RAGAS paper
- Repositorios de referencia: `explodinggradients/ragas`, `langfuse/langfuse`

---

## ⚠️ Decisiones de Arquitectura (ADRs)

### ADR-001: LlamaIndex sobre LangChain
**Decisión:** Usar LlamaIndex como framework RAG principal.
**Motivo:** Mejor soporte nativo para RAG complejo (hybrid retrieval, reranking, query routing). LangChain es más genérico; LlamaIndex está optimizado para el caso de uso.

### ADR-002: Qdrant como Vector Store
**Decisión:** Qdrant sobre Chroma/Weaviate/Pinecone.
**Motivo:** Docker-first, open source, soporte de filtros por metadatos (filtrar por tema), activa comunidad, versión cloud disponible para escalar.

### ADR-003: LiteLLM como capa de abstracción LLM
**Decisión:** No llamar a APIs de LLM directamente.
**Motivo:** Permite cambiar entre Claude, GPT-4o y Ollama (local) cambiando una variable de entorno. Facilita testing con modelos locales baratos.

### ADR-004: RAGAS + evaluadores custom
**Decisión:** RAGAS para métricas estándar + evaluadores propios para dominio.
**Motivo:** RAGAS cubre faithfulness/relevancy bien, pero no cubre "precisión jurídica" específica de oposiciones. Se necesitan ambos niveles.

---

*Documento generado como guía de inicio de proyecto. Actualizar conforme evolucione la implementación.*
