# RAG Oposiciones

Sistema RAG (Retrieval-Augmented Generation) modular y extensible para preparación de oposiciones. Permite ingestar temario en PDF o TXT, hacer preguntas en lenguaje natural, generar exámenes y evaluar respuestas con retroalimentación del LLM.

---

## Índice

- [Arquitectura](#arquitectura)
- [Quickstart — despliegue local](#quickstart--despliegue-local)
- [Pipelines y guías de uso](#pipelines-y-guías-de-uso)
  - [Pipeline de ingesta](#1-pipeline-de-ingesta)
  - [Pipeline de chat RAG](#2-pipeline-de-chat-rag)
  - [Pipeline de generación de examen](#3-pipeline-de-generación-de-examen)
  - [Pipeline de evaluación de respuestas](#4-pipeline-de-evaluación-de-respuestas)
  - [Pipeline de evaluación de calidad (RAGAS)](#5-pipeline-de-evaluación-de-calidad-ragas)
- [Variables de entorno](#variables-de-entorno)
- [Comandos de desarrollo](#comandos-de-desarrollo)
- [Estructura del proyecto](#estructura-del-proyecto)

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Streamlit :8501)               │
│   Página Chat  │  Página Examen  │  Evaluación de respuestas    │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP (httpx)
┌─────────────────────────▼───────────────────────────────────────┐
│                      API REST (FastAPI :8000)                   │
│   POST /ingest  │  POST /chat  │  POST /exam/generate           │
│                              │  POST /exam/evaluate             │
└──────┬───────────────────────┬────────────────────────┬─────────┘
       │                       │                        │
┌──────▼──────┐   ┌────────────▼──────────┐  ┌─────────▼────────┐
│   MinIO     │   │   Qdrant              │  │   LLM (LiteLLM)  │
│  :9000/9001 │   │  Vector DB  :6333     │  │  Anthropic/OpenAI│
│  (docs PDF) │   │  (embeddings)         │  │  /Ollama/Groq    │
└─────────────┘   └───────────────────────┘  └──────────────────┘
                                                        │
                                              ┌─────────▼────────┐
                                              │    Langfuse       │
                                              │  Observabilidad   │
                                              │    :3000          │
                                              └──────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│               PostgreSQL :5432  (Langfuse + metadata)           │
└─────────────────────────────────────────────────────────────────┘
```

**Capas del sistema:**

| Capa | Módulo | Responsabilidad |
|------|--------|-----------------|
| `src/core/ingestion` | `IngestionPipeline` | Carga, chunking e indexación de documentos |
| `src/core/retrieval` | `DenseRetriever`, `HybridRetriever` | Búsqueda vectorial y BM25+RRF |
| `src/core/generation` | `RAGGenerator`, `ExamGenerator`, `AnswerEvaluator` | Generación con LLM |
| `src/core/evals` | `RAGASEvaluator`, `OposicionesEvaluator` | Métricas de calidad RAG |
| `src/infrastructure` | `QdrantVectorStore`, `MinIOStorage`, `LiteLLMClient` | Adaptadores de infraestructura |
| `src/api` | Routers FastAPI | API REST + inyección de dependencias |
| `src/frontend` | Páginas Streamlit | Interfaz de usuario |

---

## Quickstart — despliegue local

### Requisitos previos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) ≥ 24
- [uv](https://docs.astral.sh/uv/getting-started/installation/) ≥ 0.5 (gestor de paquetes Python)
- Python 3.12+
- Clave API de Anthropic **o** OpenAI (para el LLM y los embeddings)

### Paso 1 — Clonar y configurar el entorno

```bash
git clone https://github.com/jyn93/template_rag_public_exam.git
cd template_rag_public_exam

# Instalar dependencias Python en entorno virtual
uv sync
```

### Paso 2 — Crear el fichero `.env`

```bash
cp .env.example .env
```

Edita `.env` y rellena al menos estas variables:

```dotenv
# Proveedor LLM (anthropic | openai | ollama)
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=sk-ant-...      # reemplaza con tu clave

# Embeddings (OpenAI siempre, independientemente del LLM)
OPENAI_API_KEY=sk-...             # necesario para text-embedding-3-small
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
```

El resto de variables tienen valores por defecto válidos para desarrollo local. Consulta la [sección de variables de entorno](#variables-de-entorno) para la referencia completa.

### Paso 3 — Levantar la infraestructura

```bash
# Levanta todos los servicios: Qdrant, PostgreSQL, MinIO, Langfuse, API y frontend
make up
```

Espera a que todos los contenedores estén `healthy` (~30 segundos):

```bash
docker compose ps
```

### Paso 4 — Verificar que todo está en marcha

| Servicio | URL | Descripción |
|----------|-----|-------------|
| API REST | http://localhost:8000/docs | Swagger UI interactivo |
| API health | http://localhost:8000/health | Estado del backend |
| Frontend | http://localhost:8501 | Interfaz Streamlit |
| Qdrant dashboard | http://localhost:6333/dashboard | Explorador de colecciones |
| MinIO console | http://localhost:9001 | Gestión de objetos (user: `minioadmin`, pass: `minioadmin123`) |
| Langfuse | http://localhost:3000 | Observabilidad de LLM (crear cuenta en el primer acceso) |

```bash
# Comprobación rápida desde terminal
curl http://localhost:8000/health
# → {"status": "ok", "service": "RAG Oposiciones", "uptime_seconds": ...}
```

### Paso 5 — Ingestar tu primer documento

```bash
make ingest FILE=docs/temario_admin.pdf SUBJECT="Derecho Administrativo"
```

Ya puedes abrir http://localhost:8501 y hacer preguntas sobre el documento.

---

## Pipelines y guías de uso

### 1. Pipeline de ingesta

**Qué hace:** carga un fichero PDF o TXT, lo divide en chunks, genera embeddings con OpenAI, sube el fichero original a MinIO y crea el índice vectorial en Qdrant.

```
Fichero (PDF/TXT)
      │
      ▼
 PDFLoader / TxtLoader          → extrae texto + metadatos
      │
      ▼
 Chunker (LlamaIndex)           → chunks de 512 tokens, overlap 64
      │
      ▼
 MinIOStorage.upload()          → almacena fichero original en S3
      │
      ▼
 QdrantVectorStore.add_docs()   → genera embeddings + indexa en Qdrant
```

#### Vía CLI (Makefile)

```bash
# Sintaxis
make ingest FILE=<ruta_al_fichero> SUBJECT="<nombre del tema>"

# Ejemplos
make ingest FILE=temario/tema1_procedimiento.pdf SUBJECT="Procedimiento Administrativo"
make ingest FILE=temario/codigo_civil.txt SUBJECT="Derecho Civil"
```

#### Vía API REST

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@temario/tema1.pdf;type=application/pdf" \
  -F "subject=Procedimiento Administrativo"
```

Respuesta `202 Accepted`:

```json
{
  "total_documents": 1,
  "total_chunks": 47,
  "subject": "Procedimiento Administrativo",
  "filename": "tema1.pdf"
}
```

#### Vía Swagger UI

Abre http://localhost:8000/docs → sección **ingestion** → `POST /ingest` → "Try it out".

**Restricciones:**
- Formatos aceptados: `.pdf`, `.txt`
- Tamaño máximo: 50 MB por fichero
- El campo `subject` no puede estar vacío

**Errores comunes:**

| Código | Causa | Solución |
|--------|-------|----------|
| `422` | Extensión no soportada o subject vacío | Usa `.pdf` o `.txt`; rellena el subject |
| `413` | Fichero > 50 MB | Divide el fichero antes de ingestar |
| `500` | Qdrant o MinIO no disponibles | Comprueba `docker compose ps` |

---

### 2. Pipeline de chat RAG

**Qué hace:** recibe una pregunta en lenguaje natural, recupera los chunks más relevantes del temario y genera una respuesta fundamentada con el LLM.

```
Pregunta del usuario
      │
      ▼
 HybridRetriever                → BM25 + búsqueda densa, fusión RRF
      │                           top_k chunks (por defecto 5)
      ▼
 GenerationInput                → query + contexto + filtro subject
      │
      ▼
 RAGGenerator (LiteLLM)         → prompt estructurado → respuesta
      │
      ▼
 ChatResponse                   → answer + sources + tokens_used
```

#### Vía frontend (Streamlit)

1. Abre http://localhost:8501
2. Navega a **Chat RAG** en el menú lateral
3. (Opcional) Escribe un filtro de subject en el sidebar para acotar la búsqueda
4. Ajusta el slider **top-k** (chunks recuperados, 1–20)
5. Escribe tu pregunta en el cuadro inferior y pulsa Enter

#### Vía API REST

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "¿Cuáles son los plazos del recurso de alzada?",
    "subject": "Procedimiento Administrativo",
    "top_k": 5
  }'
```

Respuesta `200 OK`:

```json
{
  "answer": "El recurso de alzada debe interponerse en el plazo de un mes...",
  "sources": [
    "Artículo 121 de la Ley 39/2015... (chunk 1)",
    "El plazo de resolución es de tres meses... (chunk 2)"
  ],
  "tokens_used": 1240
}
```

**Parámetros de la petición:**

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `query` | `string` | ✅ | Pregunta en lenguaje natural (mínimo 1 carácter) |
| `subject` | `string` | ❌ | Filtro de tema (vacío = busca en todo el temario) |
| `top_k` | `int` | ❌ | Chunks a recuperar, 1–20 (por defecto: 5) |

**Errores comunes:**

| Código | Causa |
|--------|-------|
| `422` | `query` vacío, o no se encontró contexto relevante |
| `500` | Qdrant inaccesible o error del LLM |

---

### 3. Pipeline de generación de examen

**Qué hace:** recupera contexto sobre un tema y solicita al LLM que genere preguntas de examen tipo test, desarrollo o mixto con la dificultad indicada.

```
Tema / query
      │
      ▼
 HybridRetriever                → recupera chunks relevantes
      │
      ▼
 ExamGenerator (LiteLLM)        → prompt con tipo y dificultad
      │                           parseo incremental de JSON
      ▼
 ExamGenerateResponse           → exam dict + sources
```

#### Vía frontend (Streamlit)

1. Abre http://localhost:8501 → **Generar Examen**
2. Configura en el sidebar:
   - **Subject filter**: tema opcional
   - **Exam type**: `test` (tipo test), `desarrollo` (respuesta abierta), `mixto`
   - **Difficulty**: `facil`, `media`, `dificil`
   - **Number of questions**: 1–20
   - **Context chunks (top-k)**: fragmentos de contexto a recuperar
3. Escribe el concepto o tema a examinar y pulsa **Generate Exam**
4. Responde las preguntas (radio buttons para tipo test, textarea para desarrollo)
5. Pulsa **Evaluate My Answers** para obtener puntuación y retroalimentación

#### Vía API REST

```bash
curl -X POST http://localhost:8000/exam/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "recurso de alzada",
    "subject": "Procedimiento Administrativo",
    "num_questions": 5,
    "exam_type": "test",
    "difficulty": "media",
    "top_k": 8
  }'
```

Respuesta `200 OK`:

```json
{
  "exam": {
    "questions": [
      {
        "id": 1,
        "type": "test",
        "question": "¿Ante qué órgano se interpone el recurso de alzada?",
        "options": ["A) El mismo órgano", "B) El superior jerárquico", "C) Los tribunales", "D) El Defensor del Pueblo"],
        "correct_answer": "B) El superior jerárquico"
      }
    ]
  },
  "sources": ["Artículo 121 Ley 39/2015..."]
}
```

**Parámetros de la petición:**

| Campo | Tipo | Requerido | Valores válidos |
|-------|------|-----------|-----------------|
| `query` | `string` | ✅ | Tema o concepto a examinar |
| `subject` | `string` | ❌ | Filtro de materia |
| `num_questions` | `int` | ❌ | 1–20 (por defecto: 5) |
| `exam_type` | `string` | ❌ | `test`, `desarrollo`, `mixto` |
| `difficulty` | `string` | ❌ | `facil`, `media`, `dificil` |
| `top_k` | `int` | ❌ | 1–20 (por defecto: 5) |

**Errores comunes:**

| Código | Causa | Solución |
|--------|-------|----------|
| `422` | `query` vacío, no se encontró contexto relevante, o contexto vacío al generar | Verifica que el tema tiene temario ingestado |
| `500` | Qdrant inaccesible o error del LLM durante la generación | Comprueba `docker compose ps` y la clave del LLM |

---

### 4. Pipeline de evaluación de respuestas

**Qué hace:** dado un par (pregunta, respuesta correcta), puntúa la respuesta del alumno del 0 al 10 con retroalimentación detallada.

```
Pregunta + respuesta correcta + respuesta del alumno
      │
      ▼
 AnswerEvaluator (LiteLLM)      → prompt de evaluación estructurada
      │
      ▼
 EvaluationResult               → score + is_correct + feedback
                                   + missing_points + strengths
```

#### Vía frontend (Streamlit)

La evaluación está integrada en la página de examen (ver paso 5 del [pipeline de generación](#3-pipeline-de-generación-de-examen)). Tras responder las preguntas, el botón **Evaluate My Answers** llama a este endpoint por cada pregunta respondida y muestra los resultados inline.

#### Vía API REST

```bash
curl -X POST http://localhost:8000/exam/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Cuál es el plazo para interponer el recurso de alzada?",
    "correct_answer": "Un mes si el acto es expreso; tres meses si es presunto.",
    "student_answer": "El plazo es de un mes desde la notificación."
  }'
```

Respuesta `200 OK`:

```json
{
  "score": 7.0,
  "is_correct": true,
  "feedback": "Correcta la parte del acto expreso, pero no mencionas el plazo para actos presuntos.",
  "missing_points": ["Plazo de 3 meses para actos presuntos"],
  "strengths": ["Plazo de 1 mes correcto", "Referencia a la notificación"]
}
```

**Parámetros de la petición:**

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `question` | `string` | ✅ | Enunciado de la pregunta (mínimo 1 carácter) |
| `correct_answer` | `string` | ✅ | Respuesta de referencia |
| `student_answer` | `string` | ✅ | Respuesta del alumno a evaluar |

---

### 5. Pipeline de evaluación de calidad (RAGAS)

**Qué hace:** mide la calidad del sistema RAG con las métricas estándar de RAGAS (faithfulness, answer relevancy, context precision, context recall) y con métricas propias de dominio jurídico (precisión legal, completitud, claridad, exactitud factual).

```
EvalSample (query + contexts + answer + ground_truth)
      │
      ├─▶ RAGASEvaluator        → faithfulness, answer_relevancy,
      │                           context_precision, context_recall (0–1)
      │
      └─▶ OposicionesEvaluator  → legal_precision, completeness,
                                   clarity, factual_accuracy (1–5, LLM-judge)
```

#### Ejecutar la suite de evaluación

```bash
# Requiere dependencias opcionales del grupo evals
uv sync --group evals

# Ejecutar la suite completa
make eval

# O directamente con pytest
uv run pytest tests/evals/ -v -s
```

#### Configurar muestras de evaluación

Las muestras se definen en `tests/evals/conftest_evals.py`. Cada muestra incluye:

```python
EvalSample(
    query="¿Qué es el recurso de alzada?",
    contexts=["Artículo 121..."],          # chunks recuperados
    answer="El recurso de alzada es...",   # respuesta del sistema
    ground_truth="El recurso de alzada...", # respuesta de referencia
)
```

#### Interpretar los resultados

**`RAGASEvaluator` — métricas estándar (escala 0–1):**

| Métrica | Qué mide |
|---------|----------|
| `faithfulness` | La respuesta está fundamentada en el contexto recuperado |
| `answer_relevancy` | La respuesta responde la pregunta formulada |
| `context_precision` | Los chunks recuperados son relevantes para la pregunta |
| `context_recall` | El contexto recuperado cubre la respuesta de referencia |

**`OposicionesEvaluator` — métricas de dominio (escala 1–5, LLM-as-judge):**

| Métrica | Qué mide |
|---------|----------|
| `legal_precision` | Exactitud de referencias legales, números de artículo y normas |
| `completeness` | Cobertura de todos los aspectos exigibles en el examen |
| `clarity` | Claridad pedagógica para un opositor |
| `factual_accuracy` | Exactitud de fechas, cifras y pasos procedimentales |

Los resultados se publican en Langfuse (http://localhost:3000) cuando `LANGFUSE_PUBLIC_KEY` y `LANGFUSE_SECRET_KEY` están configurados.

---

## Variables de entorno

Copia `.env.example` a `.env` y ajusta los valores. Las variables marcadas con ✅ son obligatorias para el funcionamiento básico.

| Variable | Por defecto | ✅ | Descripción |
|----------|-------------|---|-------------|
| `LLM_PROVIDER` | `anthropic` | | Proveedor LLM: `anthropic`, `openai`, `ollama`, `groq` |
| `LLM_MODEL` | `claude-3-5-sonnet-20241022` | | Identificador del modelo |
| `LLM_TEMPERATURE` | `0.1` | | Temperatura de sampling (0 = determinista) |
| `ANTHROPIC_API_KEY` | — | ✅* | Requerida si `LLM_PROVIDER=anthropic` |
| `OPENAI_API_KEY` | — | ✅ | Siempre necesaria (embeddings `text-embedding-3-small`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | | URL de Ollama si `LLM_PROVIDER=ollama` |
| `GROQ_API_KEY` | — | ✅* | Requerida si `LLM_PROVIDER=groq` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | | Modelo de embeddings de OpenAI |
| `EMBEDDING_DIM` | `1536` | | Dimensión del vector (debe coincidir con el modelo) |
| `QDRANT_URL` | `http://localhost:6333` | | URL del servidor Qdrant |
| `QDRANT_COLLECTION` | `oposiciones_temario` | | Nombre de la colección vectorial |
| `DATABASE_URL` | `postgresql+asyncpg://...` | | URL de conexión PostgreSQL |
| `MINIO_ENDPOINT` | `localhost:9000` | | Endpoint de MinIO |
| `MINIO_ACCESS_KEY` | `minioadmin` | | Credencial de acceso MinIO |
| `MINIO_SECRET_KEY` | `minioadmin123` | | Credencial secreta MinIO |
| `MINIO_BUCKET` | `temario-docs` | | Bucket para almacenar documentos |
| `RETRIEVER_TYPE` | `hybrid` | | Tipo de retriever: `dense` o `hybrid` |
| `TOP_K` | `5` | | Chunks por defecto a recuperar |
| `CHUNK_SIZE` | `512` | | Tamaño de chunk en tokens |
| `CHUNK_OVERLAP` | `64` | | Solapamiento entre chunks en tokens |
| `LANGFUSE_HOST` | `http://localhost:3000` | | URL de Langfuse self-hosted |
| `LANGFUSE_PUBLIC_KEY` | — | | Clave pública de proyecto Langfuse |
| `LANGFUSE_SECRET_KEY` | — | | Clave secreta de proyecto Langfuse |

> **Nota Ollama:** para usar modelos locales con Ollama, configura `LLM_PROVIDER=ollama` y `OLLAMA_BASE_URL=http://localhost:11434`. Los embeddings siguen usando OpenAI, por lo que `OPENAI_API_KEY` sigue siendo necesaria.

> **Nota Groq:** para usar la inferencia rápida de Groq, configura `LLM_PROVIDER=groq`, `GROQ_API_KEY=gsk_...` y el modelo con prefijo `groq/`, por ejemplo `LLM_MODEL=groq/llama-3.3-70b-versatile`. Los embeddings siguen usando OpenAI.

---

## Comandos de desarrollo

```bash
# ── Infraestructura ───────────────────────────────────────────────
make up           # Levantar todos los servicios (API + frontend + infra)
make down         # Parar todos los contenedores
make infra        # Solo infraestructura (Qdrant, PostgreSQL, MinIO, Langfuse)
make dev          # Modo desarrollo con hot-reload

# ── Calidad de código ─────────────────────────────────────────────
make lint         # ruff check + ruff format --check
make format       # Auto-corrección de estilo (ruff format + ruff --fix)
make type-check   # mypy en modo estricto

# ── Tests ─────────────────────────────────────────────────────────
make test         # Suite completa con cobertura (mínimo 80%)
make test-unit    # Solo tests unitarios
make test-integration  # Tests de integración (requiere Docker activo)
make eval         # Suite de evaluación RAGAS

# ── Operaciones ───────────────────────────────────────────────────
make ingest FILE=<ruta> SUBJECT="<tema>"   # Ingestar documento
make logs SERVICE=<nombre>                 # Ver logs de un servicio
make shell SERVICE=<nombre>                # Shell dentro de un contenedor
make clean        # ⚠️ Elimina todos los volúmenes Docker (pérdida de datos)
```

---

## Estructura del proyecto

```
.
├── docker/
│   ├── Dockerfile.api          # Imagen del backend FastAPI
│   └── Dockerfile.frontend     # Imagen del frontend Streamlit
├── src/
│   ├── api/
│   │   ├── main.py             # App FastAPI + health check
│   │   ├── dependencies.py     # Inyección de dependencias
│   │   └── routers/
│   │       ├── ingestion.py    # POST /ingest
│   │       ├── chat.py         # POST /chat
│   │       └── exam.py         # POST /exam/generate, POST /exam/evaluate
│   ├── core/
│   │   ├── config/
│   │   │   ├── settings.py     # Configuración vía Pydantic Settings
│   │   │   └── prompts.py      # Plantillas de prompts LLM
│   │   ├── ingestion/          # Loaders (PDF, TXT) + IngestionPipeline
│   │   ├── retrieval/          # DenseRetriever, HybridRetriever, RetrieverFactory
│   │   ├── generation/         # RAGGenerator, ExamGenerator, AnswerEvaluator
│   │   ├── evals/              # RAGASEvaluator, OposicionesEvaluator
│   │   ├── exceptions.py       # Jerarquía de excepciones del dominio
│   │   └── protocols.py        # Protocolos compartidos (interfaces)
│   ├── infrastructure/
│   │   ├── llm/                # LiteLLMClient (Anthropic/OpenAI/Ollama/Groq)
│   │   ├── storage/            # MinIOStorage
│   │   └── vector_store/       # QdrantVectorStore
│   └── frontend/
│       ├── app.py              # Página principal Streamlit
│       ├── api_client.py       # Helpers HTTP reutilizables
│       └── pages/
│           ├── 01_chat.py      # Página Chat RAG
│           └── 02_exam.py      # Página Examen + Evaluación
├── tests/
│   ├── unit/                   # Tests unitarios por capa
│   ├── integration/            # Tests de integración (requieren Docker)
│   └── evals/                  # Suite de evaluación RAGAS
├── docker-compose.yml
├── pyproject.toml
├── Makefile
└── .env.example
```
