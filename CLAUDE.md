# CLAUDE.md — Guía de Desarrollo para RAG Oposiciones

> Este fichero define las reglas, flujos de trabajo y convenciones que Claude debe
> seguir **siempre** al desarrollar este proyecto. Leerlo completo antes de escribir
> cualquier línea de código o crear cualquier fichero.

---

## 0. Principio Fundamental: Think Before You Code

Antes de escribir **cualquier** código, Claude debe:

1. **Leer** los ficheros relevantes del proyecto (`src/`, `tests/`, `docker/`)
2. **Entender** qué existe ya para no duplicar lógica
3. **Redactar un plan** en forma de lista numerada con los pasos exactos
4. **Esperar confirmación** del usuario antes de ejecutar el plan
5. **Ejecutar paso a paso**, confirmando cada paso antes del siguiente si hay dudas

```
⛔ NUNCA escribir código sin haber redactado el plan primero.
⛔ NUNCA asumir que algo "ya está bien" sin verificarlo en el árbol de ficheros.
✅ SIEMPRE mostrar el plan y esperar "ok", "adelante" o equivalente.
```

---

## 1. Flujo de Trabajo con Git

### 1.1 Rama por Funcionalidad

Cada nueva funcionalidad **siempre** se desarrolla en su propia rama:

```bash
# Nomenclatura obligatoria
feature/<nombre-descriptivo-en-kebab-case>

# Ejemplos válidos
feature/ingestion-pipeline
feature/hybrid-retriever
feature/exam-generator
feature/ragas-eval-suite
feature/streamlit-chat-ui
feature/jwt-auth
feature/export-exam-pdf

# Ejemplos INVÁLIDOS
fix-bug          # no tiene prefijo feature/
feature/fix      # demasiado genérico
feature/WIP      # no descriptivo
```

### 1.2 Secuencia de Trabajo Obligatoria

```bash
# 1. Partir siempre de main actualizado
git checkout main
git pull origin main

# 2. Crear la rama de la funcionalidad
git checkout -b feature/<nombre>

# 3. Desarrollar con commits atómicos (ver 1.3)
# ... código ...
git add -p   # staging interactivo, nunca 'git add .'
git commit -m "<tipo>(<scope>): <descripción>"

# 4. Antes de abrir PR: pasar todos los checks
make lint
make type-check
make test

# 5. Abrir PR hacia main (nunca push directo a main)
git push origin feature/<nombre>
# → Abrir Pull Request en GitHub
```

### 1.3 Convención de Commits (Conventional Commits)

```
<tipo>(<scope>): <descripción imperativa en español o inglés>

Tipos permitidos:
  feat      Nueva funcionalidad
  fix       Corrección de bug
  test      Añadir o modificar tests
  refactor  Cambio que no añade feat ni fix
  docs      Documentación
  chore     Tareas de mantenimiento (deps, config)
  perf      Mejora de rendimiento
  ci        Cambios en CI/CD

Ejemplos:
  feat(ingestion): add PDF loader with metadata extraction
  test(retrieval): add unit tests for hybrid retriever
  fix(exam-gen): handle malformed JSON from LLM response
  refactor(core): extract prompt templates to dedicated module
  docs(api): update OpenAPI descriptions for exam endpoints
```

### 1.4 Tamaño de los Commits

- Un commit = un cambio lógico cohesionado
- Si el diff supera ~300 líneas, dividir en commits más pequeños
- Los tests de una funcionalidad van en el **mismo commit** que la implementación, o en el commit inmediatamente siguiente

---

## 2. Estándares de Código Python

### 2.0 Idioma del Código — Obligatorio Inglés

```
✅ ALL code must be written in English:
   - Variable names, function names, class names
   - Docstrings and inline comments
   - Log messages
   - Exception messages
   - Type aliases and constants

⛔ NEVER write Spanish in source code, docstrings, or comments.
   Spanish is only allowed in: this CLAUDE.md file, user-facing UI strings,
   and exam/domain content (e.g. question text stored as data).
```

### 2.1 Versión y Herramientas

```toml
# pyproject.toml — configuración obligatoria
[project]
requires-python = ">=3.12"

[tool.ruff]
target-version = "py312"
line-length = 88
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "N",   # pep8-naming
    "ANN", # flake8-annotations (type hints)
    "S",   # flake8-bandit (security)
    "PTH", # use pathlib over os.path
]
ignore = ["ANN101", "ANN102", "S101"]  # self/cls hints, assert en tests

[tool.ruff.per-file-ignores]
"tests/*" = ["ANN", "S"]  # relajar annotations y security en tests

[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "--cov=src --cov-report=term-missing --cov-fail-under=80"
```

### 2.2 Type Hints — Obligatorios en Todo el Código

```python
# ✅ CORRECTO
from collections.abc import AsyncIterator
from pathlib import Path

async def load_document(path: Path, subject: str) -> list[Document]:
    ...

def create_chunks(text: str, size: int = 512) -> list[str]:
    ...

# ❌ INCORRECTO — sin type hints
async def load_document(path, subject):
    ...
```

### 2.3 Docstrings — Formato Google Style

```python
def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
    """Retrieve the most relevant chunks for a given query.

    Args:
        query: User question or search in natural language.
        top_k: Maximum number of results to return.

    Returns:
        List of RetrievalResult sorted by descending relevance.

    Raises:
        VectorStoreError: If Qdrant is not available.
        ValueError: If query is empty.

    Example:
        >>> retriever = HybridRetriever(index)
        >>> results = await retriever.retrieve("What is the appeal procedure?")
        >>> print(results[0].content)
    """
```

### 2.4 Estructura de Módulos

```python
# Orden obligatorio en cada fichero .py:
# 1. Docstring del módulo
# 2. __future__ imports
# 3. Standard library imports
# 4. Third-party imports
# 5. Local imports
# 6. __all__ (si el módulo es público)
# 7. Constantes
# 8. Clases y funciones

"""Módulo de recuperación híbrida BM25 + vectores."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from llama_index.core import VectorStoreIndex
from qdrant_client import QdrantClient

from src.core.config.settings import get_settings
from src.core.retrieval.base import Retriever, RetrievalResult

if TYPE_CHECKING:
    from src.infrastructure.vector_store.base import VectorStore

__all__ = ["HybridRetriever"]

DEFAULT_TOP_K = 5
```

### 2.5 Gestión de Errores

```python
# src/core/exceptions.py — Domain exception hierarchy
class RAGOposicionesError(Exception):
    """Base class for all project exceptions."""

class IngestionError(RAGOposicionesError):
    """Error during document ingestion."""

class RetrievalError(RAGOposicionesError):
    """Error during context retrieval."""

class GenerationError(RAGOposicionesError):
    """Error during response generation."""

class VectorStoreError(RAGOposicionesError):
    """Error communicating with Qdrant."""

# Correct usage: catch specific, re-raise with context
try:
    results = await qdrant_client.search(...)
except Exception as e:
    raise VectorStoreError(f"Qdrant search failed for query '{query}'") from e
```

### 2.6 Logging — Siempre Estructurado

```python
import structlog

logger = structlog.get_logger(__name__)

# ✅ CORRECT — with context
logger.info("document_ingested", file=path.name, chunks=len(chunks), subject=subject)
logger.error("retrieval_failed", query=query, error=str(e))

# ❌ INCORRECT — unstructured
print(f"Ingested {path.name}")
logging.info("Error: " + str(e))
```

---

## 3. Testing

### 3.1 Cobertura Mínima Obligatoria

| Capa | Cobertura mínima |
|------|-----------------|
| `src/core/` | **90%** |
| `src/infrastructure/` | **75%** |
| `src/api/` | **80%** |
| **Total proyecto** | **80%** |

El pipeline de CI **bloqueará** el merge si no se alcanza la cobertura mínima.

```bash
# Verificar cobertura localmente antes de hacer push
make test
# o directamente:
uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=80
```

### 3.2 Estructura de Tests

```
tests/
├── conftest.py              # Fixtures compartidas (settings, mock LLM, etc.)
├── unit/
│   ├── core/
│   │   ├── test_ingestion_pipeline.py
│   │   ├── test_hybrid_retriever.py
│   │   ├── test_exam_generator.py
│   │   └── test_answer_evaluator.py
│   └── infrastructure/
│       ├── test_qdrant_store.py
│       └── test_litellm_client.py
├── integration/
│   ├── test_api_chat.py
│   ├── test_api_exams.py
│   └── test_ingestion_flow.py   # requiere Qdrant en Docker
└── evals/
    ├── conftest_evals.py
    ├── test_ragas_metrics.py
    └── eval_suite.py            # suite ejecutable manualmente
```

### 3.3 Convenciones de Tests

```python
# tests/unit/core/test_exam_generator.py
import pytest
from unittest.mock import AsyncMock, patch
from src.core.generation.exam_generator import ExamGenerator
from src.core.generation.base import GenerationInput

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """Simulated LLM client that returns valid exam JSON."""
    client = AsyncMock()
    client.complete.return_value = '{"questions": [{"id": 1, "type": "test", ...}]}'
    return client

@pytest.fixture
def exam_generator(mock_llm):
    return ExamGenerator(llm_client=mock_llm, num_questions=5, exam_type="test")

@pytest.fixture
def sample_input():
    return GenerationInput(
        query="Appeal procedure",
        context=["The appeal procedure is...", "It is filed before..."],
        metadata={"subject": "Administrative Law"},
    )

# ── Tests ─────────────────────────────────────────────────────────────────────

class TestExamGenerator:
    """Tests unitarios para ExamGenerator."""

    @pytest.mark.asyncio
    async def test_generate_returns_questions(self, exam_generator, sample_input):
        """Verify that generate() returns questions in the expected format."""
        result = await exam_generator.generate(sample_input)

        assert result.content is not None
        assert "questions" in result.content

    @pytest.mark.asyncio
    async def test_generate_calls_llm_once(self, exam_generator, sample_input, mock_llm):
        """Verify that the LLM is called exactly once."""
        await exam_generator.generate(sample_input)

        mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_raises_on_empty_context(self, exam_generator):
        """Verify that ValueError is raised with empty context."""
        bad_input = GenerationInput(query="test", context=[])

        with pytest.raises(ValueError, match="context"):
            await exam_generator.generate(bad_input)

    @pytest.mark.asyncio
    async def test_generate_handles_malformed_json(self, exam_generator, sample_input, mock_llm):
        """Verify that malformed JSON from the LLM is handled without crashing."""
        mock_llm.complete.return_value = "Here is the exam: ```json{...}```"

        result = await exam_generator.generate(sample_input)

        # Must not raise; returns empty structure as fallback
        assert isinstance(result.content, dict)

# ── Parametrize para casos límite ─────────────────────────────────────────────

@pytest.mark.parametrize("exam_type,difficulty", [
    ("test", "facil"),
    ("desarrollo", "media"),
    ("mixto", "dificil"),
])
@pytest.mark.asyncio
async def test_generator_all_types(mock_llm, sample_input, exam_type, difficulty):
    """Verify that all exam types and difficulty levels work correctly."""
    generator = ExamGenerator(mock_llm, num_questions=3,
                               exam_type=exam_type, difficulty=difficulty)
    result = await generator.generate(sample_input)
    assert result is not None
```

### 3.4 Fixtures Globales (`conftest.py`)

```python
# tests/conftest.py
import pytest
from unittest.mock import AsyncMock
from src.core.config.settings import Settings

@pytest.fixture(scope="session")
def test_settings():
    """Settings with test values (no real API keys)."""
    return Settings(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        openai_api_key="test-key-not-real",
        qdrant_url="http://localhost:6333",
        database_url="sqlite+aiosqlite:///:memory:",
        debug=True,
    )

@pytest.fixture
def mock_llm_client():
    """Reusable LLM client mock for all tests."""
    client = AsyncMock()
    client.complete = AsyncMock(return_value='{"result": "ok"}')
    client.stream = AsyncMock()
    return client

@pytest.fixture
def mock_vector_store():
    """Vector store mock."""
    store = AsyncMock()
    store.add_documents = AsyncMock(return_value={"inserted": 10})
    store.search = AsyncMock(return_value=[])
    store.get_index = AsyncMock()
    return store
```

---

## 4. Plan de Ejecución — Formato Obligatorio

Cuando el usuario pida desarrollar una nueva funcionalidad, Claude **siempre** debe responder primero con este formato antes de escribir código:

```markdown
## 📋 Plan: <nombre de la funcionalidad>

**Rama:** `feature/<nombre-rama>`
**Estimación:** X ficheros nuevos, Y modificados

### Pasos

1. [ ] Crear rama `feature/<nombre>` desde `main`
2. [ ] Leer ficheros existentes: `src/core/X.py`, `tests/Y.py`
3. [ ] Crear `src/core/<modulo>/base.py` — Abstract base class
4. [ ] Crear `src/core/<modulo>/<implementacion>.py` — Implementación concreta
5. [ ] Crear `tests/unit/core/test_<modulo>.py` — Tests unitarios (≥90% cobertura)
6. [ ] Modificar `src/api/routers/<router>.py` — Exponer endpoint
7. [ ] Modificar `src/api/dependencies.py` — Registrar en DI
8. [ ] Ejecutar `make lint && make test` — Verificar calidad
9. [ ] Commit: `feat(<scope>): <descripción>`
10. [ ] Abrir PR hacia `main`

### Ficheros que se van a tocar
- **Nuevos:** `src/...`, `tests/...`
- **Modificados:** `src/api/dependencies.py`, `src/api/main.py`
- **Sin cambios:** `docker-compose.yml`, `pyproject.toml` (salvo nueva dep)

¿Procedo con este plan?
```

---

## 5. Checklist Antes de Hacer Push

Claude debe verificar cada punto antes de cualquier `git push`:

```
Calidad de código:
  [ ] ruff check src/ tests/          → 0 errores
  [ ] ruff format --check src/ tests/ → 0 cambios pendientes
  [ ] mypy src/                        → 0 errores de tipos

Tests:
  [ ] pytest tests/unit/              → todos pasan
  [ ] pytest --cov=src --cov-fail-under=80 → cobertura ≥ 80%
  [ ] No hay tests con `pytest.mark.skip` sin justificación documentada

Git:
  [ ] La rama se llama feature/<nombre-descriptivo>
  [ ] Todos los commits siguen Conventional Commits
  [ ] No hay ficheros de desarrollo temporal (.bak, debug_*, temp_*)
  [ ] No hay secrets, API keys ni passwords en ningún fichero

Documentación:
  [ ] Docstrings en todas las clases y métodos públicos
  [ ] README actualizado si se añade un comando o variable de entorno nueva
  [ ] CHANGELOG.md actualizado (si existe)
```

---

## 6. Reglas de Infraestructura y Docker

```
✅ Añadir nuevos servicios al docker-compose.yml con health checks
✅ Usar variables de entorno para toda configuración (nunca hardcodear)
✅ Nuevas dependencias Python → añadir a pyproject.toml y regenerar lock
✅ Nuevas variables de entorno → añadir a .env.example con descripción
⛔ Nunca hacer commit de ficheros .env con valores reales
⛔ Nunca usar 'latest' como tag de imagen Docker (fijar versión)
⛔ Nunca exponer puertos innecesarios en producción
```

### Variables de Entorno Nuevas

```bash
# .env.example — formato obligatorio para cada variable nueva
# Descripción de qué hace esta variable y valores posibles
NOMBRE_VARIABLE=valor_por_defecto_seguro
```

---

## 7. Reglas de Seguridad

```
⛔ Nunca loggear API keys, tokens o passwords
⛔ Nunca usar eval() o exec() con input del usuario
⛔ Validar SIEMPRE el input del usuario con Pydantic antes de usarlo
⛔ Los endpoints de ingesta deben validar tipo y tamaño de fichero
✅ Usar httpx con timeouts explícitos en todas las llamadas externas
✅ Sanitizar nombres de fichero antes de almacenarlos (Path.name)
✅ Rate limiting en endpoints públicos (slowapi)
```

---

## 8. Gestión de Dependencias

```bash
# Añadir dependencia de producción
uv add <paquete>

# Añadir dependencia de desarrollo
uv add --dev <paquete>

# Actualizar lock file
uv lock

# Sincronizar entorno
uv sync

# ⛔ Nunca usar pip install directamente en el proyecto
# ⛔ Nunca modificar uv.lock a mano
```

---

## 9. Referencia Rápida de Comandos

```bash
make up          # Levantar toda la infraestructura Docker
make down        # Parar todos los contenedores
make infra       # Solo qdrant + postgres + minio + langfuse
make test        # Ejecutar tests con cobertura
make lint        # ruff check + ruff format --check
make type-check  # mypy src/
make format      # ruff format src/ tests/ (auto-fix)
make eval        # Ejecutar suite de evals RAGAS
make ingest      # FILE=path SUBJECT="nombre" → ingestar documento
make logs        # SERVICE=api → ver logs en tiempo real
make shell       # SERVICE=api → entrar al contenedor
make clean       # ⚠️  Borrar todos los volúmenes Docker
```

---

## 10. Estructura de Ramas y PRs

```
main
 └── feature/ingestion-pipeline         ← fase 1
 └── feature/dense-retriever            ← fase 2a
 └── feature/hybrid-retriever           ← fase 2b
 └── feature/rag-chat-generator         ← fase 2c
 └── feature/exam-generator-test        ← fase 3a
 └── feature/exam-generator-desarrollo  ← fase 3b
 └── feature/answer-evaluator           ← fase 3c
 └── feature/ragas-eval-suite           ← fase 4
 └── feature/fastapi-ingestion-router   ← fase 5a
 └── feature/fastapi-exam-router        ← fase 5b
 └── feature/streamlit-chat-page        ← fase 6a
 └── feature/streamlit-exam-page        ← fase 6b
 └── feature/jwt-auth                   ← v1.0
 └── feature/study-progress-tracking    ← v1.0
```

### Descripción de PR (template)

```markdown
## ¿Qué hace este PR?
Breve descripción de la funcionalidad añadida.

## Cambios
- `src/core/X.py`: nueva clase Y con patrón Z
- `tests/unit/test_X.py`: cobertura del módulo X

## Testing
- [ ] Tests unitarios pasan (`make test`)
- [ ] Cobertura ≥ 80% verificada
- [ ] Lint limpio (`make lint`)
- [ ] Type check limpio (`make type-check`)

## Notas para el reviewer
Cualquier decisión de diseño relevante o deuda técnica aceptada.
```

---

*Última actualización: inicio del proyecto. Revisar y actualizar este fichero
cuando cambien las convenciones del equipo.*