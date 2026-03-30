.PHONY: help up down infra test lint type-check format eval ingest logs shell clean install-venv

# ── Default ───────────────────────────────────────────────────────────────────
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Local setup ───────────────────────────────────────────────────────────────
install-venv: ## Create local virtualenv and install all dependencies with uv
	uv lock
	uv sync

# ── Docker ────────────────────────────────────────────────────────────────────
up: ## Start all services (production mode)
	docker compose up --build -d

down: ## Stop all services
	docker compose down

infra: ## Start only infrastructure services (qdrant, postgres, minio, langfuse)
	docker compose up -d qdrant postgres minio langfuse

dev: ## Start all services in development mode (hot-reload)
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# ── Quality ───────────────────────────────────────────────────────────────────
lint: ## Run ruff linter and format check
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format: ## Auto-fix formatting and linting issues
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

type-check: ## Run mypy type checker
	uv run mypy src/

# ── Tests ─────────────────────────────────────────────────────────────────────
test: ## Run tests with coverage (fails if coverage < 80%)
	uv run pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=80

test-unit: ## Run only unit tests
	uv run pytest tests/unit/ -v

test-integration: ## Run only integration tests (requires Docker services)
	uv run pytest tests/integration/ -v

# ── Evals ─────────────────────────────────────────────────────────────────────
eval: ## Run RAGAS evaluation suite
	uv run pytest tests/evals/ -v -s

# ── Operations ────────────────────────────────────────────────────────────────
ingest: ## Ingest a document: make ingest FILE=path/to/doc.pdf SUBJECT="Tema 1"
	uv run python -m src.core.ingestion.pipeline --file $(FILE) --subject "$(SUBJECT)"

logs: ## Tail logs for a service: make logs SERVICE=api
	docker compose logs -f $(SERVICE)

shell: ## Open a shell in a running service: make shell SERVICE=api
	docker compose exec $(SERVICE) /bin/bash

clean: ## WARNING: Remove all Docker volumes (data loss!)
	@echo "⚠️  This will DELETE all data in Docker volumes. Press Ctrl+C to abort."
	@sleep 3
	docker compose down -v
