# ─────────────────────────────────────────────────────────────────────────────
# Elderly Assistant System — Makefile
# ─────────────────────────────────────────────────────────────────────────────
# All targets are phonies (not files) unless stated otherwise.
# Usage:
#   make help         — show this message
#   make setup        — create venv and install dependencies
#   make format       — auto-format code with black
#   make lint         — lint with ruff
#   make check        — lint + type-check (no auto-fix)
#   make test         — run all tests
#   make test-unit    — run unit tests only
#   make test-perf    — run performance tests only
#   make clean        — remove build artifacts
#   make dvc-repro    — reproduce the full DVC pipeline
# ─────────────────────────────────────────────────────────────────────────────

.DEFAULT_GOAL := help
PYTHON       := python
VENV         := .venv
PIP          := $(VENV)/bin/pip
SRC          := src scripts tests

# ─── Detect OS for path handling ─────────────────────────────────────────────
ifeq ($(OS),Windows_NT)
    VENV_PYTHON := $(VENV)/Scripts/python
    VENV_PIP    := $(VENV)/Scripts/pip
    SEP         := \\
else
    VENV_PYTHON := $(VENV)/bin/python
    VENV_PIP    := $(VENV)/bin/pip
    SEP         := /
endif

# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help
help:  ## Show this help message
	@echo ""
	@echo "  Elderly Assistant System — Development Commands"
	@echo "  ─────────────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ─── Environment ─────────────────────────────────────────────────────────────

.PHONY: setup
setup:  ## Create virtualenv and install all dependencies
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	$(VENV_PIP) install -e .
	@echo ""
	@echo "  ✅ Environment ready. Activate with:"
	@echo "       source $(VENV)/bin/activate   (Linux/macOS)"
	@echo "       $(VENV)\\Scripts\\activate      (Windows)"

.PHONY: setup-smolvlm
setup-smolvlm:  ## Also install SmolVLM2 dependencies (hardware-dependent)
	$(VENV_PIP) install -r requirements-smolvlm.txt

# ─── Code Quality ─────────────────────────────────────────────────────────────

.PHONY: format
format:  ## Auto-format all Python files with black
	black $(SRC)

.PHONY: lint
lint:  ## Lint all Python files with ruff
	ruff check $(SRC)

.PHONY: lint-fix
lint-fix:  ## Lint and auto-fix with ruff
	ruff check --fix $(SRC)

.PHONY: typecheck
typecheck:  ## Run mypy type checker
	mypy src/

.PHONY: check
check: lint typecheck  ## Run linter + type checker (no auto-fix)

.PHONY: check-all
check-all: lint typecheck test  ## Full pre-commit check (lint + types + tests)

# ─── Testing ──────────────────────────────────────────────────────────────────

.PHONY: test
test:  ## Run all tests
	pytest tests/ -v

.PHONY: test-unit
test-unit:  ## Run unit tests only
	pytest tests/unit/ -v -m unit

.PHONY: test-integration
test-integration:  ## Run integration tests only
	pytest tests/integration/ -v -m integration

.PHONY: test-system
test-system:  ## Run system tests only (requires full pipeline)
	pytest tests/system/ -v -m system

.PHONY: test-perf
test-perf:  ## Run performance budget tests
	pytest tests/performance/ -v -m performance

.PHONY: coverage
coverage:  ## Run tests with coverage report
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

# ─── DVC Pipeline ─────────────────────────────────────────────────────────────

.PHONY: dvc-repro
dvc-repro:  ## Reproduce the full DVC pipeline (download → process → train → eval)
	dvc repro

.PHONY: dvc-dag
dvc-dag:  ## Show DVC pipeline DAG
	dvc dag

.PHONY: dvc-status
dvc-status:  ## Show DVC pipeline status
	dvc status

.PHONY: dvc-metrics
dvc-metrics:  ## Show training metrics comparison
	dvc metrics diff

# ─── Utilities ────────────────────────────────────────────────────────────────

.PHONY: clean
clean:  ## Remove build artifacts, __pycache__, .pyc files
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -not -path "./.venv/*" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -not -path "./.venv/*" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage
	@echo "  ✅ Cleaned."

.PHONY: clean-logs
clean-logs:  ## Clear all runtime logs (privacy — use before sharing)
	find logs/ -name "*.jsonl" -delete 2>/dev/null || true
	find logs/ -name "*.log" -delete 2>/dev/null || true
	@echo "  ✅ Logs cleared."

.PHONY: version
version:  ## Show current project version
	@python -c "import tomllib; f=open('pyproject.toml','rb'); d=tomllib.load(f); print(d['project']['version'])"
