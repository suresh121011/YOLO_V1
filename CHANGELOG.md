# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Stage 1: Repository foundation and project skeleton
  - Production-ready folder structure
  - `pyproject.toml` with Black, Ruff, MyPy, Pytest configuration
  - `requirements.txt` with all V1 dependencies
  - `Makefile` with development workflow targets
  - `configs/` — YAML configuration stubs for data, training, deployment, rules, feature flags
  - `src/pipeline/__init__.py` — Locked data contracts (Detection, Alert, SceneContext, etc.)
  - `src/pipeline/` — Module stubs for all pipeline components
  - `src/config/config_loader.py` — Configuration system stub
  - `src/logging/structured_logger.py` — Logging system stub
  - `tests/` — Complete test directory structure (unit / integration / system / performance)
  - `dvc.yaml` — DVC pipeline definition stub
  - `docs/` — Full technical documentation (3-document structure)

---

## [0.1.0] - 2026-07-14 (Planned)

### Stage 1 — Project Bootstrap & Repository Foundation

Initial skeleton release. No application logic implemented.
Establishes production-ready engineering infrastructure for all subsequent stages.

---

*Future versions will be documented here as each stage is completed.*
