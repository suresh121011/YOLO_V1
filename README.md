# Elderly Assistant System

> *"To give every elderly person living in an Indian home a silent, always-on safety companion — one that watches, understands, and speaks — without ever invading their privacy."*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: ruff](https://img.shields.io/badge/linting-ruff-red.svg)](https://github.com/astral-sh/ruff)

---

## Overview

The **Elderly Assistant System** is a real-time, 100% offline AI safety companion for elderly people living in Indian homes. It uses a camera to detect household hazards and delivers spoken guidance in natural Indian English — without sending any data to the cloud.

| Capability | Technology | Status |
|:-----------|:-----------|:-------|
| Object Detection (23 classes) | YOLO11n | 🔄 Stage 2 |
| Scene Understanding | SmolVLM2-256M | 🔄 Stage 6 |
| Safety Rule Engine | YAML-configured | 🔄 Stage 6 |
| Spoken Guidance | Piper TTS (en_IN) | 🔄 Stage 6 |
| Event Memory | Python sliding window | 🔄 Stage 6 |
| Active Learning Logging | JSONL structured logs | 🔄 Stage 6 |

---

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd YOLO_V1

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install project as editable package
pip install -e .

# 5. Verify setup
make check
```

---

## Project Structure

```
YOLO_V1/
├── configs/           # All YAML configuration files
├── data/              # DVC-tracked datasets (raw → processed)
├── docs/              # Full technical documentation
├── logs/              # Runtime, training, and QA logs (gitignored)
├── models/            # Trained weights and exports (DVC-tracked)
├── outputs/           # Visualizations and benchmark results
├── scripts/           # Dataset, QA, training, and inference scripts
├── src/               # Core Python package
│   ├── pipeline/      # Detection, memory, rules, TTS, orchestrator
│   ├── config/        # Configuration loader
│   ├── logging/       # Structured logging
│   └── plugins/       # Plugin system (V2+)
└── tests/             # Unit, integration, system, performance tests
```

---

## Development Commands

```bash
make lint        # Run ruff linter
make format      # Run black formatter
make check       # Run ruff + mypy
make test        # Run all tests
make test-unit   # Run unit tests only
make clean       # Remove __pycache__ and .pyc files
```

---

## Documentation

All documentation is in [`docs/`](./docs/README.md):

- **[01 Executive Implementation Plan](./docs/01_executive_implementation_plan/README.md)** — Product vision, phases, roadmap
- **[02 Technical Architecture Specification](./docs/02_technical_architecture_specification/README.md)** — System design, data contracts, threading model
- **[03 Engineering Appendix](./docs/03_engineering_appendix/README.md)** — YAML examples, Python examples, QA pipeline, DVC pipeline
- **[04 Dataset Engineering & Governance](./docs/04_dataset_engineering/README.md)** — Phase-2 pipeline, license register, label-completeness policy, split governance
- **[Capture & Annotation Runbook](./docs/04_dataset_engineering/capture_annotation_runbook.md)** — Phase-3 custom Indian-home data collection SOP
- **[06 Training Engineering](./docs/06_training_engineering/README.md)** — Phase-4 missing-annotation mitigation: masked BCE loss, preflight gates, benchmarks, ADRs

### Dataset pipeline (Phase-2)

```bash
dvc repro            # smoke-scale build: download → remap → merge → split → QA
dvc metrics show     # QA verdict (data/qa_reports/annotation_qa_report.json)
```

The full dataset build is one command: set `mode: full` in
`configs/dataset_sources.yaml`, then `dvc repro`. Roboflow sources need
`ROBOFLOW_API_KEY` and dataset slugs configured (skipped gracefully otherwise).

### Custom capture & annotation (Phase-3)

8 of the 23 classes (`gas_cylinder`, `medicine_strip`, `wet_floor`,
`walking_stick`, `support_handle`, Indian `stove`/`cupboard`, `passport`)
have no adequate public-dataset coverage and require custom Indian-home
capture. The collection/annotation/QA tooling is built:

```bash
python scripts/dataset/08_ingest_capture_session.py   # inbox → validated session
python scripts/dataset/09_import_annotations.py       # CVAT export → stage/compare/finalize
python scripts/dataset/10_capture_progress.py         # progress vs governance targets
```

Full SOP (consent, capture guidelines, dual-annotator CVAT workflow, DVC
recording, eval-set locking): [capture_annotation_runbook.md](./docs/04_dataset_engineering/capture_annotation_runbook.md).

### Missing-annotation mitigation (Phase-4)

Public datasets label only part of the 23-class taxonomy (COCO has `person`
but not `face`), so stock YOLO training pushes unlabeled-but-present classes
toward background. Phase-4 removes that false supervision with a per-image
**masked BCE classification loss**, driven by DVC-tracked completeness
metadata — strictly opt-in, byte-for-byte stock behavior when disabled:

```bash
dvc repro generate_completeness                    # per-image completeness artifact
python scripts/training/preflight_check.py         # readiness gates G1–G8
python scripts/training/train_yolo.py --mitigation on
python scripts/training/benchmark_mitigation.py --smoke   # baseline vs mitigated A/B
```

Architecture, ADRs, runbook, and committed validation/benchmark evidence:
[docs/06_training_engineering](./docs/06_training_engineering/README.md).

---

## V1 Success Metrics

| Metric | Target |
|:-------|:-------|
| Safety-critical class recall | ≥ 0.80 |
| False positive alert rate | < 10% |
| End-to-end alert latency | < 2 seconds |
| System uptime (24h unattended) | ≥ 99% |
| Field test scenario pass rate | 10/10 |

---

## 23-Class Object Taxonomy

| Category | Classes |
|:---------|:--------|
| Safety-Critical | `knife` · `stove` · `gas_cylinder` · `wire` · `wet_floor` · `medicine_strip` · `medicine_bottle` |
| Navigation | `person` · `face` · `door` · `walking_stick` · `support_handle` |
| Furniture | `chair` · `bed` · `cupboard` · `toilet` · `sink` |
| Daily Objects | `water_bottle` · `laptop` · `monitor` · `charger` · `book` |
| Documents | `passport` |

---

## Version History

See [CHANGELOG.md](./CHANGELOG.md).

---

## License

See [LICENSE](./LICENSE).
