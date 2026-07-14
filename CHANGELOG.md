# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Phase-3: Custom Dataset Collection & Annotation tooling
  - `src/dataset/capture/` — collection/annotation library: typed capture
    config (`configs/capture_config.yaml`), PII-free consent verification
    against a local-only registry, EXIF/GPS metadata stripping, inbox→session
    ingest (corruption/size/duplicate gates, session manifests, aggregate
    manifest rebuild), CVAT-compatible YOLO-export import with class-order
    verification (the CVAT footgun: a subset/reordered label list silently
    shifts every class ID) and session-scoped label validation, staging +
    finalize, inter-annotator agreement (greedy IoU matching, per-class
    gates incl. a `wet_floor` R24 override), and governance-target progress
    tracking (per-class counts, houses/rooms/lighting coverage, withdrawn-
    consent flags)
  - `scripts/dataset/08_ingest_capture_session.py`,
    `09_import_annotations.py`, `10_capture_progress.py` — CLIs (ingest,
    stage/compare/finalize, progress), consistent exit 0/1/2 contract
  - `src/dataset/splitting/leave_one_house_out.py` — house-level split
    strategy (all sessions of one house share a split; `holdout_houses`
    forces named houses into test — the eval-set leakage-prevention
    mechanism); public-source images without a house match degrade to
    `group_aware` behavior
  - `scripts/qa/run_full_qa.py`: eval-set overlap guard (exact SHA-256 +
    flip-robust perceptual near-duplicate against train-facing data,
    CRITICAL) and house-exclusivity check (train/eval house overlap,
    WARNING); both opportunistic (`{"available": false}` pre-Phase-3)
  - `dvc.yaml`: `ingest_custom_captures` / `ingest_eval_set` frozen stages
    — human-in-the-loop data enters `dvc.lock` via `dvc commit -f`, never
    via `dvc repro` (which would delete-then-regenerate real photos as
    empty on any machine without the capture inbox); `merge_datasets`
    gains a dependency on `data/raw/custom_captures`
  - `docs/04_dataset_engineering/capture_annotation_runbook.md` — full
    operational SOP (consent → capture → ingest → CVAT annotation → IAA →
    finalize → DVC recording → eval-set locking → wet_floor R24 pilot gate
    → Roboflow slug checklist → dataset-v1.0.0 release checklist);
    `docs/03_engineering_appendix/consent_form_template.md`;
    `data/consent/README.md`
  - Risk register: R24 (`wet_floor` taxonomy risk) added with a measurable
    gate (docs/01 `risk_register.md`)
  - 129 new unit tests + 1 end-to-end integration test simulating the full
    human workflow on synthetic data (inbox → ingest → dual-annotator CVAT
    export → IAA → finalize → merge → split → QA → eval-set overlap →
    lock)

- WP3.0 platform remediation (Phase-2 closure review follow-up)
  - `tests/unit/test_downloaders.py` + `tests/unit/test_downloaders_parsers.py` —
    40 offline unit tests for the acquisition framework (fetch_url retry/resume/
    atomicity, download() template + manifests, COCO/Open Images/WIDER FACE
    parsers, negatives selection, Roboflow skip contract, CLI exit codes);
    downloader package coverage 0% → ~93%, overall 43% → 65%
  - `.env.example` documenting `ROBOFLOW_API_KEY` (graceful-skip semantics)

### Fixed
- Roboflow cross-dataset image budget decremented by *distinct class count*
  instead of images copied (`_consolidate_export` now returns the copied
  count; regression-tested)
- QA reports no longer embed absolute machine paths: `data_dir` and issue
  file paths are written cwd-relative with posix separators
  (`portable_path` in `scripts/qa/check_annotations.py`)
- Machine-specific Windows cache path removed from the tracked
  `.dvc/config`; per-machine relocation now documented via
  `dvc cache dir --local` (docs/04 §6)
- `generate_splits.py` docstring falsely claimed to be the DVC stage entry
  point (the stage runs `split_dataset.py`); clarified as a convenience
  wrapper
- 4 mypy errors in `src/logging/structured_logger.py` /
  `src/config/config_loader.py`; `psutil` added to stub overrides

### Changed
- DVC: default S3 remote `storage` configured in `.dvc/config`
  (`s3://elderly-assistant-mlops/datasets/yolo_v1`; activation runbook in
  docs/04 §6); `dvc` dependency now installs the S3 extra (`dvc[s3]`)
- CI: test matrix expanded to ubuntu+windows × py3.10/3.12; coverage gate
  enabled (`fail_under = 40`, ratchet-only); mypy widened to
  `src/dataset src/utils src/config src/logging` (`src/pipeline` joins in
  Phase-6); dev tooling installs use requirements.txt-matching bounds
- `run_workflow.sh` now wraps `dvc repro` — the DVC DAG is the single
  orchestration path (previously drove a divergent script chain plus
  webcam inference)

- Stage 2: Dataset Collection & Dataset Engineering (Phase-2)
  - `src/dataset/` — dataset engineering library: provenance manifests
    (source / capture-session / merged), acquisition config loader with
    smoke/full mode + license gate, class remapping (copy & in-place modes),
    indoor/quality filters, flip-robust perceptual dedup, multi-source merge
    with lineage, negative selection, split-strategy registry
    (`group_aware`, `stratified_group`; `kfold`/`leave_one_house_out` reserved)
  - `src/dataset/downloaders/` — bespoke annotations-first downloaders for
    COCO 2017, Open Images V7, WIDER FACE (license-gated), negatives, plus a
    Roboflow Universe SDK integration (graceful skip without API key)
  - `scripts/dataset/01–07` acquisition/processing CLIs matching `dvc.yaml`
  - `scripts/qa/run_full_qa.py` — QA orchestrator: structural checks + stats
    + license gate + label-completeness + blur/low-light checks (risk R01),
    all merged into the DVC metric `data/qa_reports/annotation_qa_report.json`
  - `configs/dataset_sources.yaml` — acquisition config, doubles as DVC params;
    `configs/dataset_split_config.yaml` now actually read by the split CLIs
  - DVC initialized (cache outside OneDrive), truthful `dvc.yaml` DAG
    (download → remap → merge → split → QA; training stage frozen for Phase-5)
  - `docs/04_dataset_engineering/` — license register, label-completeness
    policy, DPDP/PII notes, split governance, Phase-2 descope statement
  - `tests/integration/test_dataset_pipeline.py` — first offline end-to-end
    pipeline test; ~70 new unit tests (296 total assertions across 241+ tests)

### Changed
- Smoke dataset validated end-to-end: 188 images / 4 sources through
  `dvc repro` with QA zero critical issues (tag `dataset-v0.1.0-smoke`)
- CI: unit/integration tests now blocking; mypy gates `src/dataset`
- Repo-wide lint cleanup (60+ pre-existing ruff violations fixed);
  Windows cp1252 console crashes fixed; `.gitkeep` no longer counted as
  split leakage; `PipelineMetrics`-unrelated runtime defects logged for
  Phase-6 (see docs/04 §7 and the Phase-2 plan)

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

## [0.1.0] - Superseded

This entry originally described a Stage-1-only "skeleton release, no application
logic implemented." That is no longer accurate — Stage 1 (repository foundation)
and Stage 2 / Phase-2 (dataset engineering platform) are both complete and
documented in the `[Unreleased]` section above. No `0.1.0` tag has actually been
cut; the dataset-specific milestone is tracked instead via the `dataset-v0.1.0-smoke`
git tag. This stub is kept only for changelog continuity and should not be read
as a current status statement.

---

*Future versions will be documented here as each stage is completed.*
