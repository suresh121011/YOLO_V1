# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Phase-5: Production Dataset Engineering, Missing-Annotation Resolution &
  Dataset v1.0 — makes dataset quality the primary solution and demotes
  Phase-4 masking to a safety net. Core invariant: auto-generated labels never
  touch `labels/` (ADR-P5-01) — they live in a candidate artifact, cross a
  mandatory human CVAT round-trip, land in a verification ledger + delta labels,
  and only then overlay onto merged labels. Masking shrinks exactly as
  verification grows. **All milestone tooling (M0–M11) is implemented and
  validated; the remaining work is operational/human (see
  `docs/07_dataset_production/phase5_milestone_status.md`).**
  - Missing-annotation resolution L1–L5: L1 human annotation (P3 CVAT flow +
    verification batches) · L2 model-assisted (`src/dataset/annotation/` —
    `AutoAnnotator` ABC + registry, `yolo_world` primary backend, optional
    `grounding_dino`, MobileSAM refine pass; sha256-pinned weights; within-
    machine determinism, `--verify-determinism`) · L3 cross-dataset salvage
    (`src/dataset/cross_dataset_salvage.py` — exact-SHA256 label transplant with
    IoU≥0.9 suppression + near-dup `cross_dataset` candidates) · L4 coverage
    estimation (`coverage.py` — pure arithmetic over pinned candidates, no
    inference at report time) · L5 dataset quality report (`quality.py` —
    residual-risk quantification, verification-progress + batch throughput)
  - Verification loop: candidate artifact → verification batches
    (`cvat_package.py`, CVAT YOLO-1.1 pre-annotation zips) → human CVAT verify →
    import (`verified_import.py`, class-order + non-target byte-equality
    hard-fails, dual-annotator IAA gate) → append-only ledger (`ledger.py`,
    `cache: false`, git-tracked) → labels overlay `data/merged_verified`
    (`apply.py`) — empty ledger ⇒ byte-identical passthrough (golden regression)
  - Completeness expansion: new policy mode `trusted_list_with_ledger`
    (`completeness_policies.py`) composes source policies with ledger-verified
    trust; additive `PolicyContext.verification_ledger` field (back-compatible);
    preflight gate G9 (ledger ↔ verified_labels ↔ provenance consistency)
  - Releases as code: `src/dataset/release/{gates,manifest}.py` — gates RG1–RG10
    + MODE/WETFLOOR prerequisites (pure functions over loaded artifacts), per-
    track thresholds in `configs/release.yaml`, frozen `record_release` stage,
    v0.5.0→v0.7.0→v0.9.0→v1.0.0 ladder
  - CLI ladder `12`–`19` (`scripts/dataset/`): auto_annotate, build/import
    verification batches, apply verified labels, coverage/quality reports,
    make_release, quality_delta; `scripts/training/evaluate_model.py` (single-
    checkpoint eval + wet_floor AP50 checkpoint); `scripts/qa/validate_phase5.py`
    (M6 correctness gate) + `full_build_preflight.py` (FB1–FB6)
  - New config files `configs/annotation.yaml` + `configs/release.yaml` (no new
    keys in acquisition params — DVC hash hygiene); `configs/eval_data.yaml`
  - DVC stages: `auto_annotate`, `apply_verified_labels`, `coverage_report`,
    `dataset_quality_report` (auto-repro); `build_verification_batches`,
    `import_verified_annotations`, `record_release`, `evaluate_yolo11n` (frozen,
    human/GPU-committed — the human-loop stages declare no deps to break the
    declared-graph cycle; freshness enforced by sha256 cross-checks)
  - Vectorized-Hamming dedup for full-mode scale (`dedup.py`, ADR-P5-12) with a
    decision-equivalence property test; WIDER FACE `class_caps`; auto-derived
    Roboflow per-slug licenses for RG7
  - Local-drive DVC remote as default (`C:\dvc_remote`, off the OneDrive tree) —
    first `dvc push` executed, closing audit risk C-1; S3 kept as secondary
  - Docs: `docs/07_dataset_production/` (README, auto-annotation / verification /
    release runbooks, **twelve ADRs `adr/ADR-P5-01…12`**, milestone-status doc);
    risk register R30–R38
  - ~440 new tests (suite → 1003 passing) across unit/integration/system/
    performance, including the M6 full-loop system smoke test; coverage floor
    raised 40 → 50

- Phase-4: Missing Annotation Mitigation — masked-loss training framework
  (public datasets label only part of the 23-class taxonomy; stock BCE turns
  every unlabeled class into false background supervision — mean smoke image
  trusts just 6.25/23 classes). Strictly opt-in; disabled ⇒ byte-for-byte
  stock pipeline (golden train-kwargs regression pins this).
  - M1 completeness metadata: top-level `completeness.policies` section in
    `configs/dataset_sources.yaml` (explicit per-source semantics — never
    name-inferred; `negatives` = verified absence of ALL classes → all-ones
    mask; `custom_captures` = per finalized session manifest), pluggable
    policy-provider registry (`src/dataset/completeness_policies.py`),
    hard-fail generator + validator (`src/dataset/completeness.py`, orphan
    refs / duplicate keys / drift / unknown images all fatal), CLI
    `scripts/dataset/11_generate_completeness.py`, new DVC stage
    `generate_completeness` (split → completeness → frozen train dep),
    report triplet `data/qa_reports/completeness_report.*`
  - M2 preflight gates G1–G8 (`src/training/preflight.py` +
    `scripts/training/preflight_check.py`, exit 0/1/2): artifact
    exists/valid, taxonomy fingerprint vs live data.yaml, train/val
    coverage, self-consistency, environment + loss-surface source canary
    (`assert_ultralytics_compat`), config validity, input-hash freshness,
    strict mixing-augmentation gate (mosaic/mixup/copy_paste forbidden
    under mitigation — ADR-P4-04)
  - M3 masked BCE loss + trainer injection: `MaskedDetectionLoss`
    (v8DetectionLoss subclass; `_MaskingBCE` wrapper multiplies the
    elementwise BCE map by a per-image {0,1}^23 mask — no upstream math
    copied; box/DFL untouched), criterion attached at `on_train_start` to
    train + EMA models (model class stays stock → portable checkpoints,
    ADR-P4-02), `build_masked_trainer` factory for
    `model.train(trainer=...)`, `--mitigation on|off` CLI,
    `missing_annotation_mitigation` config section (yolo11n + yolo11s)
  - M3.5 masking-correctness gate (committed evidence:
    `data/qa_reports/phase4_mitigation/masking_validation_report.*`):
    bit-identity vs stock loss under all-ones masks, exact-zero gradients
    for masked classes, real-artifact spot-checks (coco 10/23,
    openimages 3/23, wider_face 1/23, negatives 23/23), 1-epoch mitigated +
    disabled runs — all PASS; re-runnable via
    `scripts/training/validate_masking.py` + env-gated
    `tests/system/test_training_smoke.py`
  - M4 evaluation framework (`src/training/evaluation.py`,
    `scripts/training/evaluate_mitigation.py`): per-class P/R/F1/mAP,
    confusion-matrix export, mitigated−baseline delta reports with the
    partial-annotation caveat documented
  - M5 benchmark framework (`src/training/benchmark.py`,
    `scripts/training/benchmark_mitigation.py`): baseline vs mitigated,
    repeats, process-tree peak RSS (psutil), loss-forward + mask-build
    microbenchmarks, explicit performance budgets each marked PASS/FAIL
    (verdict FAIL on any breach); executed smoke benchmark committed
    (`data/qa_reports/phase4_mitigation/benchmark_report.*`) — all budgets
    PASS (end-to-end wall-time overhead ≈0 %, loss-forward ≈0.6 ms/call
    ≈0.2 % of a training step, mask build ≈0.06 ms/batch); microbenchmark
    uses interleaved stock/masked rounds with median reduction after
    single-series timing proved unreliable on desktop hardware
  - M6 documentation: `docs/06_training_engineering/` (engineering report,
    masked-loss architecture with identity proof + compat contract,
    operational runbook keyed by gate IDs, ADR-P4-01…05), risk register
    R25–R29, README Phase-4 section
  - 153 new CI-scope tests (unit incl. torch-dependent drift canaries +
    integration against the real shipped configs; suite 412 → 565, coverage
    73.35 % → 74.99 %) plus an env-gated system smoke test; CI mypy scope
    now includes `src/training`; `psutil` promoted to an explicit
    dependency

- Pre-Phase-4 production readiness audit
  (`docs/05_audit/pre_phase4_production_readiness_audit.md`) — phase
  verification (1/2/WP3.0/3 all PASS), full findings register, CI/DVC/git
  review, Phase-4 readiness assessment, prioritized action plan; verdict:
  ✅ ready for Phase 4 (first `dvc push` remains the gate before real
  capture collection)
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
- Stale DVC pipeline state (audit H-2): `dvc repro` re-run with Phase-3 code
  refreshed `dvc.lock` (merge/split/QA stages) and locked the new
  `generate_completeness` stage; regenerated QA metric verified sane
  (188 images, 0 critical, 18 pre-existing warnings)
- Local mypy gate aborted on venvs with numpy 2.x installed (PEP 695 `type`
  statements in numpy stubs vs the hard `python_version = "3.10"` pin);
  pin removed — mypy now checks under the running interpreter while CI's
  Python-3.10 quality job keeps enforcing the 3.10 floor. Supersedes the
  ineffective `numpy.*` ignore-missing-imports override attempt.
- `AlertQueue` heap comparison `TypeError` on coarse-resolution clocks
  (equal-severity alerts with identical `time.monotonic()` timestamps fell
  through to non-orderable `Alert` objects, failing windows-latest CI);
  strictly increasing sequence-number tiebreaker added
  (`src/pipeline/alert_queue.py`)
- `requests` added to runtime dependencies (`requirements.txt`,
  `pyproject.toml`) — downloader tests import it transitively and fresh
  environments failed collection
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
