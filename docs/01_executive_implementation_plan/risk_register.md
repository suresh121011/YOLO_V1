# Risk Register

## Purpose

Comprehensive risk catalogue with impact scoring, likelihood, mitigation strategies, and ownership.

## Dependencies

Reads:
- implementation_phases.md
- architecture_overview.md

Used By:
- validation_strategy.md
- recommendations.md

Related:
- security_privacy.md

---

## Risk Scoring Guide

| Score | Impact | Likelihood |
|:------|:-------|:-----------|
| 5 | Catastrophic | Almost Certain |
| 4 | Major | Likely |
| 3 | Moderate | Possible |
| 2 | Minor | Unlikely |
| 1 | Negligible | Rare |

## Risk Register Table

| # | Risk | Impact | Likelihood | Score | Mitigation | Owner |
|:--|:-----|:-------|:-----------|:------|:-----------|:------|
| R01 | **Poor lighting** — dim/night scenes degrade detection | 5 | 5 | 25 | Collect dark-scene training data; HSV augmentation; IR camera option for V2 | CV Engineer |
| R02 | **Motion blur** — camera shake during capture/inference | 4 | 4 | 16 | Laplacian sharpness filter; motion-adaptive confidence thresholds | CV Engineer |
| R03 | **Dataset imbalance** — rare classes underrepresented | 5 | 4 | 20 | Weighted class sampling; augmentation; mandatory minimum 200 instances | ML Architect |
| R04 | **Occlusion** — objects partially hidden | 4 | 4 | 16 | Annotate ≥25% visible objects; diverse capture angles | CV Engineer |
| R05 | **Hardware limitations** — CPU-only mobile, low RAM | 4 | 4 | 16 | YOLO11n nano model; INT8 quantization; VLM optional; graceful degradation | Edge Engineer |
| R06 | **False alarms** — alert fatigue causing user to ignore alerts | 5 | 4 | 20 | Per-rule cooldown timers; severity-based filtering; confidence thresholds | Product Manager |
| R07 | **False negatives** — missed critical hazards | 5 | 3 | 15 | Prioritize recall over precision for safety classes; target ≥ 0.80 recall | ML Architect |
| R08 | **Thermal throttling** — device overheats under continuous load | 4 | 3 | 12 | Frame-skip strategy; VLM every 5th frame; thermal monitoring | Edge Engineer |
| R09 | **Camera failure** — camera feed lost or corrupted | 4 | 2 | 8 | Watchdog thread; auto-restart; fallback TTS error message | Backend Engineer |
| R10 | **TTS failure** — audio output fails silently | 4 | 2 | 8 | TTS health check on startup; fallback to system beep | Backend Engineer |
| R11 | **SmolVLM2 failure** — VLM OOM or crash | 3 | 3 | 9 | Feature flag to disable VLM; rule-only fallback mode | Edge Engineer |
| R12 | **Memory overflow** — RAM exhaustion on edge device | 4 | 3 | 12 | Memory profiling; sliding event window; memory budget tests | Backend Engineer |
| R13 | **Label quality** — poorly annotated training data | 5 | 3 | 15 | Dual-annotator workflow; automated QA scripts; annotation guidelines | ML Architect |
| R14 | **Annotation inconsistency** — different annotators label differently | 4 | 4 | 16 | Inter-annotator agreement scoring; class definition document | MLOps Engineer |
| R15 | **Domain shift** — lab data does not match real Indian home conditions | 4 | 4 | 16 | Mandatory Indian-home custom capture; field testing in 3+ homes | CV Engineer |
| R16 | **Battery usage** — continuous camera+AI drains battery rapidly | 3 | 4 | 12 | Configurable frame rate; sleep mode when no motion detected | Edge Engineer |
| R17 | **Privacy concerns** — camera in home raises ethical questions | 5 | 3 | 15 | Privacy-by-design; no cloud upload; explicit consent; no facial recognition | Security Engineer |
| R18 | **Model drift** — accuracy degrades over time | 3 | 3 | 9 | Active learning pipeline; periodic model refresh; data drift monitoring | MLOps Engineer |
| R19 | **Dependency failures** — library updates break pipeline | 3 | 3 | 9 | Pin all dependency versions; requirements.txt lock | DevOps Engineer |
| R20 | **Gas cylinder misclassification** — industrial vs Indian LPG | 5 | 3 | 15 | Mandatory Indian LPG brand images; negative hard mining | CV Engineer |
| R21 | **Wet floor detection** — shiny vs wet surface confusion | 5 | 4 | 20 | Specialized augmentation; context rules near sink or bathroom | CV Engineer |
| R22 | **Stove at night** — gas flame as brightness spike | 4 | 3 | 12 | Collect night/dim kitchen captures; brightness-normalized augmentation | CV Engineer |
| R23 | **Medicine strip confusion** — vs food packaging | 4 | 3 | 12 | Diverse medicine brand captures; food negative examples | CV Engineer |
| R24 | **wet_floor taxonomy risk** — may be ill-posed as a bounding-box class (amorphous, low-texture region, confusable with dry shiny marble) | 4 | 3 | 12 | Tightened annotation protocol (paired wet/dry negatives); measurable pilot gate — first ~50-image dual-annotated session must reach IAA ≥ 0.60 or the class is demoted to scene-level (capture_annotation_runbook.md §8); second checkpoint at Phase-5 baseline (AP50 < 0.30 reopens demotion) | CV Engineer |
| R25 | **Ultralytics internal-API drift** — a release within the >=8.3,<9.0 pin moves the v8DetectionLoss seams the masked loss wraps | 4 | 3 | 12 | Runtime source canary (`assert_ultralytics_compat`, preflight G5) + CI drift test on every matrix leg; fail-loud pre-existing-criterion check at on_train_start; validated version recorded in docs/06 architecture doc | ML Architect |
| R26 | **Masked-loss correctness regression** — silent no-op or over-masking corrupts supervision | 5 | 2 | 10 | Bit-identity + exact-zero-gradient unit tests; M3.5 validation gate (committed report) re-run on any loss/trainer change; per-epoch mask-stat logging; golden train-kwargs regression pins the disabled path | ML Architect |
| R27 | **Completeness artifact staleness** — dataset rebuilt but artifact not regenerated | 3 | 4 | 12 | Input-hash freshness gate G7 + coverage gate G3 fail training with the exact `dvc repro generate_completeness` remediation; DVC stage lineage (train stage depends on the artifact) | MLOps Engineer |
| R28 | **Mitigation forgoes mosaic/mixup** — strict G8 policy may cost augmentation benefit at full scale | 3 | 3 | 9 | Explicit product decision (ADR-P4-04) with `warn`/`ignore` escape hatches; benchmark isolates the comparison by zeroing augs in both arms; revisit with a mask-aware mosaic in Phase 5 if deltas warrant | CV Engineer |
| R29 | **Smoke-scale benchmark noise** — conclusions from 188 images do not generalize | 3 | 4 | 12 | Reports carry an explicit "not generalizable" banner; repeats ± std reported; deterministic seeds documented; full-scale A/B re-run is a Phase-5 gate before any production claim | MLOps Engineer |
| R30 | **Auto-annotator hallucination** — open-vocab candidates flood human verification with junk | 4 | 3 | 12 | Per-class confidence thresholds (configs/annotation.yaml); L2 scope restricted to detectable classes (custom-capture classes are never prompted); estimator-precision calibration from verified batches feeds threshold tuning; expected-gain batch ranking | ML Architect |
| R31 | **CVAT round-trip corruption** — class-order drift, partial exports, or accidental edits to trusted source labels | 5 | 2 | 10 | Reused `verify_class_order` importer; generated `cvat_labels.json` task spec (taxonomy order enforced at task creation); pinned self-hosted CVAT version; non-target-class byte-equality hard-fail on import; batch lifecycle statuses | Dataset Engineer |
| R32 | **Ledger/label drift** — verified_labels edited without a matching ledger entry | 4 | 2 | 8 | Preflight gate G9 (sha256 + per-class box-count cross-check); ledger and verified_labels are outs of ONE frozen DVC stage; ledger is git-tracked (cache: false) for diffable audit | MLOps Engineer |
| R33 | **GPU/driver nondeterminism** — candidate artifacts not reproducible across environments | 3 | 3 | 9 | Environment (torch/CUDA/driver) + weights sha256 + prompt fingerprint recorded in every candidates artifact; `--verify-determinism` re-run diff; coverage report derives from the pinned artifact, never re-runs inference (ADR-P5-06) | ML Architect |
| R34 | **OneDrive sync corruption** — sync races against large DVC builds/cache | 4 | 3 | 12 | DVC remote relocated off the OneDrive tree (C:\dvc_remote); full-build preflight FB2/FB5 gates warn/fail on OneDrive paths; cache relocation runbook (`dvc cache dir` + `cache.type hardlink,copy`) before the full build | MLOps Engineer |
| R35 | **Disk exhaustion at full scale** — 15–30k images + DVC cache copies overflow the drive | 4 | 3 | 12 | `scripts/qa/full_build_preflight.py` FB1 hard gate (≥150 GB free, override only with recorded decision); FB2 checks remote free space | MLOps Engineer |
| R36 | **Verification throughput shortfall** — human CVAT capacity slips the v0.7 release | 3 | 4 | 12 | Release thresholds live in configs/release.yaml (adjustable only with a recorded decision); expected-gain prioritized batches maximize value per hour; cumulative progress tracked in the quality report | Product Manager |
| R37 | **transformers/GroundingDINO drift** — optional backend breaks under dependency movement | 2 | 3 | 6 | `hf_revision` commit pin; backend disabled by default (enabled only on measured yolo_world precision < 0.4); canary import test behind importorskip | ML Architect |
| R38 | **License contamination via Roboflow slugs** — a slug's license is incompatible with the release track | 5 | 2 | 10 | Human track H-B reviews every slug license BEFORE it enters configs/dataset_sources.yaml (no download of unreviewed data); release gate RG7 blocks on missing/incompatible license entries; per-slug licenses recorded in the release manifest | Product Manager |

> [!CAUTION]
> **R01 (Poor Lighting), R03 (Dataset Imbalance), R06 (False Alarms), R21 (Wet Floor)** have the highest combined risk scores (20+) and require immediate mitigation planning before Phase 3 data collection begins.
>
> **R25–R29 (Phase-4)** are engineering-controlled risks with automated gates (preflight G1–G8, CI canaries, benchmark budgets); none require human data-collection action.
>
> **R30–R38 (Phase-5)** cover the production dataset pipeline: auto-annotation quality (R30, R33, R37), the human CVAT verification loop (R31, R32, R36), operational storage (R34, R35), and licensing (R38). R34/R35 are gated by `scripts/qa/full_build_preflight.py` (FB1–FB6); R31/R32 by the import hard-fails + preflight G9; R38 by release gate RG7.
>
> **R24 status:** gate defined, evidence pending the first wet_floor pilot session (Phase-5 human track H-A). Decision checkpoint is release v0.9.0 (`wet_floor_decision_required` in configs/release.yaml) — see `docs/04_dataset_engineering/capture_annotation_runbook.md` §8 for the protocol.

---

Previous: [implementation_phases.md](./implementation_phases.md)

Next: [validation_strategy.md](./validation_strategy.md)

Related: [security_privacy.md](./security_privacy.md)
