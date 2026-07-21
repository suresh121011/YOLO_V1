# 07 — Production Dataset Engineering (Phase 5)

## Purpose

Phase-5 documentation home: missing-annotation **resolution** (as opposed to
the Phase-4 loss-level **mitigation**), the auto-annotation + human
verification pipeline, coverage estimation, and the versioned release ladder
that produces **Dataset v1.0** for Phase-6 training.

Approved plan of record: council-reviewed Phase-5 implementation plan
(milestones M0–M11, human tracks H-A/H-B/H-C). Status lives in `CHANGELOG.md`.

## The core invariant

**Auto-generated labels never touch `labels/` directly.** Candidates flow:

```
data/merged ──12_auto_annotate──▶ data/annotation/candidates/<backend>/
        (GPU, sha256-pinned weights; skips trusted/verified cells)
                    │
                    ▼ 13_build_verification_batches (frozen stage)
        data/annotation/batches/<batch_id>/  (CVAT pre-annotation zips)
                    │
        ══ HUMAN: self-hosted CVAT — correct/verify (10% IAA sample) ══
                    │
                    ▼ 14_import_verified_batch (frozen stage)
        data/annotation/verification_ledger.json   (git-tracked audit trail)
        data/annotation/verified_labels/*.txt      (delta boxes only)
                    │
                    ▼ 15_apply_verified_labels
        data/merged_verified/labels   (base ∪ deltas overlay; images NOT copied)
                    │
                    ▼ split → generate_completeness → qa_check
        masking (Phase-4) shrinks exactly as verification grows
```

Missing-annotation resolution levels: **L1** human annotation (Phase-3 CVAT
flow + verification batches) · **L2** model-assisted pre-labeling with
mandatory human review · **L3** cross-dataset label salvage (exact-sha256
twins at merge; near-dups become candidates) · **L4** per-image coverage
estimation (`16_coverage_report.py` — pure arithmetic over the pinned
candidates artifact, no inference at report time) · **L5** dataset quality
report with residual-risk quantification (`17_dataset_quality_report.py`).

## Release ladder (configs/release.yaml, gates RG1–RG10)

| Release | Contents |
|:---|:---|
| `dataset-v0.5.0` | Full-mode public build, masked-only mitigation, first `dvc push` verified |
| `dataset-v0.7.0` | + ≥3,000 human-verified (image, class) cells on public images |
| `dataset-v0.9.0` | + custom captures (≥1,000 imgs, ≥2 houses), locked eval set, house-level split, wet_floor R24 decision |
| `dataset-v1.0.0` | All targets (≥2,000 custom, ≥200 inst/class, ≥3 houses), 0 QA criticals, full-scale A/B evidence — unfreezes `train_yolo11n` |

## Documents

| Document | Contents |
|:---|:---|
| `README.md` | This index |
| `auto_annotation_runbook.md` | (M1) Operate `12_auto_annotate.py`: weights pinning, prompts, determinism verification |
| `verification_runbook.md` | (M2) CVAT task creation → pre-annotation upload → export → import → IAA → `dvc commit -f` |
| `release_runbook.md` | (M5) `18_make_release.py check/make/verify`, gate remediation, tag + push procedure |
| `phase5_engineering_report.md` | (M11) Final report: what shipped, evidence, limitations, Phase-6 readiness |
| [`adr/`](adr/README.md) | ADR-P5-01 … ADR-P5-12 — the twelve ratified design decisions (council PASS 2026-07-17) |

## Operational prerequisites

- **DVC remote** — default `localstore` → `C:\dvc_remote` (off the OneDrive
  tree; closes audit risk C-1). First push completed at M0 (584 objects).
  Migration to a second physical disk / S3: copy the folder, update one
  config line, `dvc push`.
- **Full-build preflight** — run `python scripts/qa/full_build_preflight.py`
  before any `mode: full` build (gates FB1–FB6: disk, remote, Roboflow,
  GPU, OneDrive hazard, mode). Report triplet:
  `data/qa_reports/full_build_preflight.{json,csv,md}`.
- **GPU note (M0 finding, resolved at M1):** the M0 finding was a CPU-only
  torch build resolving from a bare `python`/`dvc` on `PATH` — a *different*
  interpreter than the project's own `.venv`, which already carries a
  CUDA-enabled torch matching the local NVIDIA GPU. Always invoke tooling
  via `.venv/Scripts/python.exe` explicitly (or prepend it to `PATH`) —
  see `auto_annotation_runbook.md` §0 for the verification command.
- **Self-hosted CVAT** — required from the M2 verification drill onward
  (privacy: custom-capture images never leave the machine); version pinned
  in `verification_runbook.md` §0.

## Key risks

R30–R38 in `docs/01_executive_implementation_plan/risk_register.md`
(auto-annotation quality, CVAT round-trip integrity, ledger drift, OneDrive/
disk operations, verification throughput, licensing).

---

Previous: [../06_training_engineering/README.md](../06_training_engineering/README.md)
