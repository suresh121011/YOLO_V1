# Phase-4 Engineering Report — Missing Annotation Mitigation

> **Status:** ✅ Complete — mitigation implemented, validated (M3.5 gate PASS),
> benchmarked (all performance budgets PASS), fully opt-in, zero regressions.
> Branch `phase-4-missing-annotation-mitigation`; validated against
> ultralytics 8.4.96 / torch 2.13.0+cpu / Python 3.14 (local) and the
> py3.10/3.12 × ubuntu/windows CI matrix.

## 1. Objectives

Public datasets annotate only a subset of the 23-class taxonomy — COCO labels
`person` but not `face`, WIDER FACE labels only `face` — and YOLO's BCE
classification loss treats every unlabeled class as background at every
anchor. On the smoke dataset the mean image trusts just **6.25/23 classes**:
~73 % of all (image, class) supervision cells carried false "push to
background" signal. Phase 4 eliminates exactly that error:

1. per-image completeness metadata as a reproducible DVC artifact (M1),
2. fail-early training preflight gates (M2),
3. a masked BCE classification loss injected via `model.train(trainer=...)`
   without modifying Ultralytics source (M3),
4. a dedicated correctness-validation gate (M3.5),
5. evaluation + benchmark frameworks with explicit performance budgets
   (M4/M5),
6. this documentation set with ADRs (M6).

Hard constraints held throughout: **strictly optional** (with
`missing_annotation_mitigation.enabled: false` training is byte-for-byte the
pre-Phase-4 pipeline), box/DFL losses untouched, no Ultralytics fork.

## 2. Implemented features

| Area | What shipped | Where |
|---|---|---|
| Completeness policies | Explicit per-source policy modes (`trusted_list` / `verified_absence_all` / `per_session`) resolving the `trusted_classes: []` ambiguity (negatives = all-ones mask; captures = per session) | `configs/dataset_sources.yaml` (top-level `completeness:`), [ADR-P4-03](adr/ADR-P4-03-completeness-artifact.md) |
| Policy provider registry | Pluggable providers; new dataset types register without touching the generator | `src/dataset/completeness_policies.py`, [ADR-P4-05](adr/ADR-P4-05-policy-provider-registry.md) |
| Completeness artifact | Hard-fail builder + validator (orphans, duplicates, drift, unknown images all fatal; unused policies warn), taxonomy fingerprint, input hashes | `src/dataset/completeness.py`, `scripts/dataset/11_generate_completeness.py` |
| DVC integration | `generate_completeness` stage (split → completeness → frozen train dep); stale H-2 lock refreshed en route | `dvc.yaml`, `dvc.lock` |
| Mitigation config | Frozen dataclass, house `with_overrides()` pattern, absent section ⇒ disabled | `src/training/mitigation_config.py`, training YAMLs |
| Preflight gates G1–G8 | Artifact/taxonomy/coverage/consistency/environment/config/freshness/augmentation gates; standalone CLI (exit 0/1/2) | `src/training/preflight.py`, `scripts/training/preflight_check.py` |
| Masked BCE loss | `_MaskingBCE` wrapper (no upstream math copied), in-place masked multiply, per-policy row-tensor cache, mask statistics | `src/training/_masked_loss_impl.py`, [ADR-P4-01](adr/ADR-P4-01-loss-level-masking.md) |
| Trainer injection | `build_masked_trainer` factory; criterion attached at `on_train_start` to train + EMA models; model class stays stock (portable checkpoints); DDP rejected | `src/training/trainer.py`, [ADR-P4-02](adr/ADR-P4-02-trainer-injection.md) |
| Training CLI | `--mitigation on\|off`; preflight before any Ultralytics work; `metrics.json` mitigation block only when enabled | `scripts/training/train_yolo.py` |
| Compat canary | `assert_ultralytics_compat()` source-marker check (gate G5 + CI test on every matrix leg) | `src/training/masked_loss.py` |
| M3.5 validation gate | Re-runnable correctness runner + env-gated system smoke test | `scripts/training/validate_masking.py`, `tests/system/test_training_smoke.py` |
| Evaluation framework | Per-class P/R/F1/mAP, confusion matrices, mitigated−baseline deltas | `src/training/evaluation.py`, `scripts/training/evaluate_mitigation.py` |
| Benchmark framework | A/B runs with repeats, process-tree RSS, interleaved-median microbenchmarks, per-budget PASS/FAIL verdict | `src/training/benchmark.py`, `scripts/training/benchmark_mitigation.py` |

## 3. Backward-compatibility proof (three layers)

1. **Kwargs layer** — `build_train_kwargs` was extracted verbatim; a golden
   regression test pins its disabled-path output to the exact pre-Phase-4
   dict (keys, values, order; never a `trainer` key)
   (`tests/unit/test_train_kwargs_regression.py`).
2. **Tensor layer** — an all-ones mask produces a **bit-identical** loss to
   stock `v8DetectionLoss` (`torch.equal`), and masked cells receive
   **exactly zero** gradient (`tests/unit/test_masked_loss.py`).
3. **Behavior layer** — the M3.5 disabled-path training run shows no
   preflight, no trainer injection, no mitigation traces, unchanged
   `metrics.json` shape
   ([validation report](../../data/qa_reports/phase4_mitigation/masking_validation_report.md)).

When disabled, `src.training`'s torch-dependent modules are never imported.

## 4. Validation evidence (M3.5 gate — PASS)

Committed at
[`data/qa_reports/phase4_mitigation/masking_validation_report.md`](../../data/qa_reports/phase4_mitigation/masking_validation_report.md):

- correctness unit suite green (bit-identity, zero gradients, injection,
  golden kwargs);
- real-artifact spot-checks: 188/188 images — coco trusts 10/23,
  openimages 3/23, wider_face 1/23, **negatives 23/23 (all-ones)**;
- mitigated 1-epoch run: preflight ran, mitigation announced, mask stats
  logged, finite losses, weights + metrics block written;
- disabled 1-epoch run: stock path clean.

## 5. Benchmark results (smoke scale — indicative, NOT generalizable)

Committed at
[`data/qa_reports/phase4_mitigation/benchmark_report.md`](../../data/qa_reports/phase4_mitigation/benchmark_report.md)
(3 epochs @ 320 px, batch 8, 3 repeats per arm, CPU; mixing augmentations
zeroed in both arms per [ADR-P4-04](adr/ADR-P4-04-strict-mixing-aug-policy.md);
deterministic seeds ⇒ metric variance across repeats ≈ 0 by design):

| Performance budget | Measured | Limit | Status |
|---|---|---|---|
| Training wall-time overhead (binding budget) | ≈ 0 % (see noise note) | ≤ 5 % | PASS |
| Peak CPU RSS delta | +0.55 % / +5.9 MB | ≤ 5 % / ≤ 200 MB | PASS |
| Peak GPU memory delta | n/a (no CUDA on this machine) | ≤ 5 % | N/A |
| Loss-forward overhead per call (interleaved median, bs=16) | +0.077 ms (1.9 % of the isolated call; ≈ 0 % of a training step) | ≤ 1 ms | PASS |
| Per-batch mask build (bs=16) | 0.050 ms | ≤ 1 ms | PASS |
| CompletenessLookup.load (188 images) | 0.002 s | ≤ 2 s | PASS |

Measurement notes. (1) The loss-forward budget was re-specified during M5
from a percentage of the isolated loss call to an absolute per-call ceiling —
the masked multiply is O(bs·anchors·nc), so the percentage form measured
"how fast is the rest of the loss call" rather than user-visible cost
(rationale in the [architecture doc](masked_loss_architecture.md)). The same
investigation optimized the hot path (in-place multiply, per-policy
row-tensor cache, stats behind `log_mask_stats`): overhead dropped from
≈ 0.6 ms to 0.077 ms per call. (2) In the committed run the baseline arms
measured ~25 % SLOWER than the mitigated arms — arms run sequentially and
post-wake background load hit the baseline arms first; quiet-system runs
measured the two arms within ±1 % (43.8 s vs 43.1 s). The honest reading is
"no measurable end-to-end overhead", not a speedup.

Accuracy columns at 3 epochs on 188 images are statistically meaningless
(the report banner says so explicitly); the deliverable here is the
framework plus the overhead validation. Measurement note: single-series
wall-clock microbenchmarks proved unreliable on desktop hardware (0.5–10 %
swings between identical runs) — the framework therefore interleaves
stock/masked rounds and reports the median, and the masked hot path was
optimized (in-place multiply, per-policy row-tensor cache, stats gated
behind `log_mask_stats`).

## 6. Testing

| Layer | Coverage |
|---|---|
| Unit (pure Python) | policies/registry (22), generator/validator (29), mitigation config (12), preflight gates (28), evaluation shaping (9), benchmark math (14) |
| Unit (torch/ultralytics, CI drift canaries) | masked BCE + criterion (14), trainer factory/attach (9), incl. bit-identity + zero-gradient proofs against the installed version |
| Regression | golden train-kwargs suite (8) — the disabled-path byte-identity guard |
| Integration | real shipped configs vs pipeline contracts + synthetic E2E with the live 23-class taxonomy (8) |
| System (env-gated, never CI) | `RUN_TRAINING_SMOKE=1` one-epoch mitigated training |

Full suite: **565 tests** in the CI scope (412 pre-Phase-4 → +153, plus the
env-gated system smoke test), black/ruff/mypy clean (mypy scope extended
with `src/training`), coverage 74.99 % (pre-phase baseline 73.35 %,
floor 40 %).

## 7. Risks & limitations

Register entries R25–R29 added
([risk register](../01_executive_implementation_plan/risk_register.md)):
Ultralytics API drift (canary-gated), masked-loss correctness (M3.5 gate),
artifact staleness (G7), mixing-augmentation trade-off (G8 strict,
ADR-P4-04), smoke-scale benchmark noise (banner + Phase-5 re-run).
Known limitations (detailed in the
[architecture doc](masked_loss_architecture.md)): no mosaic/mixup under
mitigation, no DDP, untrusted classes get no negative supervision from
untrusted sources, public-split metrics underestimate untrusted classes.

## 8. Readiness for Phase 5

- Training-side machinery for partially-annotated data is complete and
  gated; the full-dataset build (`mode: full`) plugs in without code changes.
- Custom-capture sessions integrate via the `per_session` policy — the
  completeness pipeline already consumes finalized session manifests.
- Open operational items carried forward (unchanged from the audit): first
  `dvc push` (C-1) before real capture collection; Roboflow slugs; wet_floor
  pilot (R24). Phase-5 additions: full-scale A/B benchmark + evaluation on
  the fully-annotated custom eval set before any production accuracy claim.

---
*Related:* [masked_loss_architecture.md](masked_loss_architecture.md) ·
[mitigation_runbook.md](mitigation_runbook.md) ·
[ADR index](README.md) ·
*Previous phase:* [Phase-3 report](../04_dataset_engineering/phase3_engineering_report.md)
