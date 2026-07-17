# Mitigation Runbook — Missing Annotation Mitigation (Phase 4)

> Operational guide for the masked-loss training pipeline. Architecture and
> rationale: [masked_loss_architecture.md](masked_loss_architecture.md).
> All commands run from the repo root inside the project venv.

## 1. Generate / refresh completeness metadata

```bash
dvc repro generate_completeness        # normal path (after any dataset change)
# or directly:
python scripts/dataset/11_generate_completeness.py
```

Outputs: `data/processed/completeness.json` (DVC-tracked) + report triplet
`data/qa_reports/completeness_report.{json,csv,md}`. The generator hard-fails
on any ambiguity — fix the named config/manifest issue, never hand-edit the
artifact. Note: the first `dvc repro` after a code change may also refresh
other stale stages (expected DVC behavior).

## 2. Check readiness (preflight, on demand)

```bash
python scripts/training/preflight_check.py                 # gates G1–G8
python scripts/training/preflight_check.py --json-out reports/preflight.json
```

Exit codes: `0` pass · `1` failure · `2` warnings only. The same gates run
automatically inside `train_yolo.py` whenever mitigation is enabled.

## 3. Train

```bash
# Stock training (byte-for-byte pre-Phase-4 behavior):
python scripts/training/train_yolo.py

# Mitigated training (config flag or CLI override):
python scripts/training/train_yolo.py --mitigation on
```

Enabling in config instead: set `missing_annotation_mitigation.enabled: true`
in `configs/training/yolo11n_config.yaml` **and** zero the mixing
augmentations (`augmentation.mosaic/mixup/copy_paste: 0.0`) or preflight G8
fails (strict by design — see ADR-P4-04).

**Resume:** `--resume` works with mitigation, but the `--mitigation on` flag
(or config setting) must be passed again — the trainer class is never stored
in checkpoints. Resuming without it silently trains stock; the preflight log
lines at startup are your confirmation that masking is active.

**Housekeeping:** Ultralytics writes `data/processed/labels/*.cache` during
training. `validate_masking.py` and the benchmark clean these automatically;
after manual runs remove them (or `dvc status` will report the labels out as
changed):

```bash
rm data/processed/labels/*.cache
```

## 4. Validate masking correctness (M3.5 gate)

Re-run after any change to the loss, trainer, lookup, or an Ultralytics
version bump:

```bash
python scripts/training/validate_masking.py            # full gate incl. 2 training arms
python scripts/training/validate_masking.py --skip-training
RUN_TRAINING_SMOKE=1 pytest tests/system/test_training_smoke.py -m system
```

Report: `data/qa_reports/phase4_mitigation/masking_validation_report.{json,md}`
(committed). Exit 1 = do not proceed to evaluation/benchmarking.

## 5. Benchmark baseline vs mitigated

```bash
python scripts/training/benchmark_mitigation.py --smoke          # 2 epochs, 2 repeats
python scripts/training/benchmark_mitigation.py --epochs 3 --repeats 3
```

Trains both arms (mixing augs zeroed in both — fair comparison), measures
speed/memory/metrics + loss microbenchmarks, and marks every performance
budget PASS/FAIL. Weights land in `models/benchmarks/` (gitignored); the
report triplet in `data/qa_reports/phase4_mitigation/benchmark_report.*`
(committed). Exit 1 = a budget was breached — investigate before merging.

## 6. Evaluate checkpoints

```bash
python scripts/training/evaluate_mitigation.py \
    --baseline-weights  models/benchmarks/models/baseline_r0/weights/best.pt \
    --mitigated-weights models/benchmarks/models/mitigated_r0/weights/best.pt \
    --imgsz 320
```

Writes `evaluation_comparison.{json,csv,md}` + per-arm confusion matrices.
Read per-class **deltas**, not absolute numbers (partial-annotation caveat in
the report header).

## 7. Troubleshooting by gate ID

| Symptom | Cause | Fix |
|---|---|---|
| G1 FAIL: artifact not found / invalid JSON | stage never ran or file corrupt | `dvc repro generate_completeness` |
| G2 FAIL: taxonomy drift | configs/data.yaml changed after generation | re-run the stage; if the taxonomy change is intentional, rebuild remap/merge first |
| G3 FAIL: images without records / split mismatch | data/processed changed after generation | `dvc repro` (split + completeness) |
| G3 WARN: stale records | images removed from disk | regenerate when convenient |
| G4 FAIL: orphan refs / invalid ids / duplicates | corrupt or hand-edited artifact | regenerate; never hand-edit |
| G4 WARN: unused policies | e.g. session fully deduplicated at merge | informational |
| G5 FAIL: version window / canary | ultralytics changed internals | pin to the validated version (architecture doc header) or update `src/training/masked_loss.py` markers after review |
| G6 FAIL: config invalid | bad key/value in the mitigation section | the message names the key and accepted values |
| G7 FAIL: input hashes changed | merged manifest / split summary rebuilt | `dvc repro generate_completeness` |
| G8 FAIL: mixing augmentations active | mosaic/mixup/copy_paste > 0 (or NO augmentation section — Ultralytics defaults mosaic=1.0) | zero them in the training config, or relax `mixing_augmentation_policy` (understand ADR-P4-04 first) |
| Generator: "unsupported datasets are rejected" | new source without a completeness policy | add it under `completeness.policies` in configs/dataset_sources.yaml (mode per ADR-P4-05) |
| Generator: "drift between config and manifest" | trusted_classes edited after merge | `dvc repro merge_datasets` or reconcile the config |
| Loss: UnknownImageError at train time | artifact stale vs dataloader contents | regenerate; `on_unknown_image: warn_full_supervision` is the (discouraged) escape hatch |
| RuntimeError: model already has a criterion | Ultralytics trainer flow changed | do NOT proceed; see architecture doc §compatibility |

## 8. Adding a new data source (checklist)

1. Add the source + `trusted_classes` under `sources:` (Phase-2 contract).
2. Assign its completeness mode under `completeness.policies`
   (new semantics ⇒ new provider, see ADR-P4-05).
3. `dvc repro` → `python scripts/training/preflight_check.py` → green.

---
*Previous:* [masked_loss_architecture.md](masked_loss_architecture.md) ·
*Related:* [phase4_engineering_report.md](phase4_engineering_report.md) ·
[capture & annotation runbook](../04_dataset_engineering/capture_annotation_runbook.md)
