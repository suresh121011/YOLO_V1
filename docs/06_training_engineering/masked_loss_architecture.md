# Masked-Loss Architecture — Missing Annotation Mitigation (Phase 4)

> **Status:** Implemented and validated (M3.5 gate PASS, see
> [masking validation report](../../data/qa_reports/phase4_mitigation/masking_validation_report.md)).
> **Ultralytics compatibility:** pinned `>=8.3,<9.0`; developed and validated
> against **8.4.96** with torch 2.13.0 (CPU). Drift is caught by
> `assert_ultralytics_compat()` + the CI canary test.

## Purpose

Public datasets annotate only a subset of the 23-class taxonomy. Stock YOLO
training treats every unlabeled class as background at every anchor —
systematic false supervision (on the smoke dataset, a mean image trusts only
6.25/23 classes). Phase 4 removes exactly that signal, at the loss level,
opt-in, without modifying Ultralytics source.

## Component map

```
configs/dataset_sources.yaml        configs/data.yaml
   completeness.policies                 taxonomy (nc=23)
        │                                    │
        ▼                                    ▼
scripts/dataset/11_generate_completeness.py          (DVC: generate_completeness)
   └─ src/dataset/completeness.py  ── uses ──  src/dataset/completeness_policies.py
        │                                        (provider registry: trusted_list /
        ▼                                         verified_absence_all / per_session)
data/processed/completeness.json
        │
        ▼ (only when missing_annotation_mitigation.enabled)
scripts/training/train_yolo.py
   ├─ src/training/preflight.py        gates G1–G8 (fail-early)
   ├─ src/training/completeness_lookup.py   basename → {0,1}^nc mask row
   └─ src/training/trainer.py          build_masked_trainer → trainer=...
          └─ on_train_start: model.criterion = MaskedDetectionLoss
                └─ src/training/_masked_loss_impl.py (_MaskingBCE wrapper)
```

## The masking mechanism

Stock `v8DetectionLoss` computes (8.4.96, `get_assigned_targets_and_loss`):

```python
bce_loss = self.bce(pred_scores, target_scores.to(dtype))   # (bs, anchors, nc)
loss[1]  = bce_loss.sum() / target_scores_sum
```

with `self.bce = nn.BCEWithLogitsLoss(reduction="none")`. We do **not** copy
that method. `MaskedDetectionLoss(v8DetectionLoss)` swaps `self.bce` for a
`_MaskingBCE` wrapper; `__call__` builds a `(bs, 1, nc)` {0,1} mask from
`batch["im_file"]` → `CompletenessLookup`, installs it on the wrapper,
delegates to the stock `__call__`, and clears it. The wrapper multiplies the
elementwise BCE map by the mask (broadcast over anchors) and **fails loud** if
a mask is installed but the tensor shape is not the expected `(bs, anchors,
nc)` — silent skipping would corrupt supervision.

### Identity proof (backward-compatibility contract)

- Multiplying by 1.0 is exact in IEEE-754 ⇒ an **all-ones mask produces a
  bit-identical classification loss** (asserted with `torch.equal` in
  `tests/unit/test_masked_loss.py`).
- The normalizer `target_scores_sum` derives from assigner **targets**, which
  are never touched — untrusted classes have zero targets by definition, so
  masking cannot change the denominator.
- Box/DFL losses depend only on assigner outputs (`fg_mask`,
  `target_bboxes`), computed from predictions and ground truth — untouched
  (asserted equal in the A/B loss test).
- Gradient guarantee: masked (image, class) cells contribute **exactly zero**
  gradient to the classification head (asserted on `pred.grad`).
- When **disabled**, the guarantee is stronger: no `trainer=` kwarg, no
  torch import from this package, and `build_train_kwargs` output is pinned
  byte-identical by a golden regression test.

### Mask semantics per policy

| Policy mode | Trusted set | Mask row |
|---|---|---|
| `trusted_list` (coco, openimages, wider_face, roboflow) | source's exhaustive classes | 1 for trusted, 0 elsewhere |
| `verified_absence_all` (negatives) | ALL classes (verified absent) | all-ones |
| `per_session` (custom captures) | session manifest's trusted classes | per session |

Negatives keep full supervision — their empty labels are *genuine* negatives,
which is precisely why they exist.

## Injection flow (why the model class stays stock)

See [ADR-P4-02](adr/ADR-P4-02-trainer-injection.md). Checkpoints pickle the
EMA model object by class reference, so subclassing the model would couple
every `.pt` to this repository. Instead the configured trainer subclass
(returned by `build_masked_trainer`) attaches the criterion at
`on_train_start` — after `set_model_attributes` (hyperparameters) and EMA
creation, strictly before the first batch — pre-empting the lazy
`init_criterion()`. The EMA model receives its own criterion because the
validator computes val loss through it (val-loss curves stay comparable).
`save_model` strips `criterion` before serializing (verified), so checkpoints
stay clean and portable. Resume re-passes the trainer class (handled by
`train_yolo.py` whenever mitigation is enabled).

## Preflight gates (G1–G8)

Run only when mitigation is enabled (disabled path stays untouched); also
available standalone via `scripts/training/preflight_check.py` (exit 0/1/2).

| Gate | Checks | On failure |
|---|---|---|
| G1 | artifact exists, parses, schema supported | FAIL |
| G2 | taxonomy fingerprint == live configs/data.yaml | FAIL |
| G3 | every train/val image covered, splits match; stale records | FAIL / WARN |
| G4 | self-consistency: modes registered, ids in range, orphan refs, duplicates; unused policies | FAIL / WARN |
| G5 | ultralytics importable, version window, loss-surface source canary | FAIL |
| G6 | mitigation config valid | FAIL |
| G7 | recorded input hashes match disk (freshness) | FAIL |
| G8 | mixing augmentations vs policy ([ADR-P4-04](adr/ADR-P4-04-strict-mixing-aug-policy.md)) | FAIL (default) |

Every diagnostic names the offending key/file and the remediation command.

## Performance budgets

Enforced by the benchmark (`scripts/training/benchmark_mitigation.py`), each
marked PASS/FAIL in the report; a breach fails the benchmark verdict.

| Metric | Budget |
|---|---|
| Training wall-time overhead (mitigated vs baseline) | ≤ 5 % — **the binding user-facing budget** |
| Peak CPU RSS delta | ≤ 5 % and ≤ 200 MB |
| Peak GPU memory delta (when CUDA present) | ≤ 5 % |
| Loss-forward overhead per call (microbenchmark, bs=16) | ≤ 1 ms absolute |
| Per-batch mask build (bs=16, cached policy rows) | ≤ 1 ms |
| `CompletenessLookup.load` | ≤ 2 s |

Budget-specification note: the loss-forward budget was originally drafted as
"≤ 3 % of the isolated loss call" and re-specified during M5 as an absolute
per-call ceiling. The masked multiply is O(bs·anchors·nc) — the same order as
the BCE map itself — so its *percentage* of the (small, CPU-timed) isolated
loss call is dominated by how fast the rest of that call happens to be, and
single-series wall-clock timing swung 0.5–10 % between identical runs. The
measured truth: ≈ 0.6 ms/call at bs=16 on CPU, which is ≈ 0.2 % of a full
training step — consistent with the end-to-end wall-time budget measuring
≈ 0 % overhead. The microbenchmark now interleaves stock/masked rounds and
reports the median. Smoke-scale numbers are indicative; budgets are
re-validated at full scale in Phase 5. Measured results: committed
[benchmark report](../../data/qa_reports/phase4_mitigation/benchmark_report.md).

## Known limitations

1. **Mixing augmentations** are forbidden under mitigation (G8, default
   `forbid`) — composited samples expose only the primary image's file. See
   ADR-P4-04 for the escape hatches and the Phase-5 revisit plan.
2. **DDP is unsupported** (configured trainer class cannot be reconstructed
   in worker processes) — single-device training only.
3. **Untrusted classes get no negative supervision from untrusted sources** —
   inherent to masking; verified negatives and each class's trusted sources
   carry that signal instead (ADR-P4-01).
4. **Validation metrics on public splits underestimate untrusted classes**
   (unlabeled-but-detected objects count as false positives). Per-class
   baseline-vs-mitigated *deltas* remain meaningful; unbiased absolute
   numbers require the fully-annotated custom eval set (Phase 5).
5. **EMA checkpoint saves deep-copy the attached criterion before stripping
   it** — a few MB of transient memory per save at full scale; harmless.

## Compatibility contract

`assert_ultralytics_compat()` verifies at runtime (and in CI on every matrix
leg) that `v8DetectionLoss` still constructs
`BCEWithLogitsLoss(reduction="none")` and routes the cls loss through
`self.bce(...)`. If an upstream release moves these seams, preflight gate G5
and the criterion-attach canary fail with remediation pointing here. The
repo-wide pin stays `>=8.3,<9.0` (narrowing it would constrain the whole
project for a training-only concern); the *validated* version is recorded at
the top of this document and in the engineering report.

---
*Previous:* [phase4_engineering_report.md](phase4_engineering_report.md) ·
*Related:* [mitigation_runbook.md](mitigation_runbook.md) ·
[ADR index](README.md)
