# ADR-P4-02 — Criterion Injection via Trainer Callback, Not Model Subclassing

**Status:** Accepted (Phase 4, 2026-07)

## Context

The masked loss must replace `v8DetectionLoss` without modifying Ultralytics
source. Constraints discovered against the pinned line (validated on 8.4.96):

- `model.train(trainer=...)` accepts a trainer **class**; Ultralytics
  instantiates it itself and **rejects unknown train kwargs**, so settings
  cannot be passed through the constructor.
- `BaseModel.loss()` creates the criterion lazily:
  `if criterion is None: self.criterion = self.init_criterion()`.
- `DetectionTrainer._setup_train` attaches hyperparameters
  (`self.model.args = self.args`, in `set_model_attributes`) and creates the
  EMA copy **before** the `on_train_start` callback fires; the first loss
  computation happens after.
- **Checkpoints pickle the EMA model object by class reference** and
  `save_model` strips `criterion` before serializing.

## Decision

1. `build_masked_trainer(config, lookup)` returns a dynamically configured
   subclass of `MaskedDetectionTrainer(DetectionTrainer)` with the settings as
   class attributes (fresh subclass per call — no shared mutable state;
   trainer classes are never pickled).
2. The trainer registers an `on_train_start` callback that sets
   `model.criterion = MaskedDetectionLoss(...)` on the (unwrapped) training
   model **and** the EMA model — deterministically pre-empting the lazy
   `init_criterion()` guard before the first batch. A pre-existing criterion
   at that point raises (fail-loud canary for upstream flow changes).
3. The model **class stays stock `DetectionModel`**.

## Alternatives considered

1. **`DetectionModel` subclass overriding `init_criterion()`** (built by an
   overridden `get_model`). Cleanest-looking OOP — rejected because
   checkpoints pickle the EMA model by class reference: every `best.pt` would
   require this repository importable at load time, breaking checkpoint
   portability and Phase-5/7 export. A closure-defined subclass would not
   even be picklable (checkpoint save would crash). It also required
   replicating `get_model()` internals that differ across the 8.3→8.4 line.
2. **Post-hoc `model.criterion` attach on the `YOLO(...)` object.** Rejected:
   the trainer builds a *fresh* model in `get_model()`; the attachment does
   not survive.
3. **Swapping `criterion` after training started** (e.g. first-batch hook).
   Rejected: races the lazy-init guard; harder to reason about under AMP.

## Consequences

- Positive: checkpoints stay portable (verified: `save_model` strips
  criterion); resume works — `train_yolo.py` re-passes the trainer class
  whenever mitigation is enabled; no version-fragile `get_model` replication.
- Negative: masking depends on the `on_train_start` ordering contract —
  guarded by the fail-loud pre-existing-criterion check and the G5 source
  canary (`assert_ultralytics_compat`).
- DDP is explicitly rejected (the configured class cannot be reconstructed in
  DDP worker processes); single-device training only — acceptable for this
  project's hardware.

Related: [ADR-P4-01](ADR-P4-01-loss-level-masking.md), risk R25.
