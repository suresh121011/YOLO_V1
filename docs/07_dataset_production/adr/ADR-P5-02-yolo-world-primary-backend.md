# ADR-P5-02 — YOLO-World Primary Backend; Optional GroundingDINO; Pinned-Weight Determinism

**Status:** Accepted (Phase 5, 2026-07)
**Deciders:** Phase-5 engineering; council PASS 2026-07-17

## Context

L2 auto-annotation needs an open-vocabulary detector to propose boxes for
classes public datasets don't label exhaustively. The choice drives the
dependency surface, the determinism contract, and reproducibility of every
candidate artifact.

## Decision

`yolo_world` (ultralytics `yolov8x-worldv2.pt`) is the **primary** backend — it
reuses the already-pinned ultralytics/torch stack, so no new heavy dependency
sits on the critical path. `grounding_dino` (HF `IDEA-Research/grounding-dino-base`)
is an **optional** extra, disabled by default, auto-recommended only when
calibrated yolo_world precision on a priority class falls below
`grounding_dino.enable_below_precision` (0.4). SAM is **not** a backend — it is
an optional box-tightening refine pass that never creates or deletes detections.
Determinism: weights carry a `weights_sha256` verified at load; the
ultralytics/torch/CUDA versions, prompt fingerprint, and taxonomy fingerprint
are recorded in every artifact; sorted image order, FP32 (TF32 disabled),
fixed seeds, `torch.use_deterministic_algorithms(True, warn_only=True)`;
`--verify-determinism` re-runs a fixed 20-image sample and diffs. The CLIP
dependency is pinned to a git SHA so ultralytics' runtime auto-install never
fires.

## Alternatives considered

1. **GroundingDINO-first.** Rejected: forces the heavy `transformers` extra onto
   the critical path and a second inference stack to pin, for no box-native
   advantage over YOLO-World on our priority classes.
2. **Florence-2.** Descoped: no box-native advantage; revisit in Phase 6.
3. **Promise cross-machine bit-identity.** Rejected as dishonest: GPU/driver
   float behavior varies; candidates are advisory, so within-machine
   determinism is the enforceable and sufficient contract.

## Consequences

- Positive: zero new critical-path dependencies; reproducible candidates within
  a machine; a documented, calibration-gated escalation to a second model.
- Negative: candidates are not guaranteed bit-identical across machines — the
  reason human-verified outputs (exact) are the source of truth, not candidates.

Related: [ADR-P5-01](ADR-P5-01-candidate-artifact-isolation.md),
[ADR-P5-06](ADR-P5-06-coverage-no-inference.md),
[ADR-P5-11](ADR-P5-11-annotation-deps-optional-extra.md)
