# ADR-P6-07 — `near()` Is Frame-Normalized Screen Space; `room` Is Config, `zone` Is Deferred

**Status:** Accepted (Phase 6, 2026-08)
**Deciders:** Phase-6 engineering; three-lens cold-read council 2026-08-07

## Context

Spatial predicates are the whole point of the knowledge layer — `detected(knife) AND detected(person)`
fires whenever both are anywhere in frame, which is why the rule it powers is being deleted. But
"proximity" is underspecified, and this repository already has **three conflicting conventions** in play:

- `BoundingBox` is normalized to the frame in centre format, `cx/cy/w/h ∈ [0,1]`
  (`src/pipeline/__init__.py:52-69`).
- `docs/02_technical_architecture_specification/rule_engine.md:34` documents
  `near(class_a, class_b, pixel_dist)` — **pixels**.
- `scripts/inference/test_video.py:371-392` serialises detections as **absolute xyxy pixels**, a third
  representation coexisting with the normalized one.

Meanwhile "Kitchen Cooking" is a *room*, not a geometric region, and the field-test protocol explicitly
varies camera mounting height across houses.

## Decision

**`near(a, b, max_dist)` operates on frame-normalized centre distance**, `max_dist ∈ (0, 1]`. `0.2` means
"centres within 20% of frame width". This is documented in the predicate registry docstring and in the
schema field description as **a screen-space proxy for physical proximity, not a distance** — it varies
with field of view and mounting, and it has no depth.

`overlaps(a, b, min_iou)` uses the existing `BoundingBox.iou()` (`src/pipeline/__init__.py:83-96`), which
is implemented, tested, and currently unused.

`rule_engine.md:34` is corrected to say `normalized_dist` in the same milestone, rather than leaving two
specifications in the tree.

**`room` and `zone` are separated and only `room` ships.**

- **`room`** is a deployment-time constant — one camera per room — drawn from the existing vocabulary at
  `configs/capture_config.yaml:32-33` (`kitchen`, `bedroom`, `bathroom`, `hall`, `balcony`, `corridor`,
  `pooja_room`, `staircase`). It costs nothing, needs no calibration, and is what scenarios like "Kitchen
  Cooking" actually mean. It also gives `<CAT>` in `SC-<CAT>-<NNN>` and `room_context` one shared,
  validated vocabulary.
- **`zone`** — a per-camera polygon in normalized coordinates — is **deferred**. It requires an authoring
  tool, per-installation calibration, and a recalibration story for the first time anyone bumps the
  camera. Its entry price is that recalibration story, and nothing in the current scenario set needs it.

Because `near()` is FOV-dependent, camera installation guidance (mounting height, angle, coverage) becomes
a Track C deliverable. Any threshold tuned in one house is otherwise meaningless in the next.

## Alternatives considered

1. **Pixel distance, as the Phase-2 doc says.** Rejected: resolution-dependent, so the same rule behaves
   differently on a 720p and a 1080p stream from the same mount. Normalized units at least isolate the
   resolution variable.
2. **Metric distance via monocular depth estimation.** Rejected for this phase: adds a model, a
   calibration step, and latency to a 5 ms budget, for a precision the current scenarios do not need.
   Reconsider only if a scenario genuinely requires physical distance.
3. **Ship `zone` now with hand-authored polygons.** Rejected: an uncalibrated polygon set that silently
   decays as the camera shifts is worse than no zones, because the scenarios keyed on it fail silently.
4. **Bounding-box IoU as the only spatial predicate.** Rejected: IoU is 0 for two adjacent
   non-overlapping objects, so it cannot express "person standing beside the stove" — the single most
   important spatial relation in the set.

## Consequences

- Positive: one convention, stated in the artifact and enforced by the schema; the third representation
  in `test_video.py` is documented as an export format, not an input.
- Positive: `room` delivers most of the practical value of spatial context at zero calibration cost.
- Constraint: `near()` thresholds are FOV-dependent and must be validated per capture house during M6,
  not assumed portable.
- Constraint: scenarios that genuinely need "in the walking path" (cord-on-floor, clutter) cannot be
  expressed precisely and are therefore scoped as caregiver-facing assessment findings rather than
  real-time alerts — which is independently the right call on human-factors grounds.

Related: [ADR-P6-03](ADR-P6-03-structured-ast-over-string-dsl.md)
