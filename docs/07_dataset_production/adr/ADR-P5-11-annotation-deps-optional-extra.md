# ADR-P5-11 — Annotation Dependencies as an Optional Extra Outside CI (FakeAnnotator + importorskip)

**Status:** Accepted (Phase 5, 2026-07)
**Deciders:** Phase-5 engineering; council PASS 2026-07-17

## Context

L2 backends need heavy, GPU-oriented dependencies (ultralytics/torch already
present; `transformers` + a pinned CLIP git SHA for open-vocab). Forcing these
onto the default install and into CI would slow every build and make unit tests
depend on model weights.

## Decision

Annotation dependencies live in `requirements-annotation.txt` and a pyproject
`annotation` extra (`transformers>=4.45,<5.0`, pinned CLIP git SHA; torch listed
for explicitness). They are **not** in the default install path and **not** in
CI. Unit tests register a `FakeAnnotator` (deterministic, model-free) and use
`pytest.importorskip` for the real backends, so the whole annotation flow is
tested offline without weights.

## Alternatives considered

1. **Put annotation deps in the base install.** Rejected: bloats every
   environment and CI with a GPU stack most consumers never run.
2. **Skip backend tests entirely on CI.** Rejected: the orchestration
   (targeting → annotate → filter → artifact) must be covered — `FakeAnnotator`
   covers it deterministically; `importorskip` guards the real-weight paths.

## Consequences

- Positive: CI stays fast and weight-free; the annotation flow is fully tested
  via the Fake; real backends are pinned and reproducible when installed.
- Constraint: real-backend behavior is validated only where the extra is
  installed (the GPU box), by design.

Related: [ADR-P5-02](ADR-P5-02-yolo-world-primary-backend.md)
