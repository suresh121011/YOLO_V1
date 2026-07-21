# Phase-5 Architecture Decision Records (ADR-P5)

Records the load-bearing design decisions for Production Dataset Engineering
(missing-annotation resolution L1–L5, Dataset v0.5→v1.0). Each ADR follows the
Phase-4 format (Status / Context / Decision / Alternatives / Consequences) and
is referenced from the code, configs, and `dvc.yaml` header that implement it.
All twelve were ratified together by the council PASS of 2026-07-17.

| ADR | Decision |
|:---|:---|
| [ADR-P5-01](ADR-P5-01-candidate-artifact-isolation.md) | Auto-generated labels never write `labels/` — candidate artifact only |
| [ADR-P5-02](ADR-P5-02-yolo-world-primary-backend.md) | YOLO-World primary backend; optional GroundingDINO; pinned-weight determinism |
| [ADR-P5-03](ADR-P5-03-cvat-round-trip.md) | CVAT round-trip is the only candidates→labels path; reuse P3 importer |
| [ADR-P5-04](ADR-P5-04-verification-ledger-trust-expansion.md) | Verification ledger = single source of trust expansion; `trusted_list_with_ledger` |
| [ADR-P5-05](ADR-P5-05-labels-overlay-no-mutation.md) | Labels-only overlay (`data/merged_verified`); no mutation, no image duplication |
| [ADR-P5-06](ADR-P5-06-coverage-no-inference.md) | L4 coverage is pure arithmetic over pinned candidates; no inference at report time |
| [ADR-P5-07](ADR-P5-07-releases-as-code.md) | Releases as code: gate-driven manifests, frozen `record_release`, v0.5→v1.0 ladder |
| [ADR-P5-08](ADR-P5-08-cross-dataset-label-salvage.md) | L3 = exact-SHA256 label salvage + near-dup candidates; no image registration |
| [ADR-P5-09](ADR-P5-09-local-dvc-remote.md) | Local-drive DVC remote default now; S3 secondary (migration documented) |
| [ADR-P5-10](ADR-P5-10-ab-benchmark-acceptance-evidence.md) | Full-scale A/B = v1.0 acceptance evidence (one run/arm, fixed config); tuning → Phase 6 |
| [ADR-P5-11](ADR-P5-11-annotation-deps-optional-extra.md) | Annotation deps optional extra outside CI (FakeAnnotator + importorskip) |
| [ADR-P5-12](ADR-P5-12-vectorized-dedup.md) | Vectorized-Hamming dedup with decision-equivalence guarantee, over a BK-tree |
