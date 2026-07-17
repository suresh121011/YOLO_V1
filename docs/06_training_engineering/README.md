# 06 — Training Engineering (Phase 4: Missing Annotation Mitigation)

Training-pipeline engineering documentation. Phase 4 delivered the
missing-annotation mitigation framework: public datasets annotate only a
subset of the 23-class taxonomy, and stock YOLO training turns every
unlabeled class into false "background" supervision — this phase removes that
error at the loss level, strictly opt-in, without modifying Ultralytics.

## Documents

| Document | Contents |
|---|---|
| [phase4_engineering_report.md](phase4_engineering_report.md) | What shipped, validation evidence, smoke benchmark results, limitations, Phase-5 readiness |
| [masked_loss_architecture.md](masked_loss_architecture.md) | Component map, masking mechanism + identity proof, injection flow, gates G1–G8, performance budgets, compatibility contract |
| [mitigation_runbook.md](mitigation_runbook.md) | Operate the pipeline: generate metadata, preflight, train, validate, benchmark, evaluate; troubleshooting by gate ID |

## Architecture Decision Records

| ADR | Decision |
|---|---|
| [ADR-P4-01](adr/ADR-P4-01-loss-level-masking.md) | Masked BCE at the loss level (vs sampling filters / pseudo-labels) |
| [ADR-P4-02](adr/ADR-P4-02-trainer-injection.md) | Criterion injection via trainer callback; model class stays stock (checkpoint portability) |
| [ADR-P4-03](adr/ADR-P4-03-completeness-artifact.md) | Policy-indirection completeness artifact with hard-fail validation |
| [ADR-P4-04](adr/ADR-P4-04-strict-mixing-aug-policy.md) | Mixing augmentations forbidden under mitigation (gate G8) |
| [ADR-P4-05](adr/ADR-P4-05-policy-provider-registry.md) | Pluggable completeness-policy providers (registry pattern) |

## Committed evidence

- [Masking validation report](../../data/qa_reports/phase4_mitigation/masking_validation_report.md) — M3.5 gate (PASS)
- [Benchmark report](../../data/qa_reports/phase4_mitigation/benchmark_report.md) — baseline vs mitigated + performance budgets
- [Completeness report](../../data/qa_reports/completeness_report.json) — per-policy coverage (DVC-tracked)

---
*Related:* [04 — Dataset Engineering](../04_dataset_engineering/README.md) ·
[Risk register](../01_executive_implementation_plan/risk_register.md) (R25–R29) ·
[05 — Pre-Phase-4 audit](../05_audit/pre_phase4_production_readiness_audit.md)
