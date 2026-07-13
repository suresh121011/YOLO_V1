# Business Goals & Success Metrics

## Purpose

Defines measurable business objectives and V1 go/no-go acceptance criteria.

## Dependencies

Reads:
- product_vision.md

Used By:
- validation_strategy.md
- recommendations.md

Related:
- risk_register.md

---

## Business Goals

| Goal | Measurement | V1 Target |
|:-----|:------------|:----------|
| Detect all critical home hazards | Safety-class recall | ≥ 80% |
| Low false alarm rate | False positive rate per class | < 10% |
| Usable on affordable hardware | Target device FPS | ≥ 15 FPS on Android mid-range |
| Complete privacy guarantee | Data residency audit | 100% on-device |
| Natural communication | TTS quality score | Subjective ≥ 4/5 (caregiver rating) |
| Fast deployment readiness | End-to-end latency | < 2 seconds (frame to spoken alert) |

## V1 Acceptance Criteria (Go/No-Go)

| Criterion | Threshold |
|:----------|:----------|
| All 23 classes have ≥ 200 labeled training images | Required |
| Safety-critical classes have ≥ 500 images | ≥ 80% of safety classes |
| YOLO11n mAP50 on validation set | ≥ 0.70 |
| Safety-critical class recall | ≥ 0.80 |
| End-to-end pipeline latency | < 2 seconds |
| QA pipeline: zero critical annotation errors | 0 critical errors |
| All rule engine rules tested | 100% coverage |
| Real Indian home test videos pass | ≥ 3 homes tested |

---

Previous: [product_vision.md](./product_vision.md)

Next: [project_scope.md](./project_scope.md)

Related: [validation_strategy.md](./validation_strategy.md), [risk_register.md](./risk_register.md)
