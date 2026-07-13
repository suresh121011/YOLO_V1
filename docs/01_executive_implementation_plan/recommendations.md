# Final Recommendations

## Purpose

Immediate actions, architecture decisions to resolve, and non-negotiable quality gates.

## Dependencies

Reads:
- business_goals.md
- risk_register.md
- engineering_standards.md

Used By:
- appendix_links.md

Related:
- validation_strategy.md

---

## Immediate Actions (Before Phase 1 Starts)

1. **Confirm training GPU availability** — Colab T4 or on-premises NVIDIA GPU
2. **Confirm target deployment device** — Android model or Raspberry Pi 4; determines export format (TFLite vs ONNX)
3. **Confirm TTS language** — English only, or Hindi from V1?
4. **Assign data collection team** — minimum 2 people for Indian home capture
5. **Set up annotation tool** — CVAT (self-hosted) or Roboflow account

## Architecture Decisions to Resolve Before Phase 6

1. **Threading model** — Define whether pipeline is sequential or concurrent; document in Technical Architecture Specification
2. **SmolVLM2 variant** — Confirm 256M vs 500M based on target hardware benchmarks
3. **Alert delivery** — TTS only, or add on-screen visual alert as fallback?
4. **Logging destination** — SQLite only, or JSON files, or both?

## Quality Gates (Non-Negotiable)

| Gate | Rule |
|:-----|:-----|
| Do not begin training | Until QA pipeline passes with 0 critical errors |
| Do not deploy | Until field-tested in ≥ 3 Indian homes |
| Do not merge rule engine | Until 100% rule test coverage achieved |
| Safety class recall | ≥ 0.80 is a hard production gate — no exceptions |

---

Previous: [roadmap.md](./roadmap.md)

Next: [appendix_links.md](./appendix_links.md)

Related: [validation_strategy.md](./validation_strategy.md), [business_goals.md](./business_goals.md)
