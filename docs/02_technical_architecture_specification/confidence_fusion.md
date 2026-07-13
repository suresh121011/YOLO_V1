# Confidence Fusion

## Purpose

Combines YOLO detection confidence with SmolVLM2 risk assessment to produce a fused confidence score.

## Dependencies

Reads:
- interfaces.md

Used By:
- orchestrator.md

Related:
- rule_engine.md
- event_memory.md

---

**File:** `src/pipeline/confidence_fusion.py`

## Fusion Formula

```
fused_confidence = (α × yolo_confidence) + (β × vlm_risk_score)
where:
  α = 0.7  (YOLO weight — primary signal)
  β = 0.3  (VLM weight — contextual signal)
```

## VLM Risk Score Mapping

| VLM Severity | Risk Score |
|:------------|:----------|
| Not mentioned | 0.0 |
| low | 0.3 |
| medium | 0.5 |
| high | 0.8 |
| critical | 1.0 |

## Safety Constraint

Fused score can only **increase** severity, never decrease it. If Rule Engine triggers on raw YOLO score, VLM cannot suppress the alert.

## Fallback

If VLM unavailable, `fused_confidence = yolo_confidence` (α=1.0, β=0.0).

> For full implementation, see [../03_engineering_appendix/python_examples.md](../03_engineering_appendix/python_examples.md)

---

Previous: [rule_engine.md](./rule_engine.md)

Next: [orchestrator.md](./orchestrator.md)

Related: [event_memory.md](./event_memory.md)
