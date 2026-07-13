# Project Scope & Resource Requirements

## Purpose

Defines team composition, compute requirements, and resource constraints for V1 delivery.

## Dependencies

Reads:
- business_goals.md

Used By:
- implementation_phases.md
- recommendations.md

Related:
- architecture_overview.md

---

## Team Requirements (V1)

| Role | Responsibility | Effort |
|:-----|:--------------|:-------|
| ML / CV Engineer (1–2) | YOLO training, dataset pipeline, evaluation | Full-time 6–8 weeks |
| Data Annotator (2–3) | Custom image capture and annotation | Full-time 3–4 weeks |
| Backend Engineer (1) | Pipeline integration, TTS, logging | Full-time 3–4 weeks |
| QA Engineer (1) | Test plan execution, field testing | Part-time 2–3 weeks |
| ML Lead / Engineering Manager | Code review, approval, architecture decisions | Part-time throughout |

## Compute Requirements

| Resource | Minimum | Recommended |
|:---------|:--------|:------------|
| Training GPU | NVIDIA GTX 1080 Ti (8GB VRAM) | RTX 3090 / A100 / Colab T4 |
| Training RAM | 16 GB | 32 GB |
| Training Storage | 100 GB SSD | 250 GB NVMe |
| Inference Device | Android mid-range (SD 730+) | Android flagship / Raspberry Pi 4 |
| Inference RAM | 4 GB (device) | 6+ GB |

> [!WARNING]
> **GPU Availability**: Training assumes access to an NVIDIA GPU with ≥8GB VRAM. Google Colab T4/A100 is an acceptable alternative. Please confirm training compute availability before Phase 5.

> [!WARNING]
> **Custom Data Collection**: Phases 2–3 require physically collecting and annotating 2,000+ images from Indian homes. This is a significant manual effort requiring dedicated annotator time.

---

Previous: [business_goals.md](./business_goals.md)

Next: [architecture_overview.md](./architecture_overview.md)

Related: [implementation_phases.md](./implementation_phases.md)
