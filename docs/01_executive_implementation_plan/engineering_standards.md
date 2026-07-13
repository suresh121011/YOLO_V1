# LLM Council Review & Engineering Scorecard

## Purpose

Final LLM Council review by 13 senior engineers. Contains individual assessments and the consolidated engineering scorecard.

## Dependencies

Reads:
- All preceding executive documents

Used By:
- recommendations.md

Related:
- roadmap.md

---

## Council Member Assessments

**AI/ML Architect:**
The YOLO11n → SmolVLM2 → Rule Engine pipeline is well-chosen for V1. YOLO11n provides the best FPS/accuracy tradeoff for edge deployment. SmolVLM2 as an optional contextual layer is architecturally sound. The event memory design allows temporal reasoning without expensive sequence models. Concern: The wet floor class remains the hardest to detect reliably. Recommend dedicated data collection sprints and post-hoc confidence calibration. **Score: 8.5/10**

**Computer Vision Engineer:**
The 23-class taxonomy is well-curated. The dual-annotator QA workflow is production-appropriate. The training configuration (AdamW, warmup, mosaic augmentation) follows current best practices. Concern: Motion blur in real mobile usage will significantly degrade detection quality — recommend including motion-blurred training samples. **Score: 8/10**

**Edge AI Engineer:**
The choice of YOLO11n (nano) with optional INT8 quantization for mobile is correct. The frame-skip strategy for SmolVLM2 (every 5th frame) is pragmatic. Graceful degradation to rule-only mode is critical and correctly specified. Concern: Thermal throttling on sustained inference is underestimated. Recommend adaptive frame rates and thermal monitoring from day one. **Score: 7.5/10**

**Software Architect:**
The folder structure is clean and production-oriented. The YAML-driven rule engine is the right design pattern for extensibility. Concern: The orchestrator threading model needs explicit documentation — is it single-threaded sequential or does YOLO/VLM/TTS run concurrently? This needs to be resolved before Phase 6. **Score: 8/10**

**Backend Engineer:**
Pipeline components have clear interfaces. Event logging to SQLite is appropriate for V1. The alert queue with cooldown logic is well-designed. Concern: Error propagation across the pipeline needs explicit handling — what happens if YOLO fails mid-stream? Needs watchdog and restart logic for all components. **Score: 7.5/10**

**MLOps Engineer:**
DVC integration for dataset versioning is excellent. The active learning logging strategy is production-grade. Concern: No experiment tracking tool (MLflow/WandB) is specified for training runs. Recommend adding WandB free tier for training metric tracking. **Score: 7.5/10**

**DevOps Engineer:**
The DVC pipeline definition is well-structured. Dependency pinning via `requirements.txt` is correctly specified. Concern: No CI/CD pipeline defined — even a simple GitHub Actions workflow for QA script execution would significantly improve reliability. **Score: 6.5/10**

**QA Engineer:**
The testing pyramid is comprehensive. The system testing scenario list covers realistic edge cases. Concern: No regression test suite specified. When the model is retrained, there is no automated way to verify it has not regressed on previously-passing test cases. **Score: 7/10**

**Security & Privacy Engineer:**
Privacy-by-design is correctly prioritized. No facial recognition, no cloud upload, and offline-first are all the right choices. Concern: The face class should have an explicit written policy stating it is used only for person-presence detection, not identification. **Score: 8/10**

**Product Manager:**
The product-market fit is strong. The Indian-home focus and offline-first design solve real problems. The alert cooldown mechanism addresses alert fatigue appropriately. Concern: User onboarding and caregiver configuration UX is undefined for V1. **Score: 7.5/10**

**Engineering Manager:**
The plan is detailed, realistic, and well-scoped for a small team. Concern: The 2–4 week custom data collection phase is the highest-risk timeline item. Recommend a dedicated collection sprint with daily progress tracking. **Score: 8/10**

**CTO:**
The architecture is appropriate for the problem. The choice to use proven, production-ready components (Ultralytics YOLO, Piper TTS, SmolVLM2) over custom-built alternatives shows engineering maturity. The decision to keep V1 offline-only is strategically correct — it eliminates cloud infrastructure cost and privacy liability simultaneously. **Score: 8.5/10**

**Managing Director:**
This addresses a real and growing market. India's elderly population is underserved by technology. The privacy-first approach removes a major adoption barrier for conservative Indian families. The V1/V2/V3 roadmap is credible. Key question: What is the monetization model? **Score: 8/10**

---

## Final Engineering Scorecard

| Dimension | Score /10 | Rationale |
|:----------|:---------:|:----------|
| **ML Pipeline** | 8.5 | YOLO11n well-chosen; training config sound; evaluation framework thorough |
| **Runtime Architecture** | 8.0 | Clean modular design; threading model needs documentation |
| **Dataset Strategy** | 8.0 | Comprehensive; dual-annotator QA; DVC versioning; Indian-context aware |
| **Security & Privacy** | 8.5 | Privacy-by-design correctly implemented; offline-first; no biometrics |
| **Validation** | 7.5 | Good coverage; needs regression suite and CI/CD |
| **Scalability** | 7.0 | V1 correctly scoped; V2 scalability path clear but not yet designed |
| **Maintainability** | 7.5 | YAML-configured rules; modular code; lacks automated regression guard |
| **Extensibility** | 8.0 | Plugin-ready architecture; YAML rules extensible; class taxonomy expandable |
| **Production Readiness** | 7.0 | Solid V1 design; needs CI/CD, watchdog, and thermal management |
| **Documentation Quality** | 8.5 | Three-document structure; diagrams; tables; risk register |
| **TOTAL** | **78.5/100** | |

## What Prevents 100/100

| Gap | Category | V1 or V2 Resolution |
|:----|:---------|:--------------------|
| No CI/CD pipeline | DevOps | Add GitHub Actions — V1 late phase |
| Threading model undocumented | Architecture | Resolve before Phase 6 — V1 |
| No experiment tracking | MLOps | Add WandB free tier in Phase 5 — V1 |
| No regression test suite | QA | Add post-training regression tests — V1 |
| Thermal management untested | Edge | Add thermal profiling in Phase 7 — V1 |
| User onboarding UX undefined | Product | Add minimal setup guide — V1 |
| DPDP compliance deferred | Legal/Privacy | Schedule compliance review — V2 |
| Monetization model unclear | Business | Define before V2 planning — V2 |

---

Previous: [dataset_governance.md](./dataset_governance.md)

Next: [roadmap.md](./roadmap.md)

Related: [recommendations.md](./recommendations.md)
