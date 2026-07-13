# Architecture Decisions: Assumptions, Constraints, Trade-offs & Operational Readiness

## Purpose

Documents all assumptions, constraints, design trade-offs, known limitations, non-goals, and operational readiness checklists.

## Dependencies

Reads:
- deployment_architecture.md
- error_handling.md

Used By: None (terminal document for TAS)

Related:
- ../01_executive_implementation_plan/risk_register.md
- ../01_executive_implementation_plan/roadmap.md

---

## Assumptions

| # | Assumption | Impact if Wrong |
|:--|:-----------|:---------------|
| A1 | Target device has ≥ 4GB RAM | SmolVLM2 may OOM; degrade to VLM-disabled mode |
| A2 | Camera provides ≥ 720p 30FPS | Lower resolution degrades small-object detection |
| A3 | Elderly person is typically visible in frame | Rules requiring `person` class may not trigger |
| A4 | Indian home objects have consistent visual appearance | Custom data collection may need re-scoping |
| A5 | Piper TTS audio output reaches elderly user | Speaker volume/placement is a deployment concern |
| A6 | Training GPU available for 3–5 days | Timeline extends if GPU unavailable |

## Constraints

| Constraint | Reason |
|:-----------|:-------|
| 100% offline operation | Privacy + Indian connectivity reliability |
| YOLO11n (not larger) | Edge hardware memory and FPS constraints |
| No biometric identification | Privacy by design + regulatory caution |
| English-only TTS in V1 | Piper Hindi model quality validation deferred |
| SQLite only (no cloud DB) | Offline-first requirement |
| 23 classes maximum in V1 | Dataset collection capacity constraint |

## Design Trade-offs

| Trade-off | Decision Made | Rationale |
|:----------|:-------------|:----------|
| Recall vs Precision | Prefer recall | Missed hazard worse than false alarm |
| YOLO11n vs YOLO11s | 11n primary | FPS critical; 11s used as reference only |
| VLM always-on vs every-5th | Every 5th frame | Hardware performance constraint |
| Custom vs public data | Both | Public data for common classes; custom for Indian-specific |
| Sequential vs concurrent threads | Concurrent (3 threads) | TTS and VLM must not block main inference loop |
| SQLite vs file logging | SQLite primary | Queryable; atomic writes; ACID |
| Rule-based vs ML-based safety logic | Rule-based V1 | Explainable; debuggable; low latency |

## Known V1 Limitations

| Limitation | Severity | Resolution Plan |
|:-----------|:---------|:---------------|
| No fall detection | High | V2 — pose estimation model |
| English-only voice | Medium | V2 — Hindi TTS model |
| No person re-identification | Medium | V2 — multi-person tracking |
| No activity recognition | Medium | V2 — action classification |
| Wet floor class accuracy may be low | High | Ongoing data collection; calibration tuning |
| No night vision | Medium | V2 — IR camera support |
| No medication schedule integration | Low | V3 |
| No caregiver notification | Medium | V2 — dashboard |

## Non-Goals (V1)

- ❌ Cloud-based model serving
- ❌ Real-time video streaming to caregiver
- ❌ Facial recognition or person identification
- ❌ Multi-camera support
- ❌ Mobile app UI
- ❌ Hindi or regional language TTS
- ❌ Autonomous emergency calling
- ❌ Multi-home deployment management
- ❌ Continuous recording or video storage

---

## Operational Readiness

### Pre-Deployment Checklist

| Item | Check |
|:-----|:------|
| Model weights loaded and hash verified | [ ] |
| TTS health check passed | [ ] |
| Config YAML validated (Pydantic schema) | [ ] |
| All feature flags reviewed and set | [ ] |
| SQLite database writable | [ ] |
| Camera accessible and streaming | [ ] |
| Alert queue empty at startup | [ ] |
| Memory profiling baseline captured | [ ] |
| FPS benchmark on target device run | [ ] |
| All unit tests passing | [ ] |
| All integration tests passing | [ ] |
| Field test in ≥ 1 home completed | [ ] |

### Monitoring & Observability

| Metric | Collection Method | Alert Threshold |
|:-------|:-----------------|:----------------|
| FPS | Metrics collector | < 5 FPS for > 30s |
| RAM usage | `psutil` sampler | > 90% of budget |
| TTS queue depth | Queue monitor | > 5 pending alerts |
| Error rate | Logger | > 10 errors/minute |
| Device temperature | OS sensor | > 45°C |

### Rollback Strategy

| Scenario | Rollback Action |
|:---------|:---------------|
| New model causes regression | Restore `best.pt` from previous DVC tag |
| Rule change causes false positives | `reload_rules()` with previous YAML from Git |
| Config change breaks pipeline | Restore previous config from Git |
| Dataset rollback needed | `dvc checkout dataset-v{previous}` |

### Maintenance Plan

| Activity | Frequency | Owner |
|:---------|:----------|:------|
| Review active learning samples | Weekly | ML Engineer |
| Model performance audit | Monthly | ML Lead |
| Dataset collection sprint | Quarterly | Annotation Team |
| Dependency security audit | Monthly | DevOps Engineer |
| Field test in new home | Quarterly | QA Engineer |
| Rule review with caregiver feedback | Monthly | Product Manager |

---

*For executive context: see [../01_executive_implementation_plan/README.md](../01_executive_implementation_plan/README.md)*

*For code examples: see [../03_engineering_appendix/README.md](../03_engineering_appendix/README.md)*

---

Previous: [deployment_architecture.md](./deployment_architecture.md)

Next: None (end of technical specification)

Related: [../01_executive_implementation_plan/risk_register.md](../01_executive_implementation_plan/risk_register.md)
