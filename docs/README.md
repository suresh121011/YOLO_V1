# Elderly Assistant System — Documentation

> **Project:** AI-Powered Home Safety System for Elderly Indians
> **Version:** 1.0.0 | **Last Updated:** July 2026
> **Architecture:** YOLO11n → Event Memory → SmolVLM2 → Rule Engine → Alert Queue → Piper TTS → Logger

---

## How to Use This Documentation

This documentation is **modular and LLM-friendly**. Each file covers exactly one topic.

- If you need a quick overview → read [01_executive_implementation_plan/SUMMARY.md](./01_executive_implementation_plan/SUMMARY.md)
- If you're implementing a component → go directly to the relevant file in `02_technical_architecture_specification/`
- If you need code, scripts, or templates → go directly to `03_engineering_appendix/`
- **Never load all files at once.** Read only what you need.

---

## Document Structure

```
docs/
├── README.md                             ← You are here
│
├── 01_executive_implementation_plan/     ← Business, vision, risk, roadmap (14 files)
│   ├── README.md                         ← Index
│   ├── SUMMARY.md                        ← One-page overview
│   ├── product_vision.md
│   ├── business_goals.md
│   ├── project_scope.md
│   ├── architecture_overview.md
│   ├── implementation_phases.md
│   ├── risk_register.md
│   ├── validation_strategy.md
│   ├── security_privacy.md
│   ├── dataset_governance.md
│   ├── engineering_standards.md          ← LLM Council scorecard (78.5/100)
│   ├── roadmap.md
│   ├── recommendations.md
│   └── appendix_links.md
│
├── 02_technical_architecture_specification/  ← Engineering spec (17 files)
│   ├── README.md                             ← Index
│   ├── system_architecture.md
│   ├── data_flow.md
│   ├── interfaces.md
│   ├── event_memory.md
│   ├── rule_engine.md
│   ├── confidence_fusion.md
│   ├── orchestrator.md
│   ├── plugin_architecture.md
│   ├── structured_logging.md
│   ├── feature_flags.md
│   ├── performance_budget.md
│   ├── threading_model.md
│   ├── error_handling.md
│   ├── data_contracts.md
│   ├── api_contracts.md
│   ├── deployment_architecture.md
│   └── architecture_decisions.md
│
└── 03_engineering_appendix/             ← Code, templates, checklists (12 files)
    ├── README.md                         ← Index
    ├── yaml_examples.md
    ├── python_examples.md
    ├── dataset_templates.md
    ├── training_scripts.md
    ├── qa_pipeline.md
    ├── dvc_pipeline.md
    ├── sample_logs.md
    ├── api_reference.md
    ├── annotation_guide.md
    ├── release_checklists.md
    ├── troubleshooting.md
    └── future_modules.md
```

---

## Quick Navigation by Role

| Role | Start Here |
|:-----|:-----------|
| **Managing Director / CTO** | [SUMMARY.md](./01_executive_implementation_plan/SUMMARY.md) → [business_goals.md](./01_executive_implementation_plan/business_goals.md) |
| **Engineering Manager** | [implementation_phases.md](./01_executive_implementation_plan/implementation_phases.md) → [risk_register.md](./01_executive_implementation_plan/risk_register.md) |
| **ML / CV Engineer** | [system_architecture.md](./02_technical_architecture_specification/system_architecture.md) → [data_contracts.md](./02_technical_architecture_specification/data_contracts.md) |
| **Backend Engineer** | [interfaces.md](./02_technical_architecture_specification/interfaces.md) → [threading_model.md](./02_technical_architecture_specification/threading_model.md) |
| **Data Annotator** | [annotation_guide.md](./03_engineering_appendix/annotation_guide.md) |
| **MLOps / DevOps** | [dvc_pipeline.md](./03_engineering_appendix/dvc_pipeline.md) → [release_checklists.md](./03_engineering_appendix/release_checklists.md) |
| **QA Engineer** | [validation_strategy.md](./01_executive_implementation_plan/validation_strategy.md) → [qa_pipeline.md](./03_engineering_appendix/qa_pipeline.md) |
| **New Team Member** | Read in order: SUMMARY → product_vision → architecture_overview → system_architecture |

---

## Key Technical Facts

| Fact | Value |
|:-----|:------|
| Detection model | YOLO11n (Ultralytics) |
| Scene analysis | SmolVLM2-256M (optional) |
| TTS engine | Piper neural TTS (`en_IN-medium`) |
| Classes | 23 (IDs 0–22) |
| Target FPS | ≥ 15 FPS on Android mid-range |
| Alert latency | < 2 seconds end-to-end |
| Operation mode | 100% offline, no cloud |
| Privacy | No facial recognition; no biometric data |
| LLM Council score | 78.5 / 100 |
