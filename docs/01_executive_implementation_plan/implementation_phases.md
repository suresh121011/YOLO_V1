# Implementation Phases & Roadmap

## Purpose

Defines the 7-phase project timeline, deliverables per phase, and the 23-class object taxonomy.

## Dependencies

Reads:
- architecture_overview.md
- project_scope.md

Used By:
- risk_register.md
- recommendations.md

Related:
- roadmap.md

---

## Phase Timeline

```mermaid
gantt
    title Elderly Assistant System V1 Delivery Timeline
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d

    section Phase 1 Foundation
    Project scaffold and taxonomy       :p1a, 2026-07-14, 2d
    data.yaml configs folder setup      :p1b, after p1a, 1d

    section Phase 2 Public Data
    COCO and Open Images extraction     :p2a, after p1b, 3d
    Roboflow imports and WIDER FACE     :p2b, after p2a, 2d

    section Phase 3 Custom Data
    Indian home photo capture           :p3a, after p2b, 14d
    CVAT annotation workflow            :p3b, after p3a, 7d

    section Phase 4 Dataset QA
    QA scripts and validation           :p4a, after p3b, 2d
    DVC versioning and splits           :p4b, after p4a, 1d

    section Phase 5 Training
    YOLO11n baseline training           :p5a, after p4b, 3d
    Evaluation and iteration            :p5b, after p5a, 2d

    section Phase 6 Pipeline
    Rule Engine and TTS integration     :p6a, after p5b, 5d
    SmolVLM2 integration                :p6b, after p6a, 5d

    section Phase 7 Validation
    Field testing 3 homes               :p7a, after p6b, 7d
    Bug fixes and production hardening  :p7b, after p7a, 7d
```

## Phase Deliverables

| Phase | Name | Duration | Key Deliverables |
|:------|:-----|:---------|:----------------|
| **1** | Foundation & Taxonomy | 2–3 days | Project scaffold, 23-class taxonomy, `data.yaml`, folder structure |
| **2** | Public Dataset Acquisition | 3–5 days | COCO/Open Images/Roboflow subsets, format-converted, quality-filtered |
| **3** | Custom Dataset Collection | 2–4 weeks | 2,000+ Indian-home images, fully annotated via dual-annotator workflow |
| **4** | Dataset QA & Versioning | 2–3 days | QA reports, DVC-versioned dataset, 85/15 train-val splits |
| **5** | YOLO Training & Evaluation | 3–5 days | YOLO11n model (mAP50 ≥ 0.70), evaluation artifacts, confusion matrix |
| **6** | Pipeline Integration | 1–2 weeks | SmolVLM2 + Rule Engine + Piper TTS assembled, end-to-end tested |
| **7** | Production Validation | 1–2 weeks | Field-tested in 3 Indian homes, latency benchmarked, exportable model |

## 23-Class Object Taxonomy

| Category | Classes | Source |
|:---------|:--------|:-------|
| **Safety-Critical** | `knife` · `stove` · `gas_cylinder` · `wire` · `wet_floor` · `medicine_strip` · `medicine_bottle` | Custom + COCO |
| **Navigation** | `person` · `face` · `door` · `walking_stick` · `support_handle` | COCO + Custom |
| **Furniture** | `chair` · `bed` · `cupboard` · `toilet` · `sink` | COCO + Open Images |
| **Daily Objects** | `water_bottle` · `laptop` · `monitor` · `charger` · `book` | COCO + Custom |
| **Documents** | `passport` | Custom mandatory |

> [!IMPORTANT]
> 8 of 23 classes require **mandatory custom Indian-home data collection** — these classes do not exist in adequate form in any public dataset (gas cylinders, medicine strips, wet floors, walking sticks, support handles, Indian stoves, passports, Indian cupboards).

---

Previous: [architecture_overview.md](./architecture_overview.md)

Next: [risk_register.md](./risk_register.md)

Related: [roadmap.md](./roadmap.md)
