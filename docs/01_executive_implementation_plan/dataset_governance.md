# Dataset Governance Policy

## Purpose

Defines the dataset lifecycle, versioning policy, annotation quality requirements, and approval workflow.

## Dependencies

Reads:
- implementation_phases.md

Used By:
- recommendations.md

Related:
- ../03_engineering_appendix/dataset_templates.md
- ../03_engineering_appendix/yaml_examples.md

---

## Dataset Lifecycle

```mermaid
graph LR
    RAW["Raw: Unprocessed captures and downloads"] --> CLEAN["Cleaned: Blur removed, Duplicates removed"]
    CLEAN --> ANNOTATE["Annotated: Dual-annotator CVAT workflow"]
    ANNOTATE --> VALIDATE["Validated: 8 QA scripts pass, 0 critical errors"]
    VALIDATE --> VERSION["Versioned: DVC tag, Semantic version"]
    VERSION --> RELEASE["Released: Approved for training, Changelog updated"]
    RELEASE --> ARCHIVE["Archived: Immutable snapshot, DVC remote"]

    style RAW fill:#1a1a2e,stroke:#8d99ae,color:#fff
    style CLEAN fill:#16213e,stroke:#0f3460,color:#fff
    style ANNOTATE fill:#16213e,stroke:#0f3460,color:#fff
    style VALIDATE fill:#0f3460,stroke:#e94560,color:#fff
    style VERSION fill:#533483,stroke:#e94560,color:#fff
    style RELEASE fill:#e94560,stroke:#fff,color:#fff
    style ARCHIVE fill:#2b2d42,stroke:#8d99ae,color:#fff
```

## Dataset Versioning Policy

| Aspect | Policy |
|:-------|:-------|
| **Versioning scheme** | Semantic versioning: `dataset-v{major}.{minor}.{patch}` |
| **Major version** | Class taxonomy change or train/val split reset |
| **Minor version** | New images added (100+ new images) |
| **Patch version** | Label corrections, QA fixes, metadata updates |
| **DVC tagging** | Every release tagged in Git + DVC remote |
| **Changelog** | `data/DATASET_CHANGELOG.md` updated per release |

## Annotation Quality Requirements

| Check | Minimum Standard | Tool |
|:------|:----------------|:-----|
| Bounding box tightness | ≤ 5px padding | Automated validator |
| Occlusion annotation | ≥ 25% visible — must annotate | Annotator training |
| Multi-object completeness | All visible classes annotated | Annotator checklist |
| Class ID validity | Only IDs 0–22 permitted | `check_class_consistency.py` |
| Image resolution | ≥ 320×320 pixels | `image_quality_filter.py` |
| Duplicate threshold | Perceptual hash Hamming distance ≥ 5 | `check_duplicates.py` |
| Train/val leakage | Zero hash overlap | `check_train_val_leakage.py` |
| Minimum per-class count | ≥ 200 instances | `dataset_statistics.py` |

## Dataset Approval Workflow

```
Collection → QA Auto-Pass → Lead Review → Approval Sign-off → DVC Commit → Training
```

| Stage | Actor | Gate Criterion |
|:------|:------|:--------------|
| Collection | Annotator | Images meet capture protocol |
| QA Auto-Pass | `run_full_qa.py` | 0 critical errors |
| Lead Review | ML Lead | Spot-check 50 images per class |
| Approval Sign-off | Engineering Manager | Overall dataset quality confirmed |
| DVC Commit | MLOps Engineer | Version tagged and pushed |
| Training | ML Engineer | Training job initiated with approved dataset |

## Train/Validation/Test Governance

| Split | Ratio | Rule |
|:------|:------|:-----|
| **Train** | 85% | Primary training data; stratified by class |
| **Validation** | 15% | Evaluation only; never used in training |
| **Test** | Separate held-out set | Field-captured mobile videos; never seen during any training phase |
| **Leakage prevention** | Perceptual hash + source tracking | Video clips go to same split; session-based assignment |

---

Previous: [security_privacy.md](./security_privacy.md)

Next: [engineering_standards.md](./engineering_standards.md)

Related: [../03_engineering_appendix/dataset_templates.md](../03_engineering_appendix/dataset_templates.md)
