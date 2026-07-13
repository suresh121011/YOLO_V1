# DVC Pipeline Definition

## Purpose

Complete `dvc.yaml` pipeline for reproducible dataset acquisition, processing, training, and evaluation.

## Dependencies

Reads:
- dataset_templates.md
- training_scripts.md
- qa_pipeline.md

Used By:
- release_checklists.md

Related:
- ../01_executive_implementation_plan/dataset_governance.md

---

## Full DVC Pipeline (`dvc.yaml`)

```yaml
# dvc.yaml — Full pipeline definition

stages:
  download_coco:
    cmd: python scripts/dataset/01_download_coco_subset.py
    deps:
      - scripts/dataset/01_download_coco_subset.py
    outs:
      - data/raw/coco_filtered

  download_openimages:
    cmd: python scripts/dataset/02_download_openimages_subset.py
    deps:
      - scripts/dataset/02_download_openimages_subset.py
    outs:
      - data/raw/openimages_filtered

  download_roboflow:
    cmd: python scripts/dataset/03_download_roboflow_datasets.py
    deps:
      - scripts/dataset/03_download_roboflow_datasets.py
    outs:
      - data/raw/roboflow_imports

  download_wider_face:
    cmd: python scripts/dataset/04_download_wider_face.py
    deps:
      - scripts/dataset/04_download_wider_face.py
    outs:
      - data/raw/wider_face

  remap_classes:
    cmd: python scripts/dataset/05_remap_classes.py --all
    deps:
      - scripts/dataset/05_remap_classes.py
      - data/raw/coco_filtered
      - data/raw/openimages_filtered
      - data/raw/roboflow_imports
      - data/raw/wider_face

  merge_datasets:
    cmd: python scripts/dataset/07_merge_datasets.py
    deps:
      - data/raw/coco_filtered
      - data/raw/openimages_filtered
      - data/raw/roboflow_imports
      - data/raw/wider_face
      - data/raw/custom_captures
      - scripts/dataset/07_merge_datasets.py
    outs:
      - data/processed/images
      - data/processed/labels

  qa_check:
    cmd: python scripts/qa/run_full_qa.py
    deps:
      - data/processed
      - scripts/qa/run_full_qa.py
    outs:
      - data/qa_reports
    metrics:
      - data/qa_reports/annotation_qa_report.json:
          cache: false

  split_train_val:
    cmd: python scripts/dataset/08_split_train_val.py --ratio 0.85
    deps:
      - data/processed
      - scripts/dataset/08_split_train_val.py

  train_yolo11n:
    cmd: python scripts/training/train_yolo11n.py
    deps:
      - data/processed
      - configs/data.yaml
      - configs/training/yolo11n_config.yaml
    outs:
      - models/yolo11n/weights
    metrics:
      - models/yolo11n/results/results.csv:
          cache: false

  evaluate:
    cmd: python scripts/training/evaluate_model.py --model models/yolo11n/weights/best.pt
    deps:
      - models/yolo11n/weights/best.pt
      - configs/data.yaml
    metrics:
      - models/yolo11n/results/evaluation.json:
          cache: false
```

## Key DVC Commands

```bash
# Run full pipeline from scratch
dvc repro

# Run only changed stages
dvc repro --no-run-cache

# View pipeline DAG
dvc dag

# Compare metrics between commits
dvc metrics diff

# Checkout a specific dataset version
dvc checkout dataset-v1.0.0

# Push dataset and models to remote
dvc push

# Show pipeline status
dvc status
```

---

Previous: [qa_pipeline.md](./qa_pipeline.md)

Next: [sample_logs.md](./sample_logs.md)

Related: [../01_executive_implementation_plan/dataset_governance.md](../01_executive_implementation_plan/dataset_governance.md)
