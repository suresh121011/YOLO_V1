# Data Flow — Training Pipeline

## Purpose

Training pipeline architecture showing data sources, processing stages, and model output.

## Dependencies

Reads:
- system_architecture.md

Used By:
- interfaces.md

Related:
- ../03_engineering_appendix/dataset_templates.md

---

## Training Pipeline Architecture

```mermaid
graph LR
    subgraph "Data Sources"
        COCO["COCO 2017\n(person, knife, bottle,\nchair, bed, toilet, sink,\nlaptop, book)"]
        OI["Open Images V7\n(door, cupboard, stove)"]
        RF["Roboflow Universe\n(medicine_bottle, charger,\nwire, gas_cylinder)"]
        WF["WIDER FACE\n(face)"]
        CUSTOM["Custom Indian Home\n(medicine_strip, gas_cylinder,\nwet_floor, walking_stick,\nsupport_handle, passport,\nIndian stove, Indian cupboard)"]
    end

    subgraph "Processing Pipeline"
        REMAP["Class Remapper\n(→ IDs 0–22)"]
        FILTER2["Indoor Filter\n(Remove outdoor scenes)"]
        MERGE["Dataset Merger\n(Deduplication)"]
        QA["QA Pipeline\n(8 automated checks)"]
        SPLIT["Train/Val Split\n(85%/15% stratified)"]
    end

    subgraph "Training"
        TRAIN11N["YOLO11n Training\n(150 epochs, AdamW)"]
        TRAIN11S["YOLO11s Training\n(200 epochs, reference)"]
        EVAL["Evaluation\n(mAP50, Recall, FPS)"]
        EXPORT["Model Export\n(ONNX · TFLite · CoreML)"]
    end

    COCO & OI & RF & WF & CUSTOM --> REMAP
    REMAP --> FILTER2 --> MERGE --> QA --> SPLIT
    SPLIT --> TRAIN11N & TRAIN11S
    TRAIN11N & TRAIN11S --> EVAL --> EXPORT
```

## Data Processing Stages

| Stage | Script | Input | Output |
|:------|:-------|:------|:-------|
| COCO extraction | `01_download_coco_subset.py` | COCO 2017 API | `data/raw/coco_filtered/` |
| Open Images | `02_download_openimages_subset.py` | OI V7 API | `data/raw/openimages_filtered/` |
| Roboflow | `03_download_roboflow_datasets.py` | Roboflow Universe | `data/raw/roboflow_imports/` |
| WIDER FACE | `04_download_wider_face.py` | WIDER FACE | `data/raw/wider_face/` |
| Class remap | `05_remap_classes.py` | Raw labels | Remapped labels (IDs 0–22) |
| Indoor filter | `06_filter_indoor_images.py` | All images | Indoor-only subset |
| Merge | `07_merge_datasets.py` | All raw sources | `data/processed/` |
| QA | `run_full_qa.py` | `data/processed/` | `data/qa_reports/` |
| Split | `08_split_train_val.py` | Processed data | Train/Val splits |

---

Previous: [system_architecture.md](./system_architecture.md)

Next: [interfaces.md](./interfaces.md)

Related: [../03_engineering_appendix/dataset_templates.md](../03_engineering_appendix/dataset_templates.md)
