# Training, Evaluation & Export Scripts

## Purpose

YOLO11n training script, evaluation with safety-class drill-down, and model export to edge formats.

## Dependencies

Reads:
- yaml_examples.md
- dataset_templates.md

Used By:
- dvc_pipeline.md
- release_checklists.md

Related:
- ../02_technical_architecture_specification/data_contracts.md

---

## 4.1 YOLO11n Training Script

```python
# scripts/training/train_yolo11n.py

from ultralytics import YOLO
from pathlib import Path
import yaml

# Load config
config_path = "configs/training/yolo11n_config.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

model = YOLO(cfg["model"]["base"])

results = model.train(
    data=cfg["training"]["data"],
    epochs=cfg["training"]["epochs"],
    imgsz=cfg["training"]["imgsz"],
    batch=cfg["training"]["batch"],
    patience=cfg["training"]["patience"],
    optimizer=cfg["training"]["optimizer"],
    lr0=cfg["training"]["lr0"],
    lrf=cfg["training"]["lrf"],
    momentum=cfg["training"]["momentum"],
    weight_decay=cfg["training"]["weight_decay"],
    warmup_epochs=cfg["training"]["warmup_epochs"],
    warmup_bias_lr=cfg["training"]["warmup_bias_lr"],
    # Augmentation
    hsv_h=cfg["augmentation"]["hsv_h"],
    hsv_s=cfg["augmentation"]["hsv_s"],
    hsv_v=cfg["augmentation"]["hsv_v"],
    degrees=cfg["augmentation"]["degrees"],
    translate=cfg["augmentation"]["translate"],
    scale=cfg["augmentation"]["scale"],
    flipud=cfg["augmentation"]["flipud"],
    fliplr=cfg["augmentation"]["fliplr"],
    mosaic=cfg["augmentation"]["mosaic"],
    mixup=cfg["augmentation"]["mixup"],
    copy_paste=cfg["augmentation"]["copy_paste"],
    close_mosaic=cfg["training"]["close_mosaic"],
    # Output
    project=cfg["output"]["project"],
    name=cfg["output"]["name"],
    exist_ok=cfg["output"]["exist_ok"],
    save=cfg["output"]["save"],
    save_period=cfg["output"]["save_period"],
    val=cfg["output"]["val"],
    plots=cfg["output"]["plots"],
    verbose=cfg["output"]["verbose"],
)

# Generate evaluation artifacts
metrics = model.val()
print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
```

---

## 4.2 Evaluation Script (With Safety-Class Drill-Down)

```python
# scripts/training/evaluate_model.py

"""
Generates:
  - confusion_matrix.png
  - PR_curve.png
  - F1_curve.png
  - results.csv
  - Per-class AP breakdown
  - Safety-critical class drill-down report
"""

import argparse
from ultralytics import YOLO

SAFETY_CLASSES = ["knife", "stove", "gas_cylinder", "wire", "wet_floor",
                  "medicine_strip", "medicine_bottle"]

def evaluate(model_path: str, data_path: str):
    model = YOLO(model_path)
    metrics = model.val(data=data_path, plots=True, save_json=True)

    print("\n=== OVERALL METRICS ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    print("\n=== SAFETY CLASS DRILL-DOWN ===")
    for i, name in enumerate(model.names.values()):
        if name in SAFETY_CLASSES:
            ap50      = metrics.box.ap50[i]
            recall    = metrics.box.r[i] if hasattr(metrics.box, 'r') else "N/A"
            precision = metrics.box.p[i] if hasattr(metrics.box, 'p') else "N/A"
            print(f"  {name:20s}  AP50={ap50:.3f}  P={precision}  R={recall}")
            if isinstance(ap50, float) and ap50 < 0.80:
                print(f"    ⚠️  BELOW SAFETY THRESHOLD (0.80)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="configs/data.yaml")
    args = parser.parse_args()
    evaluate(args.model, args.data)
```

---

## 4.3 Model Export Script

```python
# scripts/training/export_model.py

"""
Export trained YOLO model to edge deployment formats.

Supported targets:
  - ONNX (cross-platform)
  - TFLite (Android)
  - TFLite INT8 (Android quantized)
  - CoreML (iOS)
  - TensorRT (Jetson / GPU edge)
"""

import argparse
from ultralytics import YOLO


def export_model(model_path: str, formats: list[str], data_yaml: str):
    model = YOLO(model_path)

    for fmt in formats:
        print(f"\n{'='*50}")
        print(f"Exporting to {fmt.upper()}...")

        if fmt == "onnx":
            model.export(format="onnx", opset=17, simplify=True, dynamic=True)

        elif fmt == "tflite":
            model.export(format="tflite")

        elif fmt == "tflite_int8":
            model.export(format="tflite", int8=True, data=data_yaml)

        elif fmt == "coreml":
            model.export(format="coreml", nms=True)

        elif fmt == "tensorrt":
            model.export(format="engine", half=True)

        print(f"✅ {fmt.upper()} export complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yolo11n/weights/best.pt")
    parser.add_argument("--formats", nargs="+",
                        default=["onnx", "tflite", "tflite_int8"],
                        choices=["onnx", "tflite", "tflite_int8", "coreml", "tensorrt"])
    parser.add_argument("--data", default="configs/data.yaml")
    args = parser.parse_args()
    export_model(args.model, args.formats, args.data)
```

---

Previous: [dataset_templates.md](./dataset_templates.md)

Next: [qa_pipeline.md](./qa_pipeline.md)

Related: [dvc_pipeline.md](./dvc_pipeline.md)
