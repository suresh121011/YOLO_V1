# QA Pipeline Scripts

## Purpose

Master QA orchestrator and individual check templates for dataset validation.

## Dependencies

Reads:
- dataset_templates.md

Used By:
- dvc_pipeline.md
- release_checklists.md

Related:
- ../01_executive_implementation_plan/dataset_governance.md

---

## 6.1 Master QA Orchestrator

```python
# scripts/qa/run_full_qa.py

"""
Master QA orchestrator. Runs all 8 validation checks sequentially
and generates a comprehensive report.

Exit code 0 = all checks pass
Exit code 1 = critical errors found
Exit code 2 = warnings only
"""

import json
import sys
from pathlib import Path

CHECKS = [
    ("Missing Labels",       "check_missing_labels"),
    ("Empty Labels",         "check_empty_labels"),
    ("Class Consistency",    "check_class_consistency"),
    ("BBox Validity",        "check_bbox_validity"),
    ("Duplicates",           "check_duplicates"),
    ("Train/Val Leakage",    "check_train_val_leakage"),
    ("Class Distribution",   "check_class_distribution"),
    ("Image Quality",        "check_image_quality"),
]

def run_all(dataset_path: str, report_path: str):
    results = {"passed": 0, "warnings": 0, "critical": 0, "checks": {}}

    for name, module_name in CHECKS:
        module = __import__(module_name)
        result = module.run(dataset_path)
        results["checks"][name] = result

        if   result["status"] == "PASS":     results["passed"]   += 1
        elif result["status"] == "WARNING":  results["warnings"] += 1
        elif result["status"] == "CRITICAL": results["critical"] += 1

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"QA RESULTS: {results['passed']} passed, "
          f"{results['warnings']} warnings, {results['critical']} critical")

    if results["critical"] > 0:
        print("❌ CRITICAL ERRORS FOUND — DO NOT PROCEED TO TRAINING")
        sys.exit(1)
    elif results["warnings"] > 0:
        print("⚠️  Warnings found — review before training")
        sys.exit(2)
    else:
        print("✅ All checks passed")
        sys.exit(0)
```

---

## 6.2 BBox Validity Check

```python
# scripts/qa/check_bbox_validity.py

"""
Validates all bounding box annotations:
- All coordinates in [0, 1]
- Width and height > 0
- Class IDs in valid range (0-22)
"""

from pathlib import Path

VALID_CLASS_RANGE = range(0, 23)

def run(dataset_path: str) -> dict:
    labels_dir = Path(dataset_path) / "labels"
    errors = []
    total_boxes = 0

    for label_file in labels_dir.rglob("*.txt"):
        for line_num, line in enumerate(label_file.read_text().strip().split("\n"), 1):
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) != 5:
                errors.append({
                    "file": str(label_file), "line": line_num,
                    "error": f"Expected 5 values, got {len(parts)}"
                })
                continue

            total_boxes += 1
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])

            if class_id not in VALID_CLASS_RANGE:
                errors.append({"file": str(label_file), "line": line_num,
                                "error": f"Invalid class ID: {class_id}"})

            for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
                if not (0.0 <= val <= 1.0):
                    errors.append({"file": str(label_file), "line": line_num,
                                    "error": f"{name}={val} out of [0,1] range"})

            if w <= 0 or h <= 0:
                errors.append({"file": str(label_file), "line": line_num,
                                "error": f"Zero/negative size: w={w}, h={h}"})

    return {
        "status": "CRITICAL" if errors else "PASS",
        "total_boxes": total_boxes,
        "error_count": len(errors),
        "errors": errors[:50],  # Cap at 50 for readability
    }
```

---

## 6.3 Class Distribution Check

```python
# scripts/qa/check_class_distribution.py

"""
Verifies that:
- All 23 classes have at least MIN_COUNT instances
- No class exceeds MAX_RATIO of the total (imbalance guard)
"""

from pathlib import Path
from collections import defaultdict

MIN_COUNT = 200        # Minimum instances per class
MAX_RATIO = 0.20       # No class should be > 20% of total
NUM_CLASSES = 23

CLASS_NAMES = {
    0: "person", 1: "face", 2: "medicine_strip", 3: "medicine_bottle",
    4: "water_bottle", 5: "knife", 6: "stove", 7: "gas_cylinder",
    8: "passport", 9: "book", 10: "charger", 11: "wire",
    12: "laptop", 13: "monitor", 14: "cupboard", 15: "door",
    16: "chair", 17: "bed", 18: "toilet", 19: "sink",
    20: "wet_floor", 21: "walking_stick", 22: "support_handle",
}

def run(dataset_path: str) -> dict:
    counts = defaultdict(int)
    labels_dir = Path(dataset_path) / "labels"

    for label_file in labels_dir.rglob("*.txt"):
        for line in label_file.read_text().strip().split("\n"):
            if line.strip():
                class_id = int(line.split()[0])
                counts[class_id] += 1

    total = sum(counts.values())
    warnings, errors = [], []

    for class_id in range(NUM_CLASSES):
        count = counts.get(class_id, 0)
        name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        if count < MIN_COUNT:
            errors.append(f"{name} (id={class_id}): {count} < {MIN_COUNT} minimum")
        ratio = count / total if total > 0 else 0
        if ratio > MAX_RATIO:
            warnings.append(f"{name}: {ratio:.1%} of dataset (max {MAX_RATIO:.0%})")

    status = "CRITICAL" if errors else ("WARNING" if warnings else "PASS")
    return {"status": status, "class_counts": dict(counts), "errors": errors, "warnings": warnings}
```

---

Previous: [training_scripts.md](./training_scripts.md)

Next: [dvc_pipeline.md](./dvc_pipeline.md)

Related: [../01_executive_implementation_plan/dataset_governance.md](../01_executive_implementation_plan/dataset_governance.md)
