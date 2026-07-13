# Dataset Pipeline Scripts

## Purpose

Dataset acquisition, class remapping, and merge scripts for building the 23-class training dataset.

## Dependencies

Reads:
- yaml_examples.md

Used By:
- training_scripts.md
- dvc_pipeline.md

Related:
- ../01_executive_implementation_plan/dataset_governance.md
- annotation_guide.md

---

## 3.1 COCO Subset Extraction

```python
# scripts/dataset/01_download_coco_subset.py
# Downloads COCO 2017 annotations, filters for target classes + indoor scenes

"""
COCO -> Our ID Mapping:
  person (1)  -> 0     bottle (44)  -> 4     knife (49)  -> 5
  laptop (73) -> 12    tv (72)      -> 13    chair (62)  -> 16
  bed (65)    -> 17    toilet (70)  -> 18    sink (81)   -> 19
  book (84)   -> 9

Steps:
1. Download COCO 2017 annotations (instances_train2017.json)
2. For each target COCO class, extract image IDs
3. Cross-reference with COCO 'stuff' annotations for indoor filtering
4. Download only matching images (saves bandwidth)
5. Convert COCO JSON annotations to YOLO format
6. Remap class IDs to our taxonomy
7. Save to data/raw/coco_filtered/
"""

COCO_TO_OURS = {1: 0, 44: 4, 49: 5, 73: 12, 72: 13, 62: 16, 65: 17, 70: 18, 81: 19, 84: 9}

INDOOR_STUFF_CATEGORIES = [
    "floor-wood", "floor-tile", "floor-marble", "floor-other",
    "wall-brick", "wall-concrete", "wall-other", "wall-panel",
    "ceiling-merged", "carpet", "rug", "cabinet-merged",
    "counter", "curtain", "door-stuff", "table-merged",
]

# Per-class caps to prevent imbalance
CLASS_CAPS = {0: 800, 16: 500}  # person: 800, chair: 500
```

---

## 3.2 Class Remapping Engine

```python
# scripts/dataset/05_remap_classes.py

"""
Unified class remapping engine.
Maps any source dataset class ID to our 0-22 taxonomy.

Usage:
  python 05_remap_classes.py --source coco --input data/raw/coco_filtered/labels/
"""

from pathlib import Path

REMAP_TABLES = {
    "coco":        {1: 0, 44: 4, 49: 5, 73: 12, 72: 13, 62: 16, 65: 17, 70: 18, 81: 19, 84: 9},
    "openimages":  {"Door": 15, "Cupboard": 14, "Gas stove": 6},
    "wider_face":  {0: 1},
}

def remap_label_file(filepath: str, source: str) -> None:
    """Remap all class IDs in a YOLO label file."""
    table = REMAP_TABLES[source]
    lines = Path(filepath).read_text().strip().split("\n")
    remapped = []
    for line in lines:
        parts = line.split()
        old_id = int(parts[0]) if parts[0].isdigit() else parts[0]
        new_id = table.get(old_id)
        if new_id is not None:
            parts[0] = str(new_id)
            remapped.append(" ".join(parts))
    Path(filepath).write_text("\n".join(remapped) + "\n")
```

---

## 3.3 Indoor Image Filter

```python
# scripts/dataset/06_filter_indoor_images.py

"""
Filter out outdoor images from COCO/OpenImages downloads.
Uses scene classification heuristics:
  - COCO 'stuff' annotations (best signal)
  - Image aspect ratio (portrait = likely indoor device photo)
  - Average pixel brightness (very bright = likely outdoor)
"""

import cv2
import numpy as np
from pathlib import Path


BRIGHTNESS_OUTDOOR_THRESHOLD = 160  # Average pixel value
ASPECT_RATIO_PORTRAIT_MIN = 0.6     # w/h < 0.6 = portrait phone photo

def is_likely_indoor(image_path: str) -> bool:
    img = cv2.imread(image_path)
    if img is None:
        return False
    h, w = img.shape[:2]

    # Portrait photos are typically indoor
    if (w / h) < ASPECT_RATIO_PORTRAIT_MIN:
        return True

    # Very bright images are likely outdoor
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.mean() > BRIGHTNESS_OUTDOOR_THRESHOLD:
        return False

    return True
```

---

## 3.4 Merge and Deduplicate

```python
# scripts/dataset/07_merge_datasets.py

"""
Merge all raw sources into data/processed/.
Steps:
  1. Collect all images from raw sources
  2. Perceptual hash (pHash) each image
  3. Skip duplicates (Hamming distance < 5)
  4. Copy image + label pair to data/processed/images/ and labels/
  5. Log statistics per source
"""

import imagehash
from PIL import Image
from pathlib import Path

HAMMING_THRESHOLD = 5  # Images with distance < 5 are considered duplicates

def get_phash(image_path: str) -> imagehash.ImageHash:
    return imagehash.phash(Image.open(image_path))

def merge_all(raw_dirs: list[str], output_dir: str) -> dict:
    seen_hashes: list[imagehash.ImageHash] = []
    stats = {"total": 0, "duplicates": 0, "accepted": 0}

    for raw_dir in raw_dirs:
        for img_path in Path(raw_dir).rglob("*.jpg"):
            stats["total"] += 1
            h = get_phash(str(img_path))

            # Check for near-duplicate
            is_dup = any(abs(h - seen) < HAMMING_THRESHOLD for seen in seen_hashes)
            if is_dup:
                stats["duplicates"] += 1
                continue

            seen_hashes.append(h)
            stats["accepted"] += 1
            # Copy image + label to output_dir/...

    return stats
```

---

Previous: [python_examples.md](./python_examples.md)

Next: [training_scripts.md](./training_scripts.md)

Related: [annotation_guide.md](./annotation_guide.md), [dvc_pipeline.md](./dvc_pipeline.md)
