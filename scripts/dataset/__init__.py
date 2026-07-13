"""
scripts.dataset — Dataset Acquisition and Processing Scripts
============================================================

Script inventory:
    01_download_coco_subset.py       — Download COCO images for target classes
    02_download_openimages_subset.py — Download Open Images V7 subset
    03_download_roboflow_datasets.py — Import Roboflow public datasets
    04_download_wider_face.py        — Download WIDER FACE for face class
    05_remap_classes.py              — Remap source class IDs to 23-class taxonomy
    06_collect_negatives.py          — Collect negative images (empty rooms, outdoors)
    07_merge_datasets.py             — Merge all sources into processed/
    split_dataset.py                 — Group-aware 80/10/10 train/val/test split
    generate_splits.py               — Full split + statistics pipeline orchestrator
    dataset_stats.py                 — Per-class annotation statistics (CSV/JSON/MD)
"""
