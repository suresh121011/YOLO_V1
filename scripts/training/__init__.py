"""
scripts.training — Model Training and Evaluation Scripts
=========================================================

Script inventory:
    train_yolo.py          — YOLO11 training pipeline (config-driven, W&B optional)
    evaluate_model.py      — Evaluate trained model on held-out test set (Stage 5)
    export_model.py        — Export to ONNX and TFLite for deployment (Stage 6)
    benchmark_latency.py   — Measure inference latency on target device (Stage 7)

Training outputs → models/yolo11n/
    weights/best.pt        — Best checkpoint (val mAP50)
    weights/last.pt        — Final checkpoint
    results/metrics.json   — Final metrics (DVC tracked)
    results/results.csv    — Per-epoch metrics
"""
