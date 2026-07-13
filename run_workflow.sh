#!/bin/bash

echo "============================================================"
echo " Elderly Assistant System — End-to-End Pipeline Workflow"
echo "============================================================"

echo ""
echo "Step 1: Splitting dataset (80/10/10, grouped by capture session)"
python scripts/dataset/generate_splits.py --seed 42

echo ""
echo "Step 2: Running QA checks"
python scripts/qa/check_annotations.py

echo ""
echo "Step 3: Training YOLO11n"
python scripts/training/train_yolo.py --config configs/training/yolo11n_config.yaml

echo ""
echo "Step 4: Running Inference on webcam (source 0)"
echo "Note: Press 'q' to stop inference."
# To run on a video file, change --source 0 to --source path/to/video.mp4
python scripts/inference/test_video.py --source 0 --model models/yolo11n/weights/best.pt --output-dir outputs/

echo ""
echo "Pipeline execution finished."
