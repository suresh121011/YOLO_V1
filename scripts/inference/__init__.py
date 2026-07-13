"""
scripts.inference — Inference and Benchmarking Scripts
=======================================================

Script inventory:
    test_video.py           — Multi-source YOLO inference (webcam/video/folder)
                              Annotated output video, per-frame JSON, FPS/latency metrics.
    run_pipeline.py         — Launch the full pipeline on a camera stream (Stage 6)
    benchmark_pipeline.py   — Measure end-to-end latency on target device (Stage 7)
    validate_scenarios.py   — Run the 10 standard field test scenarios (Stage 7)

Usage:
    python scripts/inference/test_video.py --source 0
    python scripts/inference/test_video.py --source video.mp4 --output-dir outputs/
"""
