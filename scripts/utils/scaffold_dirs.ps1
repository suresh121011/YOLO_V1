$dirs = @(
    "data/raw/coco_filtered",
    "data/raw/openimages_filtered",
    "data/raw/roboflow_imports",
    "data/raw/wider_face",
    "data/raw/custom_captures",
    "data/raw/negatives",
    "data/processed/images/train",
    "data/processed/images/val",
    "data/processed/images/test",
    "data/processed/labels/train",
    "data/processed/labels/val",
    "data/processed/labels/test",
    "data/qa_reports",
    "logs/pipeline",
    "logs/training",
    "logs/qa",
    "models/yolo11n/weights",
    "models/yolo11n/exports",
    "models/yolo11n/results",
    "models/smolvlm2",
    "models/tts",
    "outputs/visualizations",
    "outputs/benchmarks"
)

foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
    $gk = Join-Path $d ".gitkeep"
    if (-not (Test-Path $gk)) {
        New-Item -ItemType File -Force -Path $gk | Out-Null
    }
    Write-Host "OK: $d"
}
Write-Host "All directories scaffolded successfully."
