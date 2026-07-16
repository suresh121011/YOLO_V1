"""System smoke test: one real mitigated training epoch on the smoke dataset.

Env-gated — NEVER runs in CI (no DVC data there, and a real training epoch
does not belong in the unit/integration gate):

    RUN_TRAINING_SMOKE=1 pytest tests/system/test_training_smoke.py -m system

Requires the repo's real smoke dataset (dvc repro) and the completeness
artifact. Launches training through the production CLI in a subprocess (the
exact invocation users run, with proper __main__ multiprocessing handling on
Windows) and asserts: exit 0, weights written, finite losses, and the
mitigation block present in metrics.json.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.system,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("RUN_TRAINING_SMOKE") != "1",
        reason="real-training smoke is opt-in: set RUN_TRAINING_SMOKE=1",
    ),
]


def _write_smoke_config(tmp_path: Path, enabled: bool, name: str) -> Path:
    """Derive a 1-epoch smoke config from the shipped yolo11n config."""
    with open(REPO_ROOT / "configs" / "training" / "yolo11n_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["training"].update({"epochs": 1, "imgsz": 320, "batch": 8})
    cfg["output"].update({"project": str(tmp_path / "models"), "name": name, "plots": False})
    cfg["missing_annotation_mitigation"]["enabled"] = enabled
    if enabled:
        # Gate G8 (strict): mixing augmentations must be off under mitigation.
        cfg["augmentation"].update({"mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0})
    path = tmp_path / f"{name}_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _assert_finite_losses(results_csv: Path) -> None:
    """Every loss column in Ultralytics' results.csv must be finite."""
    with open(results_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, f"no epochs recorded in {results_csv}"
    for row in rows:
        for column, value in row.items():
            if "loss" in column:
                assert math.isfinite(float(value)), f"non-finite {column}={value}"


def test_one_epoch_mitigated_training_produces_finite_losses(tmp_path: Path) -> None:
    config = _write_smoke_config(tmp_path, enabled=True, name="m35_mitigated")
    proc = subprocess.run(
        [sys.executable, "scripts/training/train_yolo.py", "--config", str(config)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,
    )
    assert proc.returncode == 0, f"training failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"

    run_dir = tmp_path / "models" / "m35_mitigated"
    assert (run_dir / "weights" / "last.pt").exists()
    _assert_finite_losses(run_dir / "results.csv")

    metrics = json.loads((run_dir / "results" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["mitigation"]["enabled"] is True
    assert metrics["mitigation"]["images_covered"] > 0

    # The masked trainer must have announced itself and logged mask stats.
    assert "Missing-annotation mitigation ACTIVE" in proc.stdout + proc.stderr
    assert "Mask stats epoch" in proc.stdout + proc.stderr
