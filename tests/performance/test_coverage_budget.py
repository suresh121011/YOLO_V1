"""Performance budget: L4 coverage estimation at 30k-image scale (plan
§Performance budgets — "coverage_report (no inference) | 30k images | ≤ 5 min").

The plan's own test list named ``tests/performance/test_coverage_budget.py``;
it was never created (final-audit Fix-7 gap). Coverage is pure arithmetic over
pinned candidates + completeness + ledger (ADR-P5-06) — no image decode, no
inference — so a synthetic 30k-image completeness + candidates pair isolates
the O(n) aggregation this budget is about. Labels are intentionally absent
(missing label files are treated as zero-annotation by ``build_coverage_report``).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from src.dataset.annotation.coverage import build_coverage_report
from src.dataset.completeness import taxonomy_fingerprint

pytestmark = [
    pytest.mark.performance,
    pytest.mark.slow,
    pytest.mark.skipif(bool(os.environ.get("CI")), reason="No scale budget check on CI"),
]

_BUDGET_SECONDS = 300  # 5 minutes
_SCALE = 30_000
_NAMES = {0: "charger", 1: "wire", 2: "person"}
_NC = 3
_FP = taxonomy_fingerprint(_NC, _NAMES)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_coverage_report_scales_to_30k_within_budget(tmp_path: Path) -> None:
    images = {f"img_{i}.jpg": {"policy": "coco", "split": "train"} for i in range(_SCALE)}
    _write_json(
        tmp_path / "completeness.json",
        {
            "schema_version": 1,
            "taxonomy": {
                "nc": _NC,
                "names": {str(k): v for k, v in _NAMES.items()},
                "fingerprint": _FP,
            },
            # Empty trusted list => every candidate is untrusted + unverified
            # ("unknown"), the discount-by-prior path this budget stresses.
            "policies": {"coco": {"mode": "trusted_list", "trusted_class_ids": []}},
            "images": images,
            "stats": {"images_total": _SCALE},
        },
    )
    _write_json(
        tmp_path / "candidates" / "yolo_world" / "candidates.json",
        {
            "schema_version": 1,
            "run_id": "perf_run",
            "backend": "yolo_world",
            "taxonomy_fingerprint": _FP,
            "images": {
                f"img_{i}.jpg": {
                    "targeted_class_ids": [i % _NC],
                    "detections": [
                        {
                            "class_id": i % _NC,
                            "conf": 0.9,
                            "bbox_xywhn": [0.5, 0.5, 0.2, 0.2],
                            "refined": False,
                            "origin": "perf",
                        }
                    ],
                }
                for i in range(_SCALE)
            },
            "stats": {"images_processed": _SCALE, "detections_total": _SCALE},
        },
    )
    data_yaml = tmp_path / "data.yaml"
    _write_json(data_yaml, {"nc": _NC, "names": {str(k): v for k, v in _NAMES.items()}})
    (tmp_path / "labels").mkdir()

    start = time.perf_counter()
    report = build_coverage_report(
        candidates_root=tmp_path / "candidates",
        ledger_path=tmp_path / "ledger.json",
        completeness_path=tmp_path / "completeness.json",
        processed_labels_root=tmp_path / "labels",
        data_yaml_path=data_yaml,
        iou_match_threshold=0.5,
        estimation_conf={"default": 0.35},
    )
    elapsed = time.perf_counter() - start

    assert report["dataset"]["unknown_objects_total"] == _SCALE
    assert (
        elapsed <= _BUDGET_SECONDS
    ), f"coverage_report took {elapsed:.1f}s for {_SCALE} images (budget {_BUDGET_SECONDS}s)"
