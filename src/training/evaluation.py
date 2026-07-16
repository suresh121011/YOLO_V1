"""
src.training.evaluation — Baseline-vs-Mitigated Evaluation Framework (M4)
=========================================================================

Reproducible evaluation of trained checkpoints on the project dataset:
per-class precision/recall/mAP, confusion-matrix export, and a delta report
comparing a baseline run against a mitigated run. Reports follow the house
JSON+CSV+MD triplet convention (src/utils/report_utils.write_all_formats)
and embed reproducibility metadata (git commit, library versions, weights,
split, seed).

Caveat documented for readers of the numbers: validation labels on public
sources are themselves partially annotated, so absolute metrics UNDERESTIMATE
true performance on untrusted classes (an unlabeled-but-detected face counts
as a false positive). Baseline-vs-mitigated DELTAS on the same split remain
meaningful; the unbiased verdict comes from the fully-annotated custom eval
set in Phase 5+.

torch/ultralytics are imported lazily — this module is importable (and its
report shaping unit-testable) in torch-less environments.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.utils.config_helpers import get_class_names_from_data_yaml, load_data_config
from src.utils.report_utils import save_json_report, timestamp_str, write_all_formats

logger = logging.getLogger(__name__)

COMPARISON_REPORT_BASENAME = "evaluation_comparison"


@dataclass(frozen=True)
class EvalRunSpec:
    """One checkpoint evaluation to execute.

    Attributes:
        weights:   Path to the .pt checkpoint.
        label:     Run label used in reports (e.g. "baseline", "mitigated").
        data_yaml: Dataset YAML (taxonomy + split paths).
        split:     Dataset split to evaluate ("val" or "test").
        device:    Compute device string.
        imgsz:     Evaluation image size.
        seed:      Recorded for reproducibility metadata.
    """

    weights: Path
    label: str
    data_yaml: Path = Path("configs/data.yaml")
    split: str = "val"
    device: str = "cpu"
    imgsz: int = 640
    seed: int = 42


def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall (0.0 when both are 0)."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _environment_metadata() -> dict[str, Any]:
    """Versions + git commit for report reproducibility blocks."""
    meta: dict[str, Any] = {
        "python": sys.version.split()[0],
        "git_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or "unknown",
    }
    try:
        import torch
        import ultralytics

        meta["ultralytics"] = ultralytics.__version__
        meta["torch"] = torch.__version__
    except ImportError:  # pragma: no cover - eval always has them installed
        meta["ultralytics"] = meta["torch"] = "unavailable"
    return meta


def extract_per_class_metrics(results: Any, names: dict[int, str]) -> list[dict[str, Any]]:
    """Extract per-class P/R/mAP50/mAP50-95/F1 rows from a val Results object.

    Classes absent from the split's ground truth produce no row (Ultralytics
    only scores classes present in ``results.box.ap_class_index``).

    Args:
        results: Ultralytics validation results (DetMetrics).
        names:   Taxonomy id → name.

    Returns:
        One dict per scored class, sorted by class id.
    """
    rows: list[dict[str, Any]] = []
    for position, class_id in enumerate(results.box.ap_class_index):
        precision, recall, ap50, ap = results.box.class_result(position)
        rows.append(
            {
                "class_id": int(class_id),
                "class": names.get(int(class_id), f"?{class_id}"),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(f1_score(float(precision), float(recall)), 4),
                "mAP50": round(float(ap50), 4),
                "mAP50_95": round(float(ap), 4),
            }
        )
    return sorted(rows, key=lambda r: r["class_id"])


def extract_aggregate_metrics(results: Any) -> dict[str, float]:
    """Aggregate P/R/F1/mAP metrics from a val Results object."""
    precision = float(results.box.mp)
    recall = float(results.box.mr)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1_score(precision, recall), 4),
        "mAP50": round(float(results.box.map50), 4),
        "mAP50_95": round(float(results.box.map), 4),
    }


def export_confusion_matrix(results: Any, names: dict[int, str], out_path: Path) -> Path:
    """Write the confusion matrix (incl. background row/col) as JSON.

    Args:
        results:  Ultralytics validation results.
        names:    Taxonomy id → name.
        out_path: Destination .json path.

    Returns:
        The written path.
    """
    matrix = results.confusion_matrix.matrix
    labels = [names[i] for i in sorted(names)] + ["background"]
    payload = {
        "labels": labels,
        "note": "rows = predicted, columns = ground truth (Ultralytics convention)",
        "matrix": [[float(cell) for cell in row] for row in matrix],
    }
    return save_json_report(payload, out_path)


def _run_single_eval(spec: EvalRunSpec, out_dir: Path) -> dict[str, Any]:
    """Evaluate one checkpoint and export its artifacts."""
    from ultralytics import YOLO

    if not spec.weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {spec.weights.absolute()}")

    names = get_class_names_from_data_yaml(load_data_config(spec.data_yaml))
    logger.info(f"Evaluating '{spec.label}': {spec.weights} on split '{spec.split}'")
    model = YOLO(str(spec.weights))
    results = model.val(
        data=str(spec.data_yaml),
        split=spec.split,
        imgsz=spec.imgsz,
        device=spec.device,
        plots=False,
        verbose=False,
        workers=0,
        project=str(out_dir / "ultralytics_runs"),
        name=spec.label,
        exist_ok=True,
    )

    confusion_path = export_confusion_matrix(
        results, names, out_dir / f"confusion_matrix_{spec.label}.json"
    )
    summary = {
        "spec": {
            **asdict(spec),
            "weights": spec.weights.as_posix(),
            "data_yaml": spec.data_yaml.as_posix(),
        },
        "aggregate": extract_aggregate_metrics(results),
        "per_class": extract_per_class_metrics(results, names),
        "confusion_matrix": confusion_path.as_posix(),
    }
    save_json_report(summary, out_dir / f"evaluation_{spec.label}.json")
    return summary


def build_delta_report(baseline: dict[str, Any], mitigated: dict[str, Any]) -> dict[str, Any]:
    """Compute mitigated-minus-baseline deltas (aggregate and per class).

    Args:
        baseline:  Summary dict of the baseline run (_run_single_eval output).
        mitigated: Summary dict of the mitigated run.

    Returns:
        Delta report dict with 'aggregate_delta' and 'per_class_delta' rows.
    """
    metric_keys = ("precision", "recall", "f1", "mAP50", "mAP50_95")
    aggregate_delta = {
        key: round(mitigated["aggregate"][key] - baseline["aggregate"][key], 4)
        for key in metric_keys
    }

    baseline_by_class = {row["class"]: row for row in baseline["per_class"]}
    mitigated_by_class = {row["class"]: row for row in mitigated["per_class"]}
    per_class_delta: list[dict[str, Any]] = []
    for class_name in sorted(set(baseline_by_class) | set(mitigated_by_class)):
        base_row = baseline_by_class.get(class_name)
        mit_row = mitigated_by_class.get(class_name)
        row: dict[str, Any] = {"class": class_name}
        for key in metric_keys:
            row[f"baseline_{key}"] = base_row[key] if base_row else None
            row[f"mitigated_{key}"] = mit_row[key] if mit_row else None
            row[f"delta_{key}"] = (
                round(mit_row[key] - base_row[key], 4) if base_row and mit_row else None
            )
        per_class_delta.append(row)

    return {"aggregate_delta": aggregate_delta, "per_class_delta": per_class_delta}


def run_evaluation(specs: list[EvalRunSpec], out_dir: Path) -> Path:
    """Evaluate checkpoints and write the comparison report triplet.

    With exactly two specs labeled 'baseline' and 'mitigated', the report
    includes the delta section; any other spec set produces summaries only.

    Args:
        specs:   Evaluations to run.
        out_dir: Report/artifact directory (created if needed).

    Returns:
        Path to the comparison report JSON.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {spec.label: _run_single_eval(spec, out_dir) for spec in specs}

    report: dict[str, Any] = {
        "generated_at": timestamp_str(),
        "environment": _environment_metadata(),
        "runs": summaries,
        "caveat": (
            "Public-source validation labels are partially annotated; absolute "
            "metrics underestimate untrusted classes. Deltas on the same split "
            "are meaningful; unbiased numbers require the custom eval set."
        ),
    }
    if {"baseline", "mitigated"} <= set(summaries):
        report["delta"] = build_delta_report(summaries["baseline"], summaries["mitigated"])

    csv_rows: list[dict[str, Any]] = []
    for label, summary in summaries.items():
        for row in summary["per_class"]:
            csv_rows.append({"run": label, **row})

    sections: list[dict[str, Any]] = [
        {
            "heading": "Aggregate metrics",
            "table": {
                "headers": ["Run", "P", "R", "F1", "mAP50", "mAP50-95"],
                "rows": [
                    [
                        label,
                        summary["aggregate"]["precision"],
                        summary["aggregate"]["recall"],
                        summary["aggregate"]["f1"],
                        summary["aggregate"]["mAP50"],
                        summary["aggregate"]["mAP50_95"],
                    ]
                    for label, summary in summaries.items()
                ],
            },
        }
    ]
    if "delta" in report:
        sections.append(
            {
                "heading": "Mitigated − baseline (per class)",
                "content": report["caveat"],
                "table": {
                    "headers": ["Class", "ΔP", "ΔR", "ΔF1", "ΔmAP50", "ΔmAP50-95"],
                    "rows": [
                        [
                            row["class"],
                            row["delta_precision"],
                            row["delta_recall"],
                            row["delta_f1"],
                            row["delta_mAP50"],
                            row["delta_mAP50_95"],
                        ]
                        for row in report["delta"]["per_class_delta"]
                    ],
                },
            }
        )

    paths = write_all_formats(
        report_data=report,
        csv_rows=csv_rows,
        md_title="Baseline vs Mitigated — Evaluation Report",
        md_sections=sections,
        output_dir=out_dir,
        base_name=COMPARISON_REPORT_BASENAME,
        md_metadata={
            "Runs": ", ".join(summaries),
            "Commit": report["environment"]["git_commit"],
            "ultralytics": report["environment"].get("ultralytics", "?"),
        },
    )
    logger.info(f"Evaluation comparison written: {paths['markdown']}")
    return paths["json"]
