"""
src.dataset.release.manifest — Release Manifest (ADR-P5-07)
===============================================================

``data/releases/dataset-vX.Y.Z/release_manifest.json`` is the immutable
record of one release: what gates it passed, what every input artifact
hashed to, and enough reproducibility metadata (python/dvc versions, seed,
param-file hashes) to explain "what exactly did this build consist of" long
after ``data/merged``/``data/processed`` have moved on. Written once by
``18_make_release.py make`` after :func:`~src.dataset.release.gates.evaluate_release`
reports a non-FAIL verdict; re-checked (not rewritten) by ``18_make_release.py
verify``.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.dataset.manifest import _JsonManifest, utc_now_iso
from src.dataset.release.gates import ReleaseReport
from src.utils.config_helpers import load_yaml
from src.utils.dataset_utils import compute_file_hash
from src.utils.report_utils import git_commit_short

RELEASE_SCHEMA_VERSION = 1
RELEASE_MANIFEST_FILENAME = "release_manifest.json"


def _dvc_version() -> str:
    """Best-effort ``dvc --version`` output (never raises)."""
    try:
        out = subprocess.run(
            ["dvc", "--version"], capture_output=True, text=True, timeout=15, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _hash_if_exists(path: Path) -> str:
    """sha256 of ``path``, or ``""`` when it does not exist."""
    return compute_file_hash(path) if path.exists() else ""


@dataclass
class ReleaseManifest(_JsonManifest):
    """Immutable snapshot record for one dataset release.

    Attributes:
        version:                 ``dataset-vX.Y.Z``.
        created_at:               ISO-8601 UTC timestamp.
        git_commit:               Short commit hash the release was cut from.
        git_tag:                  Tag name (== version; confirmed by RG5).
        dvc_lock_sha256:          Hash of ``dvc.lock`` at release time.
        mode:                     Acquisition mode (``full`` for every real release).
        split_strategy:           ``configs/dataset_split_config.yaml`` strategy in effect.
        taxonomy_fingerprint:     Live taxonomy fingerprint at release time.
        counts:                  {images_total, by_split, by_source, instances_per_class}.
        gates:                   Per-gate {gate_id, name, status, details} (from ReleaseReport).
        artifact_hashes:          {completeness, qa_report, coverage_report, quality_report,
                                  merged_manifest, ledger} -> sha256 (``""`` if absent).
        licenses:                {allow_noncommercial, noncommercial_sources,
                                 roboflow_slug_licenses}.
        changelog_entry_present:  Whether RG4 passed.
        dvc_push_verified:        Whether RG6 passed.
        reproducibility:          {python, dvc, seed, params_files_sha256}.
    """

    version: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    git_commit: str = ""
    git_tag: str = ""
    dvc_lock_sha256: str = ""
    mode: str = ""
    split_strategy: str = ""
    taxonomy_fingerprint: str = ""
    counts: dict[str, Any] = field(default_factory=dict)
    gates: list[dict[str, Any]] = field(default_factory=list)
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    licenses: dict[str, Any] = field(default_factory=dict)
    changelog_entry_present: bool = False
    dvc_push_verified: bool = False
    reproducibility: dict[str, Any] = field(default_factory=dict)
    schema_version: int = RELEASE_SCHEMA_VERSION


def _gate_status(report: ReleaseReport, gate_id: str) -> str | None:
    """Status of one gate in the report, or ``None`` if it wasn't evaluated."""
    for result in report.results:
        if result.gate_id == gate_id:
            return result.status
    return None


def build_release_manifest(
    report: ReleaseReport,
    quality_report: dict[str, Any] | None,
    completeness_path: Path = Path("data/processed/completeness.json"),
    qa_report_path: Path = Path("data/qa_reports/annotation_qa_report.json"),
    coverage_report_path: Path = Path("data/qa_reports/coverage_report.json"),
    quality_report_path: Path = Path("data/qa_reports/dataset_quality_report.json"),
    merged_manifest_path: Path = Path("data/merged/merged_manifest.json"),
    ledger_path: Path = Path("data/annotation/verification_ledger.json"),
    dvc_lock_path: Path = Path("dvc.lock"),
    split_config_path: Path = Path("configs/dataset_split_config.yaml"),
    sources_mode: str = "",
    allow_noncommercial: bool = True,
    noncommercial_sources: list[str] | None = None,
    roboflow_slug_licenses: dict[str, str] | None = None,
    params_files: tuple[Path, ...] = (
        Path("configs/data.yaml"),
        Path("configs/dataset_sources.yaml"),
        Path("configs/dataset_split_config.yaml"),
        Path("configs/annotation.yaml"),
        Path("configs/release.yaml"),
    ),
) -> ReleaseManifest:
    """Assemble a :class:`ReleaseManifest` from an evaluated release report.

    Deliberately a pure aggregation over already-generated artifacts (like
    ``quality.py``'s L5 report) — every count/hash here is read, not
    recomputed, so the manifest can never drift from what it describes.

    Args:
        report:                 A non-FAIL :class:`ReleaseReport` (the caller
                                enforces this — this function only assembles).
        quality_report:         Parsed ``dataset_quality_report.json`` (for
                                ``counts``), or ``None`` if RG3 wasn't required.
        sources_mode, allow_noncommercial, noncommercial_sources,
        roboflow_slug_licenses: Pulled by the caller from the already-loaded
                                sources config / license gate inputs.
    """
    scale = (quality_report or {}).get("dataset_scale", {})
    counts = {
        "images_total": scale.get("images_total", 0),
        "by_split": scale.get("images_by_split", {}),
        "by_source": scale.get("images_by_source", {}),
        "instances_per_class": scale.get("instances_per_class", {}),
    }

    split_cfg = load_yaml(split_config_path) if split_config_path.exists() else {}
    split_section = split_cfg.get("split", {}) if isinstance(split_cfg, dict) else {}

    artifact_hashes = {
        "completeness": _hash_if_exists(completeness_path),
        "qa_report": _hash_if_exists(qa_report_path),
        "coverage_report": _hash_if_exists(coverage_report_path),
        "quality_report": _hash_if_exists(quality_report_path),
        "merged_manifest": _hash_if_exists(merged_manifest_path),
        "ledger": _hash_if_exists(ledger_path),
    }

    params_files_sha256 = {p.as_posix(): _hash_if_exists(p) for p in params_files if p.exists()}

    taxonomy_fp = ""
    completeness_taxonomy = (quality_report or {}).get("taxonomy_fingerprint")
    if isinstance(completeness_taxonomy, str):
        taxonomy_fp = completeness_taxonomy

    return ReleaseManifest(
        version=report.version,
        git_commit=git_commit_short(),
        git_tag=report.version,
        dvc_lock_sha256=_hash_if_exists(dvc_lock_path),
        mode=sources_mode,
        split_strategy=str(split_section.get("strategy", "")),
        taxonomy_fingerprint=taxonomy_fp,
        counts=counts,
        gates=report.to_dict()["gates"],
        artifact_hashes=artifact_hashes,
        licenses={
            "allow_noncommercial": allow_noncommercial,
            "noncommercial_sources": sorted(noncommercial_sources or []),
            "roboflow_slug_licenses": dict(roboflow_slug_licenses or {}),
        },
        changelog_entry_present=_gate_status(report, "RG4") == "pass",
        dvc_push_verified=_gate_status(report, "RG6") == "pass",
        reproducibility={
            "python": sys.version.split()[0],
            "dvc": _dvc_version(),
            "seed": split_section.get("seed"),
            "params_files_sha256": params_files_sha256,
        },
    )
