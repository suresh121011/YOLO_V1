"""
src.training.preflight — Pre-Training Gates for Mitigation (G1–G8)
==================================================================

Fail-early validation that runs before any Ultralytics work whenever
missing-annotation mitigation is enabled. Every failure names the artifact,
the offending key/file, and the remediation command — training never starts
against inconsistent completeness metadata.

Gates:
    G1  artifact exists, parses, schema supported
    G2  taxonomy fingerprint matches live configs/data.yaml
    G3  every train/val image has a record with the correct split
    G4  artifact self-consistency (policy sanity, orphans; unused → warn)
    G5  environment: ultralytics importable + version window (+ compat canary
        once the masked loss lands)
    G6  mitigation config self-valid
    G7  freshness: recorded input hashes match the files on disk
    G8  mixing augmentations (mosaic/mixup/copy_paste) vs the configured policy

Backward-compat contract: when mitigation is disabled the training script
never calls this module (byte-for-byte stock behavior). The standalone CLI
(scripts/training/preflight_check.py) provides on-demand gate runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.dataset.completeness import (
    find_unused_policies,
    load_completeness,
    taxonomy_fingerprint,
    validate_completeness,
)
from src.dataset.completeness_policies import CompletenessError
from src.training.mitigation_config import MITIGATION_SECTION, MitigationConfig
from src.utils.config_helpers import get_class_names_from_data_yaml, load_data_config
from src.utils.dataset_utils import compute_file_hash, find_image_files

logger = logging.getLogger(__name__)

#: Ultralytics version window the mitigation is designed against
#: (mirrors the requirements.txt pin; G5 enforces it at runtime).
ULTRALYTICS_MIN = (8, 3)
ULTRALYTICS_MAX_EXCLUSIVE = (9, 0)

#: Effective mixing-augmentation values when the training YAML has an
#: ``augmentation:`` section but omits a key — MUST mirror the defaults in
#: scripts/training/train_yolo.py's kwargs assembly.
_TRAIN_YOLO_AUG_DEFAULTS = {"mosaic": 0.8, "mixup": 0.1, "copy_paste": 0.1}

#: Effective values when the YAML has NO augmentation section at all — the
#: script then passes nothing and Ultralytics applies its own defaults.
_ULTRALYTICS_AUG_DEFAULTS = {"mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0}

GATE_STATUS_PASS = "pass"  # noqa: S105 — gate status literal, not a credential
GATE_STATUS_WARN = "warn"
GATE_STATUS_FAIL = "fail"
GATE_STATUS_SKIPPED = "skipped"


@dataclass(frozen=True)
class GateResult:
    """Outcome of one preflight gate."""

    gate_id: str
    name: str
    status: str
    details: str

    def format_line(self) -> str:
        """One-line human-readable rendering."""
        badge = {
            GATE_STATUS_PASS: "PASS",
            GATE_STATUS_WARN: "WARN",
            GATE_STATUS_FAIL: "FAIL",
            GATE_STATUS_SKIPPED: "SKIP",
        }[self.status]
        return f"[{badge}] {self.gate_id} {self.name}: {self.details}"


@dataclass(frozen=True)
class PreflightReport:
    """Aggregated preflight outcome."""

    results: tuple[GateResult, ...]

    @property
    def verdict(self) -> str:
        """PASS, WARN, or FAIL (skipped gates do not affect the verdict)."""
        statuses = {r.status for r in self.results}
        if GATE_STATUS_FAIL in statuses:
            return "FAIL"
        if GATE_STATUS_WARN in statuses:
            return "WARN"
        return "PASS"

    def failures(self) -> list[GateResult]:
        """All failed gates."""
        return [r for r in self.results if r.status == GATE_STATUS_FAIL]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form (report files, logs)."""
        return {
            "verdict": self.verdict,
            "gates": [
                {
                    "gate_id": r.gate_id,
                    "name": r.name,
                    "status": r.status,
                    "details": r.details,
                }
                for r in self.results
            ],
        }

    def format_lines(self) -> list[str]:
        """Human-readable line per gate plus a verdict line."""
        return [r.format_line() for r in self.results] + [f"Preflight verdict: {self.verdict}"]


def effective_mixing_augs(train_cfg: dict[str, Any]) -> dict[str, float]:
    """Resolve the mosaic/mixup/copy_paste values training would actually use.

    Mirrors scripts/training/train_yolo.py: a present ``augmentation:``
    section fills missing keys with the script's defaults; an absent section
    means Ultralytics' own defaults apply (mosaic=1.0!).

    Args:
        train_cfg: Parsed training config dict.

    Returns:
        Key → effective float value for the three mixing augmentations.
    """
    aug_section = train_cfg.get("augmentation") or {}
    if aug_section:
        return {
            key: float(aug_section.get(key, default))
            for key, default in _TRAIN_YOLO_AUG_DEFAULTS.items()
        }
    return dict(_ULTRALYTICS_AUG_DEFAULTS)


def _gate_environment() -> GateResult:
    """G5: ultralytics importable, version in window, loss surface intact."""
    try:
        import ultralytics  # noqa: PLC0415 — heavyweight import, gate-local by design
    except ImportError:
        return GateResult(
            "G5",
            "environment",
            GATE_STATUS_FAIL,
            "ultralytics is not installed — pip install -r requirements.txt",
        )
    raw = getattr(ultralytics, "__version__", "0.0.0")
    try:
        version = tuple(int(p) for p in raw.split(".")[:2])
    except ValueError:
        return GateResult(
            "G5", "environment", GATE_STATUS_FAIL, f"unparseable ultralytics version '{raw}'"
        )
    if not (ULTRALYTICS_MIN <= version < ULTRALYTICS_MAX_EXCLUSIVE):
        low = ".".join(map(str, ULTRALYTICS_MIN))
        high = ".".join(map(str, ULTRALYTICS_MAX_EXCLUSIVE))
        return GateResult(
            "G5",
            "environment",
            GATE_STATUS_FAIL,
            f"ultralytics {raw} outside the supported window >={low},<{high}",
        )
    from src.training.masked_loss import assert_ultralytics_compat

    try:
        assert_ultralytics_compat()
    except RuntimeError as e:
        return GateResult("G5", "environment", GATE_STATUS_FAIL, str(e))
    return GateResult(
        "G5",
        "environment",
        GATE_STATUS_PASS,
        f"ultralytics {raw} in window; loss-surface compat canary passed",
    )


def _gate_mixing_augs(mitigation: MitigationConfig, train_cfg: dict[str, Any]) -> GateResult:
    """G8: mixing augmentations vs the configured policy."""
    effective = effective_mixing_augs(train_cfg)
    active = {k: v for k, v in effective.items() if v > 0}
    if not active:
        return GateResult(
            "G8", "mixing-augmentations", GATE_STATUS_PASS, "mosaic/mixup/copy_paste all 0"
        )
    detail = (
        f"active mixing augmentation(s) {active} composite multiple images per sample, "
        f"but batch['im_file'] only exposes the primary image — per-image masks are "
        f"approximate under them (ADR-P4-04). Set "
        f"{', '.join(f'augmentation.{k}: 0.0' for k in active)} in the training config, "
        f"or relax {MITIGATION_SECTION}.mixing_augmentation_policy."
    )
    if mitigation.mixing_augmentation_policy == "forbid":
        return GateResult("G8", "mixing-augmentations", GATE_STATUS_FAIL, detail)
    if mitigation.mixing_augmentation_policy == "warn":
        return GateResult("G8", "mixing-augmentations", GATE_STATUS_WARN, detail)
    return GateResult(
        "G8",
        "mixing-augmentations",
        GATE_STATUS_PASS,
        f"policy 'ignore': accepting active mixing augmentations {sorted(active)}",
    )


def run_preflight(
    mitigation: MitigationConfig,
    data_yaml_path: Path,
    train_cfg: dict[str, Any],
    processed_root: Path = Path("data/processed"),
) -> PreflightReport:
    """Run all preflight gates for an enabled mitigation config.

    Args:
        mitigation:     The mitigation settings (normally enabled=True; the
                        standalone CLI may run gates for a disabled config).
        data_yaml_path: Live taxonomy YAML (configs/data.yaml).
        train_cfg:      Parsed training config (for G8 augmentation policy).
        processed_root: Dataset root containing images/{train,val}.

    Returns:
        A PreflightReport; callers decide on the verdict (training fails on
        FAIL and proceeds on WARN with the warnings logged).
    """
    results: list[GateResult] = []

    # G6 — config self-validation (cheap, independent).
    try:
        mitigation.validate()
        results.append(
            GateResult(
                "G6",
                "mitigation-config",
                GATE_STATUS_PASS,
                f"section '{MITIGATION_SECTION}' valid "
                f"(artifact: {mitigation.completeness_path.as_posix()})",
            )
        )
    except ValueError as e:
        results.append(GateResult("G6", "mitigation-config", GATE_STATUS_FAIL, str(e)))

    # G1 — artifact loads.
    artifact: dict[str, Any] | None = None
    try:
        artifact = load_completeness(mitigation.completeness_path)
        results.append(
            GateResult(
                "G1",
                "artifact-exists",
                GATE_STATUS_PASS,
                f"{mitigation.completeness_path.as_posix()} loaded "
                f"(schema_version={artifact.get('schema_version')})",
            )
        )
    except (FileNotFoundError, CompletenessError) as e:
        results.append(
            GateResult(
                "G1",
                "artifact-exists",
                GATE_STATUS_FAIL,
                f"{e} — generate it via `dvc repro generate_completeness`.",
            )
        )

    if artifact is None:
        for gate_id, name in (
            ("G2", "taxonomy"),
            ("G3", "coverage"),
            ("G4", "consistency"),
            ("G7", "freshness"),
        ):
            results.append(
                GateResult(
                    gate_id, name, GATE_STATUS_SKIPPED, "not evaluated: artifact unavailable"
                )
            )
    else:
        results.append(_gate_taxonomy(artifact, data_yaml_path))
        results.append(_gate_coverage(artifact, processed_root))
        results.append(_gate_consistency(artifact))
        results.append(_gate_freshness(artifact))

    results.append(_gate_environment())
    results.append(_gate_mixing_augs(mitigation, train_cfg))

    results.sort(key=lambda r: r.gate_id)
    report = PreflightReport(results=tuple(results))
    logger.info(f"Mitigation preflight: {report.verdict}")
    return report


def _gate_taxonomy(artifact: dict[str, Any], data_yaml_path: Path) -> GateResult:
    """G2: artifact taxonomy fingerprint vs live data.yaml."""
    try:
        live_cfg = load_data_config(data_yaml_path)
    except (FileNotFoundError, ValueError) as e:
        return GateResult("G2", "taxonomy", GATE_STATUS_FAIL, str(e))
    live_fp = taxonomy_fingerprint(int(live_cfg["nc"]), get_class_names_from_data_yaml(live_cfg))
    artifact_fp = (artifact.get("taxonomy") or {}).get("fingerprint")
    if artifact_fp != live_fp:
        return GateResult(
            "G2",
            "taxonomy",
            GATE_STATUS_FAIL,
            f"artifact taxonomy {artifact_fp} != live {data_yaml_path.as_posix()} "
            f"({live_fp}) — re-run `dvc repro generate_completeness`.",
        )
    return GateResult(
        "G2", "taxonomy", GATE_STATUS_PASS, f"fingerprint match (nc={live_cfg['nc']})"
    )


def _gate_coverage(artifact: dict[str, Any], processed_root: Path) -> GateResult:
    """G3: every train/val image has a record with the correct split."""
    images: dict[str, Any] = artifact.get("images") or {}
    missing: list[str] = []
    split_mismatch: list[str] = []
    seen: set[str] = set()
    for split in ("train", "val"):
        for img in find_image_files(processed_root / "images" / split):
            seen.add(img.name)
            entry = images.get(img.name)
            if entry is None:
                missing.append(img.name)
            elif entry.get("split") != split:
                split_mismatch.append(f"{img.name} (artifact: {entry.get('split')}, disk: {split})")
    if not seen:
        return GateResult(
            "G3",
            "coverage",
            GATE_STATUS_FAIL,
            f"no images found under {processed_root.as_posix()}/images/{{train,val}} — "
            f"run the split stage first.",
        )
    problems: list[str] = []
    if missing:
        problems.append(f"{len(missing)} image(s) without records (first 10: {missing[:10]})")
    if split_mismatch:
        problems.append(
            f"{len(split_mismatch)} split mismatch(es) (first 10: {split_mismatch[:10]})"
        )
    if problems:
        return GateResult(
            "G3",
            "coverage",
            GATE_STATUS_FAIL,
            "; ".join(problems) + " — re-run `dvc repro generate_completeness`.",
        )
    stale = sorted(
        name
        for name, entry in images.items()
        if entry.get("split") in ("train", "val") and name not in seen
    )
    if stale:
        return GateResult(
            "G3",
            "coverage",
            GATE_STATUS_WARN,
            f"{len(seen)} train/val images covered; {len(stale)} stale record(s) "
            f"reference images no longer on disk (first 10: {stale[:10]})",
        )
    return GateResult("G3", "coverage", GATE_STATUS_PASS, f"{len(seen)} train/val images covered")


def _gate_consistency(artifact: dict[str, Any]) -> GateResult:
    """G4: artifact self-consistency; unused policies are warnings."""
    errors = validate_completeness(artifact)
    if errors:
        preview = " | ".join(errors[:5])
        return GateResult(
            "G4",
            "consistency",
            GATE_STATUS_FAIL,
            f"{len(errors)} validation error(s): {preview}",
        )
    unused = find_unused_policies(artifact)
    if unused:
        return GateResult(
            "G4",
            "consistency",
            GATE_STATUS_WARN,
            f"valid; {len(unused)} unused policy(ies): {unused}",
        )
    return GateResult(
        "G4",
        "consistency",
        GATE_STATUS_PASS,
        f"{len(artifact.get('policies') or {})} policies, "
        f"{len(artifact.get('images') or {})} images consistent",
    )


def _gate_freshness(artifact: dict[str, Any]) -> GateResult:
    """G7: recorded input hashes match the files on disk."""
    inputs = artifact.get("inputs") or {}
    problems: list[str] = []
    for input_name in ("merged_manifest", "split_summary"):
        record = inputs.get(input_name)
        if not isinstance(record, dict) or "path" not in record or "sha256" not in record:
            problems.append(f"{input_name}: no hash recorded in artifact")
            continue
        path = Path(str(record["path"]))
        if not path.exists():
            problems.append(f"{input_name}: {path.as_posix()} missing on disk")
            continue
        if compute_file_hash(path) != record["sha256"]:
            problems.append(f"{input_name}: {path.as_posix()} changed since generation")
    if problems:
        return GateResult(
            "G7",
            "freshness",
            GATE_STATUS_FAIL,
            "; ".join(problems) + " — re-run `dvc repro generate_completeness`.",
        )
    return GateResult("G7", "freshness", GATE_STATUS_PASS, "input hashes match disk")
