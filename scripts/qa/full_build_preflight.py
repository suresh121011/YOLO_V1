"""
scripts.qa.full_build_preflight — Full-Mode Build Preflight Gates (FB1–FB6)
===========================================================================

Fail-early environment validation before the Phase-5 ``mode: full`` dataset
build (expected volume ~15–30k images / 10–40 GB raw). Run it BEFORE flipping
``mode`` in configs/dataset_sources.yaml and launching ``dvc repro``.

Gates:
    FB1  free disk space on the data drive (hard gate, default ≥ 150 GB)
    FB2  DVC default remote configured, reachable, and off the OneDrive tree
    FB3  Roboflow readiness (slugs populated + API key env set) — blocks the
         4 Roboflow-covered classes when unmet (medicine_bottle, charger,
         wire, gas_cylinder)
    FB4  GPU visibility (needed by the auto_annotate stage, not downloads)
    FB5  OneDrive sync hazard for the repo data tree / DVC cache (risk R34)
    FB6  acquisition mode is ``full`` (informational until the M7 flip)

Reuses the Phase-4 gate vocabulary (GateResult / PreflightReport from
src/training/preflight.py — torch-free by design). Report triplet is written
to data/qa_reports/full_build_preflight.{json,csv,md}.

Exit codes: 0 = pass, 1 = failures, 2 = warnings only (mirrors run_full_qa).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.preflight import (
    GATE_STATUS_FAIL,
    GATE_STATUS_PASS,
    GATE_STATUS_WARN,
    GateResult,
    PreflightReport,
)
from src.utils.config_helpers import load_yaml
from src.utils.report_utils import timestamp_str, write_all_formats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

#: Minimum free space for the full build: raw downloads + interim + merged +
#: processed + DVC cache copies of all of them, with headroom.
DEFAULT_MIN_FREE_GB = 150.0

#: Classes whose only public coverage comes through Roboflow slugs.
ROBOFLOW_BLOCKED_CLASSES = ("medicine_bottle", "charger", "wire", "gas_cylinder")

_BYTES_PER_GB = 1024**3


def _free_gb(path: Path) -> float:
    """Free space in GB on the drive hosting ``path`` (first existing parent)."""
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:  # filesystem root that does not exist
            break
        probe = parent
    return shutil.disk_usage(probe).free / _BYTES_PER_GB


def _is_onedrive_path(path: Path) -> bool:
    """True when a path lives inside a OneDrive-synced tree.

    Matches path COMPONENTS named ``OneDrive`` / ``OneDrive - <org>`` rather
    than a raw substring, so unrelated names that merely contain the word
    (e.g. pytest tmp dirs named after a test) never false-positive.
    """
    for part in path.parts:
        lowered = part.lower()
        if lowered == "onedrive" or lowered.startswith(("onedrive ", "onedrive-")):
            return True
    return False


def _parse_dvc_config(text: str) -> dict[str, dict[str, str]]:
    """Minimal INI parse of a DVC config (indented keys, quoted sections).

    stdlib configparser cannot read DVC's format — the indented ``key = value``
    lines parse as continuation lines — so this walks the file manually.
    """
    sections: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip()
            # Unquote ONE symmetric layer (['remote "x"'] → remote "x") —
            # a blanket strip("'\"") would eat the inner closing quote too.
            if len(name) >= 2 and name[0] == name[-1] and name[0] in "'\"":
                name = name[1:-1]
            current = sections.setdefault(name, {})
            continue
        if current is not None and "=" in line:
            key, _, value = line.partition("=")
            current[key.strip()] = value.split("#", 1)[0].strip()
    return sections


def read_dvc_remote(dvc_config_path: Path) -> tuple[str, str] | None:
    """Return (remote_name, url) for the default DVC remote, or None.

    Reads .dvc/config directly so the QA suite needs no dvc import.
    """
    if not dvc_config_path.exists():
        return None
    try:
        sections = _parse_dvc_config(dvc_config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        logger.warning(f"Cannot read {dvc_config_path}: {exc}")
        return None

    default_name = sections.get("core", {}).get("remote")
    if not default_name:
        return None
    url = sections.get(f'remote "{default_name}"', {}).get("url")
    if not url:
        return None
    return default_name, url


def gate_disk_space(data_root: Path, min_free_gb: float) -> GateResult:
    """FB1 — free space on the drive hosting the data tree."""
    free = _free_gb(data_root)
    if free < min_free_gb:
        return GateResult(
            "FB1",
            "disk space",
            GATE_STATUS_FAIL,
            f"{free:.1f} GB free on the drive hosting {data_root} — the full build "
            f"needs ≥ {min_free_gb:.0f} GB. Free space or use --min-free-gb to "
            f"override with a recorded decision.",
        )
    return GateResult(
        "FB1", "disk space", GATE_STATUS_PASS, f"{free:.1f} GB free (≥ {min_free_gb:.0f} GB)"
    )


def gate_dvc_remote(dvc_config_path: Path, min_free_gb: float) -> GateResult:
    """FB2 — default DVC remote configured, reachable, off OneDrive."""
    remote = read_dvc_remote(dvc_config_path)
    if remote is None:
        return GateResult(
            "FB2",
            "dvc remote",
            GATE_STATUS_FAIL,
            f"No default DVC remote in {dvc_config_path} — the dataset would be "
            f"single-copy (audit risk C-1). Configure one: "
            f"dvc remote add -d localstore <path-off-OneDrive>",
        )
    name, url = remote
    if "://" in url:  # non-local remote (s3://, gdrive://, …)
        return GateResult(
            "FB2",
            "dvc remote",
            GATE_STATUS_WARN,
            f"Default remote '{name}' is non-local ({url}) — reachability is not "
            f"checked here; verify credentials and run `dvc push && dvc status -c` "
            f"manually before the build.",
        )
    remote_path = Path(url)
    if _is_onedrive_path(remote_path):
        return GateResult(
            "FB2",
            "dvc remote",
            GATE_STATUS_FAIL,
            f"Default remote '{name}' ({url}) is INSIDE a OneDrive-synced tree — "
            f"sync races corrupt DVC object stores (risk R34). Move it outside "
            f"OneDrive and update .dvc/config.",
        )
    if not remote_path.exists():
        return GateResult(
            "FB2",
            "dvc remote",
            GATE_STATUS_FAIL,
            f"Default remote '{name}' path {url} does not exist (drive unplugged?). "
            f"Mount/create it, then re-run.",
        )
    free = _free_gb(remote_path)
    if free < min_free_gb:
        return GateResult(
            "FB2",
            "dvc remote",
            GATE_STATUS_WARN,
            f"Remote '{name}' at {url} has {free:.1f} GB free — the full build "
            f"pushes may need up to {min_free_gb:.0f} GB.",
        )
    return GateResult(
        "FB2", "dvc remote", GATE_STATUS_PASS, f"'{name}' at {url} ({free:.1f} GB free)"
    )


def gate_roboflow(sources_yaml: Path, env: dict[str, str]) -> GateResult:
    """FB3 — Roboflow slugs populated and API key available."""
    config = load_yaml(sources_yaml)
    roboflow = config.get("sources", {}).get("roboflow", {})
    if not roboflow.get("enabled", False):
        return GateResult(
            "FB3",
            "roboflow readiness",
            GATE_STATUS_WARN,
            f"Roboflow source disabled — no public coverage for "
            f"{', '.join(ROBOFLOW_BLOCKED_CLASSES)}.",
        )
    datasets = roboflow.get("datasets") or []
    key_env = str(roboflow.get("api_key_env", "ROBOFLOW_API_KEY"))
    have_key = bool(env.get(key_env))
    if not datasets:
        return GateResult(
            "FB3",
            "roboflow readiness",
            GATE_STATUS_WARN,
            f"sources.roboflow.datasets is empty — human track H-B (slug selection "
            f"+ per-slug license review BEFORE download) has not landed; public "
            f"coverage for {', '.join(ROBOFLOW_BLOCKED_CLASSES)} stays blocked.",
        )
    if not have_key:
        return GateResult(
            "FB3",
            "roboflow readiness",
            GATE_STATUS_FAIL,
            f"{len(datasets)} Roboflow dataset(s) configured but {key_env} is not "
            f"set — the download stage would silently skip them. Export the key "
            f"and re-run.",
        )
    return GateResult(
        "FB3",
        "roboflow readiness",
        GATE_STATUS_PASS,
        f"{len(datasets)} dataset(s) configured, {key_env} set",
    )


def detect_gpu() -> tuple[bool | None, str]:
    """Detect an NVIDIA GPU via torch, falling back to nvidia-smi.

    Returns:
        (visible, details) — visible is None when undeterminable.
    """
    try:
        import torch  # local import: heavy, optional in QA environments

        if torch.cuda.is_available():
            return True, f"CUDA device: {torch.cuda.get_device_name(0)}"
        return False, "torch is installed but reports no CUDA device"
    except ImportError:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return True, f"nvidia-smi: {result.stdout.strip().splitlines()[0]}"
        return False, "nvidia-smi found no GPU"
    except (OSError, subprocess.TimeoutExpired):
        return None, "torch not installed and nvidia-smi unavailable"


def gate_gpu() -> GateResult:
    """FB4 — GPU visibility (auto_annotate needs it; downloads do not)."""
    visible, details = detect_gpu()
    if visible:
        return GateResult("FB4", "gpu", GATE_STATUS_PASS, details)
    return GateResult(
        "FB4",
        "gpu",
        GATE_STATUS_WARN,
        f"{details} — the download/merge/split stages run fine, but the "
        f"auto_annotate stage needs the local NVIDIA GPU (user decision, "
        f"Phase-5 plan).",
    )


def gate_onedrive(repo_root: Path) -> GateResult:
    """FB5 — OneDrive sync hazard for the repo data tree / DVC cache (R34)."""
    if not _is_onedrive_path(repo_root):
        return GateResult(
            "FB5", "onedrive hazard", GATE_STATUS_PASS, f"{repo_root} is outside OneDrive"
        )
    return GateResult(
        "FB5",
        "onedrive hazard",
        GATE_STATUS_WARN,
        f"Repo (and .dvc/cache) live under OneDrive ({repo_root}) — sync can race "
        f"large builds (risk R34). Before the full build: pause OneDrive sync or "
        f"relocate the cache off the synced tree "
        f"(`dvc cache dir <path>` + `dvc config cache.type hardlink,copy`).",
    )


def gate_mode(sources_yaml: Path) -> GateResult:
    """FB6 — acquisition mode (informational until the M7 flip)."""
    mode = str(load_yaml(sources_yaml).get("mode", "smoke"))
    if mode == "full":
        return GateResult("FB6", "acquisition mode", GATE_STATUS_PASS, "mode: full")
    return GateResult(
        "FB6",
        "acquisition mode",
        GATE_STATUS_WARN,
        f"mode: {mode} — this preflight targets the full build; flip "
        f"configs/dataset_sources.yaml mode to 'full' at M7 (bundled with the "
        f"WIDER class_caps + Roboflow slug changes).",
    )


def run_full_build_preflight(
    repo_root: Path,
    sources_yaml: Path,
    dvc_config_path: Path,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    env: dict[str, str] | None = None,
) -> PreflightReport:
    """Run all FB gates and aggregate the report."""
    env_map = dict(os.environ) if env is None else env
    results = (
        gate_disk_space(repo_root / "data", min_free_gb),
        gate_dvc_remote(dvc_config_path, min_free_gb),
        gate_roboflow(sources_yaml, env_map),
        gate_gpu(),
        gate_onedrive(repo_root),
        gate_mode(sources_yaml),
    )
    return PreflightReport(results=results)


def write_report(report: PreflightReport, output_dir: Path) -> dict[str, Path]:
    """Write the FB report triplet (JSON + CSV + Markdown)."""
    data: dict[str, Any] = {"generated_at": timestamp_str(), **report.to_dict()}
    csv_rows = [
        {
            "gate_id": g["gate_id"],
            "name": g["name"],
            "status": g["status"],
            "details": g["details"],
        }
        for g in data["gates"]
    ]
    sections = [
        {
            "heading": "Gates",
            "content": "",
            "table": {
                "headers": ["Gate", "Name", "Status", "Details"],
                "rows": [
                    [g["gate_id"], g["name"], g["status"], g["details"]] for g in data["gates"]
                ],
            },
        }
    ]
    return write_all_formats(
        report_data=data,
        csv_rows=csv_rows,
        md_title="Full-Mode Build Preflight (FB1–FB6)",
        md_sections=sections,
        output_dir=output_dir,
        base_name="full_build_preflight",
        csv_fieldnames=["gate_id", "name", "status", "details"],
        md_metadata={"verdict": data["verdict"], "generated_at": data["generated_at"]},
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Preflight gates for the Phase-5 full-mode dataset build.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--sources-config", type=Path, default=Path("configs/dataset_sources.yaml"))
    parser.add_argument("--dvc-config", type=Path, default=Path(".dvc/config"))
    parser.add_argument("--output", type=Path, default=Path("data/qa_reports"))
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_GB,
        help="Free-space floor for FB1/FB2 (override only with a recorded decision).",
    )
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns 0 (pass), 1 (failures), or 2 (warnings)."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    args = parse_args()
    logger.info("=" * 60)
    logger.info("Full-Mode Build Preflight — Elderly Assistant System")
    logger.info("=" * 60)

    report = run_full_build_preflight(
        repo_root=args.repo_root.resolve(),
        sources_yaml=args.sources_config,
        dvc_config_path=args.dvc_config,
        min_free_gb=args.min_free_gb,
    )
    for result in report.results:
        logger.info(result.format_line())

    paths = write_report(report, args.output)
    logger.info(f"Report written: {paths['json']}")

    logger.info("=" * 60)
    if report.verdict == "FAIL":
        logger.error("PREFLIGHT VERDICT: FAIL — do not start the full build")
        return 1
    if report.verdict == "WARN":
        if args.strict:
            logger.warning("PREFLIGHT VERDICT: warnings present (strict mode → exit 1)")
            return 1
        logger.warning("PREFLIGHT VERDICT: warnings present (exit 2)")
        return 2
    logger.info("PREFLIGHT VERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
