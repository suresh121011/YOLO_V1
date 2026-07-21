"""
scripts.dataset.12_auto_annotate — Candidate Generation (DVC: auto_annotate)
============================================================================

Phase-5 L2 entry point: runs every enabled auto-annotation backend over the
untrusted+unverified (image, class) cells of ``data/merged`` and writes one
candidate artifact per backend to ``data/annotation/candidates/<backend>/``.
Candidates NEVER touch labels/ (ADR-P5-01) — they flow to humans through
verification batches (script 13) and come back through the ledger (script 14).

Determinism (ADR-P5-02): sorted image order, fixed seeds,
``torch.use_deterministic_algorithms(True, warn_only=True)``, sha256-pinned
weights, run_id derived from config+commit (never wall clock).
``--verify-determinism`` re-annotates a fixed sample and diffs against the
just-written artifact (non-empty diff exits 1).

Exit codes: 0 = ok, 1 = error (validation problems, pin mismatch, ambiguity).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.annotation import backends as _backends  # noqa: F401 — registry side effect
from src.dataset.annotation.base import AnnotationError, BackendConfig
from src.dataset.annotation.candidates import (
    CANDIDATES_FILENAME,
    ImageCandidates,
    build_candidates_artifact,
    load_candidates,
    save_candidates,
    validate_candidates,
)
from src.dataset.annotation.ledger import LedgerView
from src.dataset.annotation.registry import available_annotators, get_annotator
from src.dataset.annotation.targeting import build_targets, promptable_class_ids
from src.dataset.completeness import taxonomy_fingerprint
from src.dataset.manifest import MERGED_MANIFEST_FILENAME, MergedManifest
from src.utils.config_helpers import get_class_names_from_data_yaml, load_data_config, load_yaml
from src.utils.dataset_utils import compute_file_hash

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

#: Fixed sample size for --verify-determinism re-runs.
DETERMINISM_SAMPLE = 20


def _setup_determinism() -> dict[str, Any]:
    """Fix seeds and deterministic algorithms; return the artifact record.

    TF32 (enabled by default on Ampere+ GPUs) trades precision for speed by
    rounding matmul/conv inputs — it is NOT the FP32 the determinism contract
    (ADR-P5-02) promises, and its reduced-precision accumulation is a
    measured source of run-to-run geometry/confidence drift on this class of
    GPU. CUBLAS_WORKSPACE_CONFIG must be set before any CUDA context exists
    in the process (harmless no-op on a second call within the same run).
    """
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {"seed": 0, "deterministic_algorithms": True, "image_order": "sorted"}


def run_backend(
    backend_name: str,
    backend_cfg: BackendConfig,
    refinement_cfg: dict[str, Any],
    device: str,
    merged_dir: Path,
    manifest: MergedManifest,
    policies: dict[str, str],
    ids_by_name: dict[str, int],
    names_by_id: dict[int, str],
    taxonomy_fp: str,
    verified_cells: dict[str, frozenset[int]],
    ledger_path: Path,
    output_root: Path,
    limit: int | None = None,
) -> Path:
    """Generate, validate, and save one backend's candidate artifact.

    Returns:
        The written candidates.json path.

    Raises:
        AnnotationError: On any validation problem (artifact not written).
    """
    determinism = _setup_determinism()
    annotator = get_annotator(backend_name)
    annotator.load(backend_cfg, device, ids_by_name)

    refiner = None
    if refinement_cfg.get("enabled", False):
        from src.dataset.annotation.refine import RefinementPass

        refiner = RefinementPass(
            Path(str(refinement_cfg.get("weights", ""))),
            str(refinement_cfg.get("weights_sha256", "")),
            device,
        )

    promptable = promptable_class_ids(backend_cfg, ids_by_name)
    targets = build_targets(manifest, policies, promptable, ids_by_name, verified_cells)
    filenames = sorted(targets)
    if limit is not None:
        filenames = filenames[:limit]
    logger.info(
        f"[{backend_name}] {len(filenames)} images targeted "
        f"({len(promptable)} promptable classes)"
    )

    images_root = merged_dir / "images"
    started = time.perf_counter()
    images: dict[str, ImageCandidates] = {}
    for i, filename in enumerate(filenames, 1):
        image_path = images_root / filename
        if not image_path.exists():
            raise AnnotationError(
                f"Targeted image missing on disk: {image_path} — merged manifest and "
                f"data/merged/images disagree; re-run the merge stage."
            )
        detections = annotator.annotate(image_path, targets[filename])
        # Backends return everything above the low conf_floor (AutoAnnotator's
        # contract); per-class thresholds decide candidate status HERE, once,
        # for every backend uniformly — R30's verification-flooding mitigation
        # only holds if this actually runs (configs/annotation.yaml's own
        # comment: "conf_floor: record everything above this; per-class
        # thresholds below decide candidate status downstream").
        detections = [
            det
            for det in detections
            if det.conf >= backend_cfg.threshold_for(names_by_id.get(det.class_id, ""))
        ]
        if refiner is not None and detections:
            detections = refiner.refine(image_path, detections)
        images[filename] = ImageCandidates(
            targeted_class_ids=targets[filename], detections=tuple(detections)
        )
        if i % 200 == 0 or i == len(filenames):
            rate = i / max(time.perf_counter() - started, 1e-9)
            logger.info(f"[{backend_name}] {i}/{len(filenames)} images ({rate:.1f} img/s)")

    artifact = build_candidates_artifact(
        backend=backend_name,
        model=annotator.fingerprint(),
        taxonomy_fp=taxonomy_fp,
        inputs={
            "images_root": str(images_root),
            "merged_manifest_sha256": compute_file_hash(merged_dir / MERGED_MANIFEST_FILENAME),
            "ledger_sha256": (compute_file_hash(ledger_path) if ledger_path.exists() else "absent"),
        },
        determinism=determinism,
        images=images,
        runtime_s=time.perf_counter() - started,
        class_names_by_id=names_by_id,
    )
    problems = validate_candidates(artifact, nc=len(names_by_id), expected_taxonomy_fp=taxonomy_fp)
    if problems:
        for problem in problems[:20]:
            logger.error(f"[{backend_name}] artifact problem: {problem}")
        raise AnnotationError(
            f"[{backend_name}] candidate artifact failed validation with "
            f"{len(problems)} problem(s) — nothing written."
        )
    out_path = output_root / backend_name / CANDIDATES_FILENAME
    save_candidates(artifact, out_path)
    return out_path


def verify_determinism(
    artifact_path: Path,
    backend_name: str,
    backend_cfg: BackendConfig,
    refinement_cfg: dict[str, Any],
    device: str,
    merged_dir: Path,
    ids_by_name: dict[str, int],
) -> list[str]:
    """Re-annotate a fixed sample and diff against the artifact.

    Must replay the exact same pipeline that produced the artifact — including
    the optional SAM refinement pass — or every refined image spuriously
    "mismatches" (same detection count, tightened geometry) even when the
    backend itself is perfectly deterministic.

    Returns:
        Mismatch descriptions (empty = deterministic on this machine).
    """
    artifact = load_candidates(artifact_path)
    _setup_determinism()
    annotator = get_annotator(backend_name)
    annotator.load(backend_cfg, device, ids_by_name)

    refiner = None
    if refinement_cfg.get("enabled", False):
        from src.dataset.annotation.refine import RefinementPass

        refiner = RefinementPass(
            Path(str(refinement_cfg.get("weights", ""))),
            str(refinement_cfg.get("weights_sha256", "")),
            device,
        )

    mismatches: list[str] = []
    sample = sorted(artifact["images"])[:DETERMINISM_SAMPLE]
    for filename in sample:
        entry = artifact["images"][filename]
        image_path = merged_dir / "images" / filename
        detections = annotator.annotate(image_path, tuple(entry["targeted_class_ids"]))
        if refiner is not None and detections:
            detections = refiner.refine(image_path, detections)
        fresh = [d.to_dict() for d in detections]
        recorded = list(entry["detections"])
        if fresh != recorded:
            mismatches.append(
                f"{filename}: fresh run produced {len(fresh)} detections vs "
                f"{len(recorded)} recorded (or differing geometry/confidence)"
            )
    return mismatches


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate auto-annotation candidates (Phase-5 L2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path("configs/annotation.yaml"))
    parser.add_argument("--sources-config", type=Path, default=Path("configs/dataset_sources.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--merged-dir", type=Path, default=Path("data/merged"))
    parser.add_argument("--output", type=Path, default=Path("data/annotation/candidates"))
    parser.add_argument(
        "--backend",
        action="append",
        default=None,
        help="Backend(s) to run (default: every enabled backend in the config).",
    )
    parser.add_argument("--device", type=str, default=None, help="Override configured device.")
    parser.add_argument("--limit", type=int, default=None, help="Cap targeted images (smoke).")
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help=f"After writing, re-annotate the first {DETERMINISM_SAMPLE} images and diff.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns 0 (ok) or 1 (error)."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parse_args()

    annotation_cfg = load_yaml(args.config).get("auto_annotation") or {}
    verification_cfg = load_yaml(args.config).get("verification") or {}
    device = args.device or str(annotation_cfg.get("device", "cpu"))
    backends_cfg: dict[str, Any] = annotation_cfg.get("backends") or {}
    refinement_cfg: dict[str, Any] = annotation_cfg.get("refinement") or {}

    data_cfg = load_data_config(args.data_config)
    names_by_id = get_class_names_from_data_yaml(data_cfg)
    ids_by_name = {name: cid for cid, name in names_by_id.items()}
    taxonomy_fp = taxonomy_fingerprint(len(names_by_id), names_by_id)

    manifest = MergedManifest.load(args.merged_dir / MERGED_MANIFEST_FILENAME)
    policies = {
        str(k): str(v)
        for k, v in (load_yaml(args.sources_config).get("completeness", {}) or {})
        .get("policies", {})
        .items()
    }
    ledger_path = Path(
        str(verification_cfg.get("ledger_path", "data/annotation/verification_ledger.json"))
    )
    verified_cells = LedgerView.load(ledger_path).verified_cells(ids_by_name)

    selected = args.backend or [
        name for name, cfg in backends_cfg.items() if (cfg or {}).get("enabled", False)
    ]
    if not selected:
        logger.error(
            f"No backends selected/enabled. Configured: {sorted(backends_cfg)}; "
            f"registered: {available_annotators()}"
        )
        return 1

    try:
        for backend_name in selected:
            raw_cfg = backends_cfg.get(backend_name)
            if raw_cfg is None:
                raise AnnotationError(
                    f"Backend '{backend_name}' has no section under "
                    f"auto_annotation.backends in {args.config}"
                )
            backend_cfg = BackendConfig.from_annotation_config(backend_name, raw_cfg)
            config_problems = backend_cfg.validate(ids_by_name)
            if config_problems:
                raise AnnotationError(f"Backend '{backend_name}' config invalid: {config_problems}")
            artifact_path = run_backend(
                backend_name=backend_name,
                backend_cfg=backend_cfg,
                refinement_cfg=refinement_cfg,
                device=device,
                merged_dir=args.merged_dir,
                manifest=manifest,
                policies=policies,
                ids_by_name=ids_by_name,
                names_by_id=names_by_id,
                taxonomy_fp=taxonomy_fp,
                verified_cells=verified_cells,
                ledger_path=ledger_path,
                output_root=args.output,
                limit=args.limit,
            )
            if args.verify_determinism:
                mismatches = verify_determinism(
                    artifact_path,
                    backend_name,
                    backend_cfg,
                    refinement_cfg,
                    device,
                    args.merged_dir,
                    ids_by_name,
                )
                if mismatches:
                    for m in mismatches:
                        logger.error(f"[{backend_name}] determinism mismatch: {m}")
                    return 1
                logger.info(f"[{backend_name}] determinism verified on {DETERMINISM_SAMPLE} images")
    except AnnotationError as exc:
        logger.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
