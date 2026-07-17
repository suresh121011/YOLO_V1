"""
scripts.dataset.13_build_verification_batches — CVAT Batch Packaging
=====================================================================

Phase-5 M2 entry point: FROZEN DVC stage (declares no deps — mirrors the
Phase-3 frozen-ingest pattern; see dvc.yaml header note and plan
§"DAG acyclicity"). Humans run this directly, then ``dvc commit -f
build_verification_batches`` to record the batch directories DVC tracks.

Ranks each backend's untouched candidate images by expected verification
gain (batches.py), chunks them into ``verification.batch_size`` groups, and
writes one ``data/annotation/batches/vbNNN_<backend>/`` per batch:
``batch_manifest.json`` (status ``created``) + ``preannotations.zip`` (YOLO
1.1, obj.names = full taxonomy order — ADR-P5-03). A shared
``cvat_labels.json`` label-constructor spec is (re)written at the batches
root so a human can always paste the exact taxonomy order into a fresh CVAT
task, regardless of which batch they are creating it for.

Exit codes: 0 = ok (including "nothing to batch"), 1 = error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.annotation.base import AnnotationError
from src.dataset.annotation.batches import (
    BATCH_MANIFEST_FILENAME,
    CVAT_LABELS_FILENAME,
    build_batch_manifests,
)
from src.dataset.annotation.candidates import CANDIDATES_FILENAME, load_candidates
from src.dataset.annotation.cvat_package import build_cvat_labels_spec, build_preannotation_zip
from src.utils.config_helpers import get_class_names_from_data_yaml, load_data_config, load_yaml
from src.utils.dataset_utils import compute_file_hash

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

PREANNOTATIONS_FILENAME = "preannotations.zip"


def build_for_backend(
    backend: str,
    candidates_dir: Path,
    batches_root: Path,
    merged_labels_dir: Path,
    class_names_by_id: dict[int, str],
    priority_classes: frozenset[str],
    batch_size: int,
) -> int:
    """Build every new batch for one backend's candidate artifact.

    Returns:
        Number of batches written.

    Raises:
        AnnotationError: On a missing/unreadable candidates artifact.
    """
    candidates_path = candidates_dir / CANDIDATES_FILENAME
    if not candidates_path.exists():
        raise AnnotationError(
            f"No candidates artifact for backend '{backend}' at {candidates_path} — "
            f"run 12_auto_annotate.py first."
        )
    candidates = load_candidates(candidates_path)
    candidates_sha256 = compute_file_hash(candidates_path)

    manifests = build_batch_manifests(
        candidates=candidates,
        backend=backend,
        candidates_sha256=candidates_sha256,
        batches_root=batches_root,
        class_names_by_id=class_names_by_id,
        priority_classes=priority_classes,
        batch_size=batch_size,
    )
    if not manifests:
        logger.info(f"[{backend}] nothing to batch (no unclaimed images with candidates)")
        return 0

    for manifest in manifests:
        batch_dir = batches_root / manifest.batch_id
        zip_path = batch_dir / PREANNOTATIONS_FILENAME
        manifest.preannotations_sha256 = build_preannotation_zip(
            batch_images=manifest.images,
            candidate_images=candidates["images"],
            merged_labels_dir=merged_labels_dir,
            class_names_by_id=class_names_by_id,
            out_zip=zip_path,
        )
        manifest.save(batch_dir / BATCH_MANIFEST_FILENAME)
        logger.info(
            f"[{backend}] {manifest.batch_id}: {len(manifest.images)} images, "
            f"target_classes={manifest.target_classes}, expected_gain={manifest.expected_gain}"
        )
    return len(manifests)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build CVAT verification batches from auto-annotation candidates (Phase-5 M2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path("configs/annotation.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--candidates-root", type=Path, default=Path("data/annotation/candidates"))
    parser.add_argument("--merged-labels-dir", type=Path, default=Path("data/merged/labels"))
    parser.add_argument(
        "--backend",
        action="append",
        default=None,
        help="Backend(s) to batch (default: every backend with a candidates artifact).",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns 0 (ok) or 1 (error)."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parse_args()

    annotation_cfg = load_yaml(args.config).get("auto_annotation") or {}
    verification_cfg: dict[str, Any] = load_yaml(args.config).get("verification") or {}
    batches_root = Path(str(verification_cfg.get("batches_root", "data/annotation/batches")))
    batch_size = int(verification_cfg.get("batch_size", 200))
    priority_classes = frozenset(
        (annotation_cfg.get("targeting") or {}).get("priority_classes") or []
    )

    data_cfg = load_data_config(args.data_config)
    class_names_by_id = get_class_names_from_data_yaml(data_cfg)

    backends = args.backend or sorted(
        p.parent.name for p in args.candidates_root.glob(f"*/{CANDIDATES_FILENAME}")
    )
    if not backends:
        logger.error(f"No candidate artifacts found under {args.candidates_root}")
        return 1

    batches_root.mkdir(parents=True, exist_ok=True)
    (batches_root / CVAT_LABELS_FILENAME).write_text(
        json.dumps(build_cvat_labels_spec(class_names_by_id), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    total = 0
    try:
        for backend in backends:
            total += build_for_backend(
                backend=backend,
                candidates_dir=args.candidates_root / backend,
                batches_root=batches_root,
                merged_labels_dir=args.merged_labels_dir,
                class_names_by_id=class_names_by_id,
                priority_classes=priority_classes,
                batch_size=batch_size,
            )
    except AnnotationError as exc:
        logger.error(str(exc))
        return 1
    logger.info(f"Wrote {total} verification batch(es) under {batches_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
