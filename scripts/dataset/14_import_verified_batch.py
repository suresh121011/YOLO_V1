"""
scripts.dataset.14_import_verified_batch — CVAT Export Import (DVC:
import_verified_annotations)
==============================================================================

Phase-5 M2 entry point: FROZEN DVC stage (no deps, like
``build_verification_batches`` — see dvc.yaml header note). Humans run this
directly after exporting a batch's CVAT task as "YOLO 1.1", then
``dvc commit -f import_verified_annotations``.

Reuses ``read_yolo_export`` + ``verify_class_order`` (Phase-3's importer) and
``src.dataset.annotation.verified_import.import_verified_batch`` for the
non-target byte-equality check, delta extraction, and ledger verdict
recording (ADR-P5-03). On success the batch transitions to ``imported``; on
any validation failure it stays at its current (pre-import) status so it can
be corrected in CVAT and re-exported — nothing partial is ever committed
(the whole batch's checks run before any verdict is recorded).

Exit codes: 0 = ok, 1 = error.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.annotation.base import AnnotationError
from src.dataset.annotation.batches import BATCH_MANIFEST_FILENAME, VerificationBatchManifest
from src.dataset.annotation.ledger import load_ledger, recompute_stats, save_ledger, validate_ledger
from src.dataset.annotation.verified_import import import_verified_batch
from src.dataset.capture.annotations import read_yolo_export
from src.dataset.completeness import taxonomy_fingerprint
from src.dataset.manifest import MERGED_MANIFEST_FILENAME, MergedManifest
from src.utils.config_helpers import get_class_names_from_data_yaml, load_data_config, load_yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

#: Statuses eligible for import (anything already imported must not be redone
#: through this path — a re-verification uses --supersedes on a NEW batch).
IMPORTABLE_STATUSES = ("created", "exported", "staged")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Import one verification batch's CVAT export into the ledger (Phase-5 M2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch", required=True, help="Batch id, e.g. vb001_yolo_world.")
    parser.add_argument(
        "--export", type=Path, required=True, help="CVAT YOLO 1.1 export (zip or dir)."
    )
    parser.add_argument("--verifier", required=True, help="Pseudonymous reviewer handle.")
    parser.add_argument(
        "--supersedes",
        default=None,
        help="Prior batch_id being intentionally overridden, if this import conflicts "
        "with existing ledger verdicts.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/annotation.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--merged-dir", type=Path, default=Path("data/merged"))
    parser.add_argument("--batches-root", type=Path, default=Path("data/annotation/batches"))
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns 0 (ok) or 1 (error)."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parse_args()

    verification_cfg = load_yaml(args.config).get("verification") or {}
    ledger_path = Path(
        str(verification_cfg.get("ledger_path", "data/annotation/verification_ledger.json"))
    )
    verified_labels_dir = Path(
        str(verification_cfg.get("verified_labels_dir", "data/annotation/verified_labels"))
    )

    data_cfg = load_data_config(args.data_config)
    class_names_by_id = get_class_names_from_data_yaml(data_cfg)
    ids_by_name = {name: cid for cid, name in class_names_by_id.items()}
    taxonomy_fp = taxonomy_fingerprint(len(class_names_by_id), class_names_by_id)

    batch_dir = args.batches_root / args.batch
    manifest_path = batch_dir / BATCH_MANIFEST_FILENAME
    if not manifest_path.exists():
        logger.error(f"No batch manifest at {manifest_path}")
        return 1
    batch = VerificationBatchManifest.load(manifest_path)
    if batch.status not in IMPORTABLE_STATUSES:
        logger.error(
            f"Batch '{batch.batch_id}' has status '{batch.status}' — importable statuses "
            f"are {IMPORTABLE_STATUSES}. A re-verification uses --supersedes on a NEW batch."
        )
        return 1

    try:
        export = read_yolo_export(args.export)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1

    manifest = MergedManifest.load(args.merged_dir / MERGED_MANIFEST_FILENAME)
    ledger = load_ledger(ledger_path)

    try:
        result = import_verified_batch(
            batch=batch,
            export=export,
            class_names_by_id=class_names_by_id,
            ids_by_name=ids_by_name,
            merged_labels_dir=args.merged_dir / "labels",
            verified_labels_dir=verified_labels_dir,
            ledger=ledger,
            source_by_image=manifest.image_provenance,
            verifier=args.verifier,
            supersedes=args.supersedes,
        )
    except AnnotationError as exc:
        logger.error(f"Batch '{batch.batch_id}' import failed — staying at '{batch.status}': {exc}")
        return 1

    problems = validate_ledger(ledger, class_names=ids_by_name, expected_taxonomy_fp=taxonomy_fp)
    if problems:
        for problem in problems[:20]:
            logger.error(f"Ledger validation problem: {problem}")
        logger.error(f"Ledger failed validation with {len(problems)} problem(s) — nothing saved.")
        return 1

    recompute_stats(ledger, taxonomy_fp)
    save_ledger(ledger, ledger_path)

    batch.status = "imported"
    batch.save(manifest_path)

    for note in result.problems:
        logger.warning(note)
    logger.info(
        f"Batch '{batch.batch_id}' imported: {result.images_imported} image(s), "
        f"{result.verdicts_recorded} verdict(s), {result.delta_files_written} delta file(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
