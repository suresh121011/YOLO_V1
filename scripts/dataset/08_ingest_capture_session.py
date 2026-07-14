"""
scripts.dataset.08_ingest_capture_session — Capture Session Ingest CLI
=======================================================================

Validates and ingests one custom capture session from a staging inbox
into ``data/raw/custom_captures`` (or the locked eval set root) via
:mod:`src.dataset.capture.ingest`: consent verification, image intake
gates, EXIF/GPS privacy stripping, session manifests and the aggregate
source manifest.

Usage:
    # Ingest a session (house/room are parsed from the session id)
    python scripts/dataset/08_ingest_capture_session.py \\
        --session-id h01_kitchen_s001 --lighting daylight \\
        --device "Pixel 7" --date 2026-07-20 \\
        --consent-ref CONSENT-h01-2026-001 --classes gas_cylinder,stove

    # Ingest into the eval set instead
    python scripts/dataset/08_ingest_capture_session.py --dataset eval ...

    # Bootstrap / maintenance
    python scripts/dataset/08_ingest_capture_session.py --init
    python scripts/dataset/08_ingest_capture_session.py --verify-all
    python scripts/dataset/08_ingest_capture_session.py --lock-eval

Exit codes: 0 = success, 1 = failure (consent/lock/structural or, with
--strict, any rejection), 2 = partial (some images rejected).

DVC integration:
    ``--verify-all`` is the cmd of the frozen ``ingest_custom_captures`` /
    ``ingest_eval_set`` stages (never auto-run; humans ingest, then
    ``dvc commit -f <stage>``). See docs/04 capture_annotation_runbook.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.capture.config import load_capture_config, parse_session_id
from src.dataset.capture.consent import load_consent_registry, verify_consent
from src.dataset.capture.ingest import (
    DEFAULT_CAPTURE_LICENSE,
    SessionMeta,
    ingest_session,
    init_captures_tree,
    lock_eval_set,
    verify_captures_tree,
)
from src.dataset.sources_config import DEFAULT_SOURCES_CONFIG_PATH, load_sources_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest a custom capture session with privacy stripping and manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to capture_config.yaml (default: configs/capture_config.yaml).",
    )
    parser.add_argument(
        "--sources-config",
        type=Path,
        default=DEFAULT_SOURCES_CONFIG_PATH,
        help="Path to dataset_sources.yaml (license string for manifests).",
    )
    parser.add_argument(
        "--dataset",
        choices=("captures", "eval"),
        default="captures",
        help="Ingest target: training captures or the locked real-home eval set.",
    )
    parser.add_argument("--session-id", help="Session ID, e.g. h01_kitchen_s001.")
    parser.add_argument(
        "--inbox",
        type=Path,
        default=None,
        help="Staging inbox override (use a local, non-cloud-synced folder).",
    )
    parser.add_argument("--lighting", help="Lighting condition (see capture config).")
    parser.add_argument("--device", default="", help="Camera/phone model used.")
    parser.add_argument("--date", default="", help="Capture date (ISO-8601, e.g. 2026-07-20).")
    parser.add_argument(
        "--consent-ref", default="", help="Consent record ID, e.g. CONSENT-h01-2026-001."
    )
    parser.add_argument(
        "--classes",
        default="",
        help="Comma-separated classes this session captures EXHAUSTIVELY "
        "(session trusted_classes).",
    )
    parser.add_argument("--notes", default="", help="Free-text session notes (no PII).")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and report without writing."
    )
    parser.add_argument("--strict", action="store_true", help="Exit 1 when any image is rejected.")
    parser.add_argument(
        "--init", action="store_true", help="Create the empty capture tree and exit."
    )
    parser.add_argument(
        "--verify-all",
        action="store_true",
        help="Re-validate the whole capture tree (read-only) and exit.",
    )
    parser.add_argument(
        "--lock-eval", action="store_true", help="Freeze the eval set (writes LOCKED.json)."
    )
    return parser.parse_args()


def _capture_license(sources_config_path: Path) -> str:
    """License string for manifests, from dataset_sources.yaml when available."""
    try:
        sources = load_sources_config(sources_config_path)
    except (FileNotFoundError, ValueError):
        return DEFAULT_CAPTURE_LICENSE
    custom = sources.sources.get("custom_captures")
    return custom.license if custom is not None and custom.license else DEFAULT_CAPTURE_LICENSE


def main() -> int:
    """Entry point. Returns 0 on success, 1 on failure, 2 on partial ingest."""
    args = parse_args()

    try:
        config = load_capture_config(args.config).with_overrides(inbox_dir=args.inbox)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1
    dest_root = config.eval_root if args.dataset == "eval" else config.captures_root

    if args.init:
        init_captures_tree(dest_root)
        return 0

    if args.verify_all:
        problems = verify_captures_tree(dest_root, config)
        for problem in problems:
            logger.error(f"verify: {problem}")
        return 1 if problems else 0

    if args.lock_eval:
        try:
            lock_eval_set(config.eval_root)
        except ValueError as e:
            logger.error(str(e))
            return 1
        return 0

    # ── Ingest mode ──────────────────────────────────────────────────────────
    if not args.session_id or not args.lighting:
        logger.error("--session-id and --lighting are required to ingest")
        return 1

    problems = config.validate_session_id(args.session_id)
    if args.lighting not in config.lighting:
        problems.append(
            f"lighting '{args.lighting}' not in configured values {list(config.lighting)}"
        )
    house_id, room = ("", "")
    if not problems:
        house_id, room = parse_session_id(args.session_id)
        registry = load_consent_registry(config.consent.registry_path)
        problems.extend(verify_consent(args.consent_ref, house_id, config.consent, registry))
    if problems:
        for problem in problems:
            logger.error(problem)
        return 1

    meta = SessionMeta(
        session_id=args.session_id,
        house_id=house_id,
        room=room,
        lighting=args.lighting,
        capture_device=args.device,
        captured_at=args.date,
        consent_reference=args.consent_ref,
        trusted_classes=tuple(c.strip() for c in args.classes.split(",") if c.strip()),
        notes=args.notes,
    )

    try:
        result = ingest_session(
            inbox_dir=config.inbox_dir,
            meta=meta,
            config=config,
            dest_root=dest_root,
            license_str=_capture_license(args.sources_config),
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
        logger.error(str(e))
        return 1

    for name, reason in result.rejected:
        logger.warning(f"rejected {name}: {reason}")
    logger.info(
        f"{'[dry-run] ' if args.dry_run else ''}session {meta.session_id}: "
        f"{result.accepted} accepted, {len(result.rejected)} rejected"
    )

    if result.accepted == 0:
        logger.error("No images accepted — nothing ingested")
        return 1
    if result.rejected:
        return 1 if args.strict else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
