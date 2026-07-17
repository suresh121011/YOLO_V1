"""
scripts.dataset.15_apply_verified_labels — Verified-Labels Overlay
(DVC: apply_verified_labels)
====================================================================

Phase-5 M3 entry point. Writes ``data/merged_verified/labels`` as base
labels UNION verified deltas (ADR-P5-05) — ``data/merged`` stays immutable;
the split stage reads images from ``data/merged`` and labels from this
overlay (``split.source_labels_dir``).

Unlike the M2 human-loop stages, this stage is a pure deterministic
derivation of already-frozen outputs — it is a normal (non-frozen) DVC stage
and reruns safely any time its deps change.

Exit codes: 0 = ok. (There is nothing to validate here — schema/consistency
checks live in preflight gate G9, M3 commit 4.)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.annotation.apply import build_verified_labels_overlay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build the verified-labels overlay (Phase-5 M3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--merged-labels-dir", type=Path, default=Path("data/merged/labels"))
    parser.add_argument(
        "--verified-labels-dir", type=Path, default=Path("data/annotation/verified_labels")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/merged_verified/labels"))
    return parser.parse_args()


def main() -> int:
    """Entry point. Always returns 0."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parse_args()

    result = build_verified_labels_overlay(
        merged_labels_dir=args.merged_labels_dir,
        verified_labels_dir=args.verified_labels_dir,
        output_dir=args.output_dir,
    )
    logger.info(
        f"Wrote {result.images_total} label file(s) "
        f"({result.images_with_deltas} with deltas) to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
