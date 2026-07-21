"""
src.dataset.annotation.apply — Verified-Labels Overlay Builder
=================================================================

Builds ``data/merged_verified/labels`` as base labels
(``data/merged/labels``, immutable) UNION verified deltas
(``data/annotation/verified_labels``, M2's target-class-only boxes) —
ADR-P5-05. ``data/merged`` is never mutated; the split stage reads images
from ``data/merged`` and labels from this overlay
(``split.source_labels_dir``) — no image duplication at 10-30k scale.

An empty ledger (hence an empty/missing ``verified_labels`` dir — nothing
verified yet) produces a byte-identical passthrough of
``data/merged/labels`` — the golden regression this milestone pins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ApplyResult:
    """Summary of one overlay build."""

    images_total: int = 0
    images_with_deltas: int = 0
    delta_lines_added: int = 0


def build_verified_labels_overlay(
    merged_labels_dir: Path,
    verified_labels_dir: Path,
    output_dir: Path,
) -> ApplyResult:
    """Write the overlay: base label UNION verified deltas, per image.

    ``merged_labels_dir`` is the data-of-record for which images exist (the
    merge stage writes exactly one label file per merged image, possibly
    empty) — this function never invents or drops an image.

    Args:
        merged_labels_dir:   ``data/merged/labels`` (base, immutable).
        verified_labels_dir: ``data/annotation/verified_labels`` (target-
                             class-only delta boxes; may not exist yet).
        output_dir:          ``data/merged_verified/labels``. Wiped and
                             rewritten wholesale each run — deterministic,
                             matches the DVC out-regeneration convention (a
                             base image later dropped must not leave a
                             stale overlay file behind).

    Returns:
        Summary counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.txt"):
        stale.unlink()

    result = ApplyResult()
    for base_path in sorted(merged_labels_dir.glob("*.txt")):
        stem = base_path.stem
        result.images_total += 1
        base_text = base_path.read_text(encoding="utf-8")
        delta_path = verified_labels_dir / f"{stem}.txt"

        if delta_path.exists():
            delta_text = delta_path.read_text(encoding="utf-8")
            result.images_with_deltas += 1
            result.delta_lines_added += sum(1 for line in delta_text.splitlines() if line.strip())
            combined = base_text
            if combined and not combined.endswith("\n"):
                combined += "\n"
            combined += delta_text
        else:
            combined = base_text

        (output_dir / f"{stem}.txt").write_text(combined, encoding="utf-8")

    logger.info(
        f"Verified-labels overlay: {result.images_total} images, "
        f"{result.images_with_deltas} with deltas, {result.delta_lines_added} delta line(s) added"
    )
    return result
