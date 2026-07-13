"""
scripts.dataset.07_merge_datasets — Dataset Merge CLI
=====================================================

Merges all remapped raw sources into ``data/merged`` via
:mod:`src.dataset.merge`: quality/indoor filtering, flip-robust
cross-source dedup (BEFORE splitting — governance rule), per-image
provenance, and the label-completeness map.

Source priority (dedup keeps the first occurrence): custom captures
first, then config order, negatives last.

Usage:
    python scripts/dataset/07_merge_datasets.py
    python scripts/dataset/07_merge_datasets.py --output data/merged

DVC integration:
    Invoked by the ``merge_datasets`` stage.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.merge import MergeSource, merge_sources
from src.dataset.sources_config import DEFAULT_SOURCES_CONFIG_PATH, load_sources_config
from src.utils.config_helpers import get_class_names_from_data_yaml, load_data_config
from src.utils.dataset_utils import find_image_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Merge remapped raw sources into data/merged with lineage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sources-config",
        type=Path,
        default=DEFAULT_SOURCES_CONFIG_PATH,
        help="Path to dataset_sources.yaml.",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to data.yaml for class names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Merged output directory (default: paths.merged_root from config).",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns 0 on success, 1 on error."""
    args = parse_args()

    try:
        config = load_sources_config(args.sources_config)
        class_names = get_class_names_from_data_yaml(load_data_config(args.data_config))
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1

    output_dir = args.output if args.output is not None else config.merged_root

    # Priority order: custom captures win dedup ties; negatives never do.
    ordered = sorted(
        config.sources,
        key=lambda n: (0 if n == "custom_captures" else (2 if n == "negatives" else 1)),
    )

    merge_inputs: list[MergeSource] = []
    for name in ordered:
        if not config.is_source_allowed(name):
            logger.info(f"[{name}] disabled/gated — excluded from merge")
            continue
        source = config.sources[name]
        if not find_image_files(source.output_dir / "images"):
            logger.info(f"[{name}] no images — excluded from merge")
            continue
        merge_inputs.append(
            MergeSource(
                name=name,
                root=source.output_dir,
                trusted_classes=list(source.trusted_classes),
                apply_indoor_filter=bool(source.options.get("indoor_filter", False)),
                allow_empty_labels=(name == "negatives"),
            )
        )

    if not merge_inputs:
        logger.error("No sources with images found — run the download stages first")
        return 1

    manifest = merge_sources(
        sources=merge_inputs,
        output_dir=output_dir,
        dedup_settings=config.dedup,
        indoor_settings=config.indoor_filter,
        class_names=class_names,
        notes=f"{config.mode}-mode merge of {[s.name for s in merge_inputs]}",
    )

    if not manifest.image_provenance:
        logger.error("Merge produced zero images")
        return 1

    logger.info(f"✅ Merge complete: {len(manifest.image_provenance)} images → {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
