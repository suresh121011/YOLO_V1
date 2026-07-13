"""
scripts.dataset.05_remap_classes — Class Remapping CLI
======================================================

Rewrites every raw source's YOLO labels from local ids into the 23-class
taxonomy using :mod:`src.dataset.remap`. Idempotent per source via the
``.remap_done.json`` sentinel.

Usage:
    python scripts/dataset/05_remap_classes.py --all
    python scripts/dataset/05_remap_classes.py --source coco
    python scripts/dataset/05_remap_classes.py --source coco --force

DVC integration:
    Invoked by the ``remap_classes`` stage (with --all).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.remap import REMAP_TABLES, remap_label_dir
from src.dataset.sources_config import (
    DEFAULT_SOURCES_CONFIG_PATH,
    SourcesConfig,
    load_sources_config,
)
from src.utils.config_helpers import get_class_names_from_data_yaml, load_data_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_table(source_name: str, config: SourcesConfig, data_yaml: Path) -> dict[str, int]:
    """Resolve the remap table for a source, applying per-dataset aliases.

    Roboflow Universe datasets may declare ``classes: {source_name:
    taxonomy_name}`` aliases in dataset_sources.yaml; these are resolved to
    taxonomy ids through configs/data.yaml.
    """
    source = config.sources[source_name]
    table = dict(REMAP_TABLES.get(source.remap_table or "identity", {}))

    datasets = source.options.get("datasets") or []
    if datasets:
        names = get_class_names_from_data_yaml(load_data_config(data_yaml))
        name_to_id = {v: k for k, v in names.items()}
        for entry in datasets:
            for alias, taxonomy_name in (entry.get("classes") or {}).items():
                if taxonomy_name not in name_to_id:
                    raise ValueError(
                        f"Unknown taxonomy class '{taxonomy_name}' in alias map "
                        f"for source '{source_name}'"
                    )
                table[str(alias)] = name_to_id[taxonomy_name]
    return table


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Remap raw source labels into the 23-class taxonomy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=str, help="Remap one source by name.")
    group.add_argument("--all", action="store_true", help="Remap every enabled source.")
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
        help="Path to data.yaml (for alias resolution).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even where the remap sentinel exists (DANGEROUS).",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns 0 on success, 1 on error."""
    args = parse_args()

    try:
        config = load_sources_config(args.sources_config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1

    names = [args.source] if args.source else list(config.sources)
    had_error = False

    for name in names:
        source = config.sources.get(name)
        if source is None:
            logger.error(f"Unknown source '{name}'")
            return 1
        if not config.is_source_allowed(name):
            logger.info(f"[{name}] disabled/gated — skipping")
            continue
        if not (source.output_dir / "labels").exists():
            logger.info(f"[{name}] no labels directory — skipping")
            continue

        try:
            table = build_table(name, config, args.data_config)
            result = remap_label_dir(source.output_dir, table, force=args.force)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"[{name}] remap failed: {e}")
            had_error = True
            continue

        if not result.skipped:
            logger.info(
                f"[{name}] remapped {result.annotations_remapped} annotations "
                f"({result.annotations_dropped} dropped)"
            )

    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
