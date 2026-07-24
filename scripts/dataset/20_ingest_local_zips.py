#!/usr/bin/env python3
"""
scripts/dataset/20_ingest_local_zips.py
=======================================
Idempotent ingestion of the 22 locally-collected ZIP archives from the
``Dataset/`` folder into ``data/raw/local_captures/``.

Each outer ZIP is a Google Drive folder export that contains one or more
inner Roboflow ``.yolov11.zip`` datasets. This script handles the
double-nesting transparently.

NOTE (Windows): All ZIP-internal path operations MUST use PurePosixPath
(forward slashes). zipfile.ZipFile.namelist() always returns forward-slash
paths regardless of OS; using pathlib.Path would convert them to backslashes
on Windows, causing namelist lookups to always fail.

Run once (or re-run safely -- already-ingested sources are skipped):
    python scripts/dataset/20_ingest_local_zips.py [--zip-dir path] [--dry-run]

It will:
 1. Print a pre-flight summary of every ZIP => detected class => taxonomy mapping.
 2. Extract each outer archive, find inner .yolov11.zip files.
 3. For each inner ZIP read data.yaml to get local class names.
 4. Build local_id => taxonomy_id mapping per inner ZIP using per-archive tables.
 5. Merge all train/valid/test splits into ONE flat, top-level
    ``data/raw/local_captures/images/`` + ``labels/`` pair. Filenames are
    slug-prefixed (``<slug>__<name>``) so sources never collide. This flat
    layout is what the remap_classes and merge_datasets stages consume
    (they look for ``<source.output_dir>/images`` + ``/labels``).
 6. Write a ``source_classes.json`` sidecar per source (under <slug>/).
 7. Generate a ``manifest.json`` (SourceManifest) per source (under <slug>/).
 8. Write ``data/raw/local_captures/ingest_index.json``.

Layout:
    data/raw/local_captures/
      images/  <slug>__<name>.<ext>   ← flat, all sources (consumed by remap/merge)
      labels/  <slug>__<name>.txt     ← flat, all sources (taxonomy-space ids)
      <slug>/  manifest.json, source_classes.json, _ingest_done.json  ← provenance

Idempotency:
    An ``_ingest_done.json`` sentinel is written in each slug provenance dir on
    success. Re-running skips that source unless ``--force`` is given (which
    wipes only that slug's prefixed files from the shared flat dirs).

Integration:
    After this script, run ``dvc repro remap_classes`` to propagate remapped
    labels into ``data/interim/`` for the merge stage.

Taxonomy reference  (configs/data.yaml):
    0=person, 1=face, 2=medicine_strip, 3=medicine_bottle, 4=water_bottle,
    5=knife, 6=stove, 7=gas_cylinder, 8=passport, 9=book, 10=charger,
    11=wire, 12=laptop, 13=monitor, 14=cupboard, 15=door, 16=chair,
    17=bed, 18=toilet, 19=sink, 20=wet_floor, 21=walking_stick,
    22=support_handle
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
import tempfile
import zipfile
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZIP_DIR = REPO_ROOT / "Dataset"
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "raw" / "local_captures"

TAXONOMY: dict[int, str] = {
    0: "person",
    1: "face",
    2: "medicine_strip",
    3: "medicine_bottle",
    4: "water_bottle",
    5: "knife",
    6: "stove",
    7: "gas_cylinder",
    8: "passport",
    9: "book",
    10: "charger",
    11: "wire",
    12: "laptop",
    13: "monitor",
    14: "cupboard",
    15: "door",
    16: "chair",
    17: "bed",
    18: "toilet",
    19: "sink",
    20: "wet_floor",
    21: "walking_stick",
    22: "support_handle",
}

# ---------------------------------------------------------------------------
# Per-archive source-class-name => taxonomy-ID remapping tables.
#
# Derived from running _inspect_classes.py against every inner data.yaml.
# Keys are the outer-ZIP stem (lower-cased, spaces stripped).
# Values are {source_class_name: taxonomy_id} -- any name NOT in the dict
# is dropped (annotation not useful for our taxonomy).
#
# bottel: visually verified 2026-07-22 (6 images inspected) -- all are
#         water/beverage bottles; Roboflow called the label "cap".
# medicine_strip: TEMPORARY -- Roboflow-sourced; replace with own data.
# wire: wire-QC dataset (OK = wire present/correct, NO = absent, breadboard).
# ---------------------------------------------------------------------------
_ARCHIVE_CLASS_REMAPS: dict[str, dict[str, int]] = {
    "bed": {
        "bed": 17,
        "person": 0,
        "sink": 19,
        "chair": 16,
        # dropped: backpack, bottle, bowl, cell phone, cup, diningtable,
        #          handbag, refrigerator, sofa, suitcase, tie, toothbrush, "15", "0"
    },
    "book": {
        "Book": 9,
        "book": 9,
    },
    "bottel": {
        "cap": 4,  # bottle cap label => water_bottle (visually verified 2026-07-22)
    },
    "chair": {
        "Chair": 16,
        "chair": 16,
        "occupied": 16,  # occupied chair
        "unoccupied": 16,  # empty chair -- still a chair
    },
    "charger": {
        "Charger-cUjN": 10,
    },
    "cupboard": {
        "Cupboard": 14,
        "Door": 15,
        "Person": 0,
        # dropped: Car, Chair, Sofa, Table, Window, Door handle
    },
    "door": {
        "door": 15,
        "doors": 15,
        # dropped: handle, open
    },
    "face": {
        "a face": 1,
    },
    "gas cylinder": {
        "Gas cylinder": 7,
        "gas cylinder": 7,
        "gas cylinder head": 7,
        "Recognising-Indian-Indian-gas": 7,
        "lpg-cylinder": 7,
    },
    "knife": {
        "Knife": 5,
        "knife": 5,
    },
    "laptop": {
        "laptop": 12,
    },
    "medicine strip": {
        # Per-inner-ZIP inspection: Strip1 and Strip2 use 'tablet',
        # Strip3 may use 'strip' or 'circle'. Map all three => medicine_strip.
        "tablet": 2,
        "strip": 2,
        "circle": 2,  # circular tablet shape -- still a medicine strip
    },
    "monitor": {
        "Monitor-Led": 13,
        # dropped: numeric string class names "1" through "10"
    },
    "passport": {
        "passport": 8,
    },
    "person": {
        "Person": 0,
        "MEN": 0,
        "WOMEN": 0,
        "Persona": 0,
        "hombre": 0,
        "insan": 0,
        "mujer": 0,
    },
    "sink": {
        "sink": 19,
        # dropped: Bathtub, Blender, Chair, Coffemaker, Dish Washer, Hair dryer,
        #          Microwave, Refrigerator, Shower, Stand Mixer, Stove, Toaster,
        #          baking oven, table lamp, tap, wall clock, BBQ, "0"
    },
    "stove": {
        "stove": 6,
        "stove-fire": 6,  # burning stove => still a stove
        # dropped: "0" (junk class name in one inner ZIP)
    },
    "support handle": {
        "crutch": 22,
        "stick": 22,
        # dropped: "0"
    },
    "toilet": {
        "toilet": 18,
        # dropped: towel
    },
    "walking stick": {
        "stick": 21,
        "crutch": 21,
        # dropped: bottle, fire, gun, knife, person (noisy multi-class dataset)
    },
    "wet floor": {
        "Wet-floor": 20,
        "Stagnant Water and Wet Surface - v1 2025-03-21 5:52pm": 20,
        "Wet floor - v3 2026-06-05 10:55am": 20,
        # dropped: separator "======..." string
    },
    "wire": {
        "OK": 11,  # wire correctly wired/present
        "breadboard": 11,  # PCB/breadboard with visible wires
        # dropped: "NO" (no wire / absent = background class)
    },
}

# Outer ZIP stem => (slug, primary_class_name, is_temporary)
# Slug is the output directory name under data/raw/local_captures/
_ZIP_META: dict[str, tuple[str, str, bool]] = {
    "bed": ("bed", "bed", False),
    "book": ("book", "book", False),
    "bottel": ("bottel", "water_bottle", False),
    "chair": ("chair", "chair", False),
    "charger": ("charger", "charger", False),
    "cupboard": ("cupboard", "cupboard", False),
    "door": ("door", "door", False),
    "face": ("face", "face", False),
    "gas cylinder": ("gas_cylinder", "gas_cylinder", False),
    "knife": ("knife", "knife", False),
    "laptop": ("laptop", "laptop", False),
    "medicine strip": ("medicine_strip", "medicine_strip", True),  # TEMPORARY
    "monitor": ("monitor", "monitor", False),
    "passport": ("passport", "passport", False),
    "person": ("person", "person", False),
    "sink": ("sink", "sink", False),
    "stove": ("stove", "stove", False),
    "support handle": ("support_handle", "support_handle", False),
    "toilet": ("toilet", "toilet", False),
    "walking stick": ("walking_stick", "walking_stick", False),
    "wet floor": ("wet_floor", "wet_floor", False),
    "wire": ("wire", "wire", False),
}

LICENSE_NOTE = (
    "Locally collected Roboflow YOLOv11 datasets -- proprietary; "
    "no redistribution without consent. "
    "See docs/04_dataset_engineering/README.md section Licensing."
)
MEDICINE_STRIP_LICENSE_NOTE = (
    "[TEMPORARY] Roboflow Universe medicine_strip dataset -- private license. "
    "Replace with own captured+annotated data before any commercial release."
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
SENTINEL_FILENAME = "_ingest_done.json"
INGEST_INDEX_FILENAME = "ingest_index.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stem_key(zip_path: Path) -> str:
    """Normalise outer ZIP stem => archive key for _ZIP_META lookup."""
    stem = zip_path.stem
    # Pattern: "Book-20260722T103743Z-1-001" => "book"
    # Pattern: "Gas Cylinder-20260722T103919Z-1-001" => "gas cylinder"
    if "-" in stem:
        # Split on first hyphen-digit boundary (timestamp start)
        import re

        stem = re.split(r"-\d{8}T", stem)[0]
    return stem.strip().lower()


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_filename(name: str, max_stem: int = 120) -> str:
    """Return a safe filename, truncating long stems to avoid Windows MAX_PATH.

    Roboflow filenames can be 200+ chars and exceed Windows MAX_PATH (260 chars)
    when combined with the full output directory path.  We truncate the stem
    to ``max_stem`` chars and append an 8-char hash of the ORIGINAL name so
    the truncated name stays unique.

    Args:
        name:     Original filename (basename only, e.g. 'very_long_name.jpg').
        max_stem: Maximum stem length; anything longer triggers truncation.

    Returns:
        Safe filename with the original extension preserved.
    """
    p = PurePosixPath(name)
    stem, ext = p.stem, p.suffix
    if len(stem) <= max_stem:
        return name
    short_hash = hashlib.sha256(name.encode()).hexdigest()[:8]
    return stem[:max_stem] + "_" + short_hash + ext


def _find_images_in(root: Path) -> Iterator[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _parse_data_yaml(data: bytes) -> list[str]:
    """Parse a Roboflow data.yaml and return the list of class names (0-indexed)."""
    if yaml is not None:
        try:
            parsed = yaml.safe_load(data.decode("utf-8", errors="replace"))
            if isinstance(parsed, dict):
                names = parsed.get("names", [])
                if isinstance(names, dict):
                    # Roboflow sometimes writes: names: {0: 'knife', 1: 'Knife'}
                    max_id = max(int(k) for k in names) if names else -1
                    return [names.get(i, names.get(str(i), "")) for i in range(max_id + 1)]
                if isinstance(names, list):
                    return [str(n) for n in names]
        except Exception as e:
            logger.debug("YAML parse error: %s", e)
    # Fallback: grep for 'names:' block
    lines = data.decode("utf-8", errors="replace").splitlines()
    names: list[str] = []
    in_names = False
    for line in lines:
        if line.strip().startswith("names:"):
            in_names = True
            bracket = line.find("[")
            if bracket != -1:
                inner = line[bracket + 1 : line.rfind("]")] if "]" in line else line[bracket + 1 :]
                return [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
            continue
        if in_names:
            stripped = line.strip()
            if stripped.startswith("-"):
                names.append(stripped.lstrip("- ").strip().strip("'\""))
            elif stripped and not stripped.startswith("#"):
                # dict style: "  0: knife" or "  '0': knife"
                import re

                m = re.match(r'^[\'"]?(\d+)[\'"]?:\s*(.*)', stripped)
                if m:
                    names.append(m.group(2).strip().strip("'\""))
                else:
                    break  # non-matching line ends the block
    return names


def _build_local_to_taxonomy(
    class_names: list[str],
    remap_table: dict[str, int],
) -> dict[int, int | None]:
    """Build {local_id: taxonomy_id_or_None} from class name list + remap table."""
    mapping: dict[int, int | None] = {}
    for local_id, name in enumerate(class_names):
        mapping[local_id] = remap_table.get(name)  # None = drop
    return mapping


def coords_to_detection_fields(coords: list[str]) -> list[str] | None:
    """Normalise a YOLO label's coordinate fields to a detection bbox.

    Some Roboflow "YOLOv11" exports are SEGMENTATION (``class x1 y1 x2 y2
    ... xN yN``, an even number ≥ 6 of polygon coords) or OBB (8 coords),
    not detection (``class cx cy w h``, exactly 4 coords). This pipeline
    trains a detector, so polygons/OBB are reduced to their axis-aligned
    bounding box: cx,cy = box centre, w,h = box extent, all clamped to
    [0, 1].

    Args:
        coords: the coordinate tokens (everything after the class id).

    Returns:
        ``["cx", "cy", "w", "h"]`` (6-decimal strings), or ``None`` when the
        line is malformed (odd/`<4` coord count) or degenerate (zero area).
    """
    n = len(coords)
    if n == 4:  # already a detection bbox — keep verbatim
        return coords
    if n < 6 or n % 2 != 0:  # not a valid polygon/OBB (needs ≥3 xy pairs)
        return None
    try:
        xs = [min(1.0, max(0.0, float(coords[i]))) for i in range(0, n, 2)]
        ys = [min(1.0, max(0.0, float(coords[i]))) for i in range(1, n, 2)]
    except ValueError:
        return None
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    if w <= 0.0 or h <= 0.0:  # degenerate polygon → no usable box
        return None
    cx, cy = x0 + w / 2.0, y0 + h / 2.0
    return [f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]


def _remap_label_content(
    text: str,
    mapping: dict[int, int | None],
    stats: dict[str, int],
) -> str:
    """Remap YOLO label file content into taxonomy-id DETECTION labels.

    Rewrites the class id and, when the line is a segmentation polygon / OBB,
    collapses it to an axis-aligned bounding box (see
    :func:`coords_to_detection_fields`).
    """
    out_lines: list[str] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        try:
            local_id = int(parts[0])
        except (ValueError, IndexError):
            stats["dropped_malformed"] = stats.get("dropped_malformed", 0) + 1
            continue
        target = mapping.get(local_id)
        if target is None:
            stats["dropped_unmapped"] = stats.get("dropped_unmapped", 0) + 1
            continue
        bbox = coords_to_detection_fields(parts[1:])
        if bbox is None:
            stats["dropped_malformed"] = stats.get("dropped_malformed", 0) + 1
            continue
        if len(parts) - 1 != 4:  # count polygon→bbox conversions
            stats["polygons_converted"] = stats.get("polygons_converted", 0) + 1
        out_lines.append(" ".join([str(target), *bbox]))
        stats["kept"] = stats.get("kept", 0) + 1
    return "\n".join(out_lines) + ("\n" if out_lines else "")


# ---------------------------------------------------------------------------
# Inner ZIP processing
# ---------------------------------------------------------------------------


def _process_inner_zip(
    inner_zip: zipfile.ZipFile,
    inner_name: str,
    remap_table: dict[str, int],
    images_out: Path,
    labels_out: Path,
    image_hashes: dict[str, str],
    stats: dict[str, int],
    slug_prefix: str = "",
) -> None:
    """Extract and remap one inner .yolov11.zip into the FLAT dest dirs.

    ``slug_prefix`` (e.g. ``"knife__"``) is prepended to every output image
    and label filename so all sources share one flat ``images/`` + ``labels/``
    pair without cross-source collisions (remap/merge consume a flat layout).
    """
    # Read data.yaml for class names
    yaml_entries = [n for n in inner_zip.namelist() if n.endswith("data.yaml")]
    class_names: list[str] = []
    if yaml_entries:
        with inner_zip.open(yaml_entries[0]) as f:
            class_names = _parse_data_yaml(f.read())

    if not class_names:
        logger.warning("    No class names found in %s -- using passthrough", inner_name)

    id_map = _build_local_to_taxonomy(class_names, remap_table)
    mapped_names = {i: (class_names[i], id_map.get(i)) for i in range(len(class_names))}
    logger.debug("    %s class map: %s", inner_name, mapped_names)

    # Collect all image entries from train/valid/test splits (use PurePosixPath
    # for suffix check -- ZIP namelist always uses forward slashes on all OSes).
    all_entries = inner_zip.namelist()
    img_entries = [
        n
        for n in all_entries
        if not n.endswith("/") and PurePosixPath(n).suffix.lower() in IMAGE_EXTS
    ]

    for img_entry in img_entries:
        orig_img_name = PurePosixPath(img_entry).name
        # Flat-layout: slug prefix guarantees cross-source uniqueness.
        img_name = slug_prefix + _safe_filename(orig_img_name)  # guard Windows MAX_PATH
        dest_img = images_out / img_name

        if dest_img.exists():
            stats["dups"] = stats.get("dups", 0) + 1
            continue

        # Write image
        with inner_zip.open(img_entry) as src:
            dest_img.write_bytes(src.read())
        image_hashes[img_name] = _sha256(dest_img)
        stats["images"] = stats.get("images", 0) + 1

        # Find matching label: train/images/foo.jpg => train/labels/foo.txt
        # MUST use PurePosixPath -- Path() on Windows would produce backslashes
        # which never match ZIP namelist entries (always forward-slash).
        p = PurePosixPath(img_entry)
        label_entry = str(p.parent.parent / "labels" / p.with_suffix(".txt").name)

        # Label dest uses safe stem matching the (possibly truncated) img name
        safe_stem = PurePosixPath(img_name).stem
        dest_lbl = labels_out / (safe_stem + ".txt")
        if label_entry in all_entries:
            with inner_zip.open(label_entry) as lf:
                raw_text = lf.read().decode("utf-8", errors="replace")
            remapped = _remap_label_content(raw_text, id_map, stats)
            dest_lbl.write_text(remapped, encoding="utf-8")
        else:
            dest_lbl.write_text("", encoding="utf-8")


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def preflight_summary(zip_dir: Path) -> list[dict]:
    zips = sorted(zip_dir.glob("*.zip"))
    if not zips:
        logger.error("No ZIP files found in %s", zip_dir)
        sys.exit(1)

    records: list[dict] = []
    print("\n" + "=" * 84)
    print("  PRE-FLIGHT INGESTION SUMMARY")
    print("=" * 84)
    print(f"  {'ZIP Archive':<47} {'Slug':<20} {'Primary Class':<18} {'Note'}")
    print("-" * 84)

    for z in zips:
        key = _stem_key(z)
        if key not in _ZIP_META:
            print(f"  !! {z.name:<44} UNMAPPED -- SKIPPED")
            records.append({"zip": z, "key": key, "mapped": False})
            continue

        slug, primary_cls, is_temp = _ZIP_META[key]
        note = "[TEMPORARY]" if is_temp else ""
        sz_mb = z.stat().st_size / 1_048_576
        inner_count = _count_inner_zips(z)
        print(
            f"  {z.name:<47} {slug:<20} {primary_cls:<18} {note} "
            f"[{inner_count} inner, {sz_mb:.0f} MB]"
        )
        records.append(
            {
                "zip": z,
                "key": key,
                "mapped": True,
                "slug": slug,
                "primary_class": primary_cls,
                "temporary": is_temp,
                "size_mb": sz_mb,
                "inner_count": inner_count,
            }
        )

    print("=" * 84)
    total_gb = sum(r.get("size_mb", 0) for r in records) / 1024
    print(f"  Total: {len(records)} archives, ~{total_gb:.1f} GB on disk\n")
    return records


def _count_inner_zips(outer: Path) -> int:
    try:
        with zipfile.ZipFile(outer) as zf:
            return sum(1 for n in zf.namelist() if n.endswith(".zip"))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------


def ingest_one(
    rec: dict,
    out_root: Path,
    force: bool = False,
) -> dict | None:
    z: Path = rec["zip"]
    key: str = rec["key"]
    slug: str = rec["slug"]
    primary_cls: str = rec["primary_class"]
    is_temp: bool = rec.get("temporary", False)

    dest = out_root / slug
    sentinel = dest / SENTINEL_FILENAME

    if sentinel.exists() and not force:
        logger.info("SKIP (already done): %s => %s", z.name, slug)
        return json.loads(sentinel.read_text(encoding="utf-8"))

    remap_table = _ARCHIVE_CLASS_REMAPS.get(key, {})
    logger.info("Ingesting: %s  =>  %s  (primary=%s)", z.name, slug, primary_cls)
    if not remap_table:
        logger.warning("  No remap table for '%s' -- all annotations will be dropped!", key)

    # Flat layout (remap/merge consume out_root/images + out_root/labels).
    # Per-source provenance (manifest.json, source_classes.json, sentinel)
    # still lives under dest = out_root/<slug>/. Filenames are slug-prefixed
    # so the shared flat dirs never collide across sources.
    slug_prefix = f"{slug}__"
    images_out = out_root / "images"
    labels_out = out_root / "labels"

    # --force: wipe only THIS slug's files from the shared flat dirs so
    # re-ingestion starts fully fresh (otherwise existing files are treated as
    # dups and image_count stays 0). Other sources' files are untouched.
    if force:
        for sub in (images_out, labels_out):
            if sub.exists():
                for stale in sub.glob(f"{slug_prefix}*"):
                    stale.unlink()
        if sentinel.exists():
            sentinel.unlink()

    dest.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    image_hashes: dict[str, str] = {}
    stats: dict[str, int] = {}

    with tempfile.TemporaryDirectory(prefix="yolo_ingest_"):
        logger.info("  Extracting outer ZIP %s ...", z.name)
        try:
            with zipfile.ZipFile(z, "r") as outer_zf:
                # Find inner .zip files
                inner_zip_entries = [n for n in outer_zf.namelist() if n.endswith(".zip")]
                if not inner_zip_entries:
                    logger.warning("  No inner ZIPs found in %s -- skipping", z.name)
                    return None

                logger.info("  Found %d inner ZIP(s)", len(inner_zip_entries))
                for inner_entry in sorted(inner_zip_entries):
                    logger.info("    Processing inner: %s", inner_entry)
                    with outer_zf.open(inner_entry) as inner_file:
                        inner_bytes = io.BytesIO(inner_file.read())
                    try:
                        with zipfile.ZipFile(inner_bytes) as inner_zf:
                            _process_inner_zip(
                                inner_zf,
                                inner_entry,
                                remap_table,
                                images_out,
                                labels_out,
                                image_hashes,
                                stats,
                                slug_prefix=slug_prefix,
                            )
                    except zipfile.BadZipFile as e:
                        logger.error("    Bad inner ZIP %s: %s", inner_entry, e)
                        continue

        except zipfile.BadZipFile as e:
            logger.error("  Bad outer ZIP %s: %s -- SKIPPED", z.name, e)
            return None

    image_count = stats.get("images", 0)
    if image_count == 0:
        logger.warning("  No images extracted from %s -- skipping", z.name)
        return None

    annotation_count = stats.get("kept", 0)
    logger.info(
        "  Done: %d images, %d annotations kept, %d dropped (unmapped), "
        "%d dropped (malformed), %d dups skipped",
        image_count,
        annotation_count,
        stats.get("dropped_unmapped", 0),
        stats.get("dropped_malformed", 0),
        stats.get("dups", 0),
    )

    # Identify which taxonomy classes actually appear in output labels
    # (only THIS slug's prefixed labels in the shared flat dir).
    taxonomy_classes_used: set[str] = set()
    for lbl in labels_out.glob(f"{slug_prefix}*.txt"):
        if lbl.stat().st_size > 0:
            for line in lbl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        tid = int(line.split()[0])
                        taxonomy_classes_used.add(TAXONOMY.get(tid, str(tid)))
                    except (ValueError, IndexError):
                        pass

    # source_classes.json -- maps "0" => primary_class (identity remap after ingestion)
    # We write one entry per taxonomy class that appears, since after ingestion
    # all labels are already in taxonomy ID space.
    sc_dict = {
        str(tid): TAXONOMY[tid] for tid in TAXONOMY if TAXONOMY[tid] in taxonomy_classes_used
    }
    if not sc_dict:
        sc_dict = {"0": primary_cls}  # fallback
    (dest / "source_classes.json").write_text(
        json.dumps(sc_dict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # manifest.json
    license_txt = MEDICINE_STRIP_LICENSE_NOTE if is_temp else LICENSE_NOTE
    manifest = {
        "schema_version": 1,
        "source": f"local_captures/{slug}",
        "license": license_txt,
        "url": f"local://Dataset/{z.name}",
        "retrieved_at": _utc_now(),
        "query": {
            "zip": z.name,
            "inner_zips": rec.get("inner_count", "?"),
            "primary_taxonomy_class": primary_cls,
            "taxonomy_classes_found": sorted(taxonomy_classes_used),
            "temporary": is_temp,
        },
        "image_count": image_count,
        "class_counts": {cls: annotation_count for cls in sorted(taxonomy_classes_used)},
        "trusted_classes": sorted(taxonomy_classes_used),
        "image_hashes": image_hashes,
        "notes": (
            f"Ingested by 20_ingest_local_zips.py from {rec.get('inner_count', '?')} "
            f"inner Roboflow YOLOv11 ZIPs. "
            + ("TEMPORARY SOURCE -- replace before release. " if is_temp else "")
            + f"Annotations kept: {annotation_count}, "
            f"dropped unmapped: {stats.get('dropped_unmapped', 0)}, "
            f"dups skipped: {stats.get('dups', 0)}."
        ),
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    summary = {
        "slug": slug,
        "zip": z.name,
        "primary_class": primary_cls,
        "taxonomy_classes": sorted(taxonomy_classes_used),
        "temporary": is_temp,
        "image_count": image_count,
        "annotation_count": annotation_count,
        "dropped_unmapped": stats.get("dropped_unmapped", 0),
        "dropped_malformed": stats.get("dropped_malformed", 0),
        "duplicates_skipped": stats.get("dups", 0),
        "ingested_at": _utc_now(),
    }
    sentinel.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# Ingest index
# ---------------------------------------------------------------------------


def write_ingest_index(out_root: Path, results: list[dict]) -> None:
    index_path = out_root / INGEST_INDEX_FILENAME
    index = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "sources": {
            r["slug"]: {
                "primary_class": r["primary_class"],
                "taxonomy_classes": r.get("taxonomy_classes", []),
                "image_count": r["image_count"],
                "annotation_count": r["annotation_count"],
                "temporary": r.get("temporary", False),
                "manifest": f"data/raw/local_captures/{r['slug']}/manifest.json",
            }
            for r in results
            if "slug" in r
        },
    }
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Ingest index => %s", index_path.relative_to(REPO_ROOT))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest local Drive ZIP archives into data/raw/local_captures/",
    )
    parser.add_argument("--zip-dir", type=Path, default=DEFAULT_ZIP_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print summary only; do not write any files."
    )
    parser.add_argument("--force", action="store_true", help="Re-ingest even if sentinel exists.")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="SLUG",
        help="Ingest only these slugs (e.g. --only bottel knife).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="After ingestion validate images/labels/sidecar exist.",
    )
    args = parser.parse_args()

    zip_dir: Path = args.zip_dir.resolve()
    out_dir: Path = args.out_dir.resolve()

    if not zip_dir.is_dir():
        logger.error("ZIP directory not found: %s", zip_dir)
        sys.exit(1)

    records = preflight_summary(zip_dir)
    mapped = [r for r in records if r.get("mapped")]
    unmapped = [r for r in records if not r.get("mapped")]

    if unmapped:
        print(f"!! {len(unmapped)} archive(s) UNMAPPED and will be skipped:")
        for u in unmapped:
            print(f"   {u['zip'].name}  (key='{u['key']}')")
        print()

    if args.only:
        slugs = {s.lower() for s in args.only}
        mapped = [r for r in mapped if r["slug"] in slugs]
        logger.info("Filtered to %d archives: %s", len(mapped), args.only)

    if args.dry_run:
        print("[DRY RUN] No files will be written.")
        for r in mapped:
            inner = r.get("inner_count", "?")
            tbl = _ARCHIVE_CLASS_REMAPS.get(r["key"], {})
            print(
                f"  {r['zip'].name} => {r['slug']}  "
                f"({inner} inner ZIPs, {len(tbl)} class mappings)"
            )
        sys.exit(0)

    out_dir.mkdir(parents=True, exist_ok=True)
    successes: list[dict] = []
    failures: list[str] = []

    for rec in mapped:
        result = ingest_one(rec, out_dir, force=args.force)
        if result is None:
            failures.append(rec["zip"].name)
        else:
            successes.append(result)

    if successes:
        write_ingest_index(out_dir, successes)

    print("\n" + "=" * 84)
    print("  INGESTION COMPLETE")
    print("=" * 84)
    total_imgs = sum(r.get("image_count", 0) for r in successes)
    total_ann = sum(r.get("annotation_count", 0) for r in successes)
    print(f"  Sources ingested : {len(successes)}")
    print(f"  Total images     : {total_imgs:,}")
    print(f"  Total annotations: {total_ann:,}")
    if failures:
        print(f"  Failures         : {len(failures)}")
        for f in failures:
            print(f"    !!  {f}")
    print(f"\n  Output root: {out_dir}")

    if args.validate:
        print("\nValidating ...")
        ok = True
        for r in successes:
            slug = r["slug"]
            imgs = list((out_dir / "images").glob(f"{slug}__*"))
            sc = out_dir / slug / "source_classes.json"
            mf = out_dir / slug / "manifest.json"
            if not imgs:
                print(f"  FAIL  {slug}: no images")
                ok = False
            elif not sc.exists() or not mf.exists():
                print(f"  FAIL  {slug}: missing sidecar files")
                ok = False
            else:
                ann = r.get("annotation_count", 0)
                classes = ", ".join(r.get("taxonomy_classes", []))
                print(f"  OK    {slug}: {len(imgs):>5} imgs, " f"{ann:>6} ann  [{classes}]")
        if ok:
            print("All sources validated successfully.")
        else:
            sys.exit(1)

    print("\n" + "=" * 84)
    print("  NEXT STEPS")
    print("=" * 84)
    print("  NOTE: Labels are already in taxonomy ID space after ingestion.")
    print("  The remap_classes stage uses remap_table: identity for local_captures.")
    print()
    print("  Run:  dvc repro remap_classes")
    print("  Then: dvc repro merge_datasets")
    print("  Then: dvc repro  (full pipeline through QA)")
    print()


if __name__ == "__main__":
    main()
