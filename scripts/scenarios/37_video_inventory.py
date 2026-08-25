"""
scripts.scenarios.37_video_inventory — Video Inventory Tool (D-1)
================================================================

Scans all extracted videos in the staging directory, probes media
metadata (duration, fps, resolution, frame count), computes SHA256 for
duplicate detection, and infers source_category and polarity from the
folder structure.

Outputs a comprehensive CSV inventory:
    ``data/scenario_engine/staging/video_inventory.csv``

Usage:
    python scripts/scenarios/37_video_inventory.py
    python scripts/scenarios/37_video_inventory.py --staging-dir data/scenario_engine/staging/extracted
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_STAGING = REPO / "data" / "scenario_engine" / "staging" / "extracted"
OUTPUT_CSV = REPO / "data" / "scenario_engine" / "staging" / "video_inventory.csv"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

INVENTORY_FIELDS = [
    "video_id",
    "original_path",
    "original_zip_folder",
    "filename",
    "extension",
    "duration_s",
    "fps",
    "width",
    "height",
    "frame_count",
    "file_size_mb",
    "sha256",
    "source_category",
    "polarity",
    "negative_kind",
    "review_status",
    "is_corrupt",
    "is_duplicate",
    "duplicate_of",
    "notes",
]

# ── Polarity inference ──────────────────────────────────────────────────────

POLARITY_MAP = {
    "positive": "POSITIVE",
    "negative": "NEGATIVE",
    "edge": "EDGE",
}

NEGATIVE_KIND_MAP = {
    "a person holding pen": "SAFE_OBJECT",
    "a person holding spoon": "SAFE_OBJECT",
    "a person holding remote": "SAFE_OBJECT",
    "a person holding fork": "SAFE_OBJECT",
    "a person holding scissors": "WRONG_OBJECT",
    "person holding pen": "SAFE_OBJECT",
    "person holding spoon": "SAFE_OBJECT",
    "person holding remote": "SAFE_OBJECT",
}


def infer_source_category(path: Path, staging_root: Path) -> str:
    """Extract the SC-XXX-NNN source category from the folder hierarchy."""
    rel = path.relative_to(staging_root)
    parts = rel.parts
    if parts:
        top_folder = parts[0]
        match = re.match(r"(SC-[A-Z]{3}-\d{3})", top_folder)
        if match:
            return match.group(1)
    return "UNKNOWN"


def infer_polarity(path: Path, staging_root: Path) -> str:
    """Infer polarity (POSITIVE/NEGATIVE/EDGE) from subfolder name."""
    rel = path.relative_to(staging_root)
    for part in rel.parts:
        key = part.strip().lower()
        if key in POLARITY_MAP:
            return POLARITY_MAP[key]
    return "UNKNOWN"


def infer_negative_kind(path: Path, staging_root: Path) -> str:
    """For NEGATIVE clips, infer the structured negative category."""
    rel = path.relative_to(staging_root)
    for part in rel.parts:
        key = part.strip().lower()
        if key in NEGATIVE_KIND_MAP:
            return NEGATIVE_KIND_MAP[key]
    return ""


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ffprobe_full(path: Path) -> dict:
    """Run ffprobe and return full JSON payload."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        payload = json.loads(result.stdout or "{}")
        return payload if isinstance(payload, dict) else {}
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            json.JSONDecodeError, OSError) as e:
        logger.warning(f"ffprobe failed for {path.name}: {e}")
        return {}


def parse_frame_rate(raw) -> float:
    """Parse ffprobe's rational frame-rate notation."""
    text = str(raw or "").strip()
    if not text or text in {"0/0", "N/A"}:
        return 0.0
    if "/" in text:
        num, _, den = text.partition("/")
        try:
            d = float(den)
            return float(num) / d if d else 0.0
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def probe_video(path: Path) -> dict:
    """Probe a video file and return metadata dict."""
    payload = ffprobe_full(path)
    is_corrupt = not bool(payload.get("streams"))

    duration = 0.0
    fps = 0.0
    width = 0
    height = 0
    frame_count = 0

    try:
        duration = float(payload.get("format", {}).get("duration", 0.0) or 0.0)
    except (TypeError, ValueError):
        pass

    for stream in payload.get("streams", []) or []:
        if stream.get("codec_type") != "video":
            continue
        fps = parse_frame_rate(stream.get("avg_frame_rate")) or parse_frame_rate(
            stream.get("r_frame_rate")
        )
        width = int(stream.get("width", 0) or 0)
        height = int(stream.get("height", 0) or 0)
        try:
            frame_count = int(stream.get("nb_frames", 0) or 0)
        except (TypeError, ValueError):
            frame_count = 0
        if not duration:
            try:
                duration = float(stream.get("duration", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
        break

    return {
        "duration_s": round(duration, 2),
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "is_corrupt": is_corrupt,
    }


def generate_video_id(source_category: str, polarity: str, index: int) -> str:
    """Generate a deterministic video_id."""
    pol_code = polarity[0].lower() if polarity != "UNKNOWN" else "u"
    return f"{source_category}_{pol_code}{index:03d}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate video inventory CSV from staging directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=DEFAULT_STAGING,
        help="Path to the extracted staging directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help="Path for the output CSV.",
    )
    args = parser.parse_args(argv)

    staging = args.staging_dir
    if not staging.exists():
        logger.error(f"Staging directory not found: {staging}")
        return 1

    # Discover all video files
    videos = sorted(
        p for p in staging.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    logger.info(f"Found {len(videos)} video files in {staging}")

    # Process each video
    rows: list[dict] = []
    sha_index: dict[str, str] = {}  # sha256 -> first video_id
    id_counters: Counter = Counter()
    corrupt_count = 0
    duplicate_count = 0

    for i, vpath in enumerate(videos, 1):
        if i % 50 == 0 or i == 1:
            logger.info(f"Processing {i}/{len(videos)}: {vpath.name}")

        source_cat = infer_source_category(vpath, staging)
        polarity = infer_polarity(vpath, staging)
        negative_kind = infer_negative_kind(vpath, staging) if polarity == "NEGATIVE" else ""

        # Generate unique video_id
        id_counters[(source_cat, polarity)] += 1
        video_id = generate_video_id(source_cat, polarity, id_counters[(source_cat, polarity)])

        # Get file size
        file_size_mb = round(vpath.stat().st_size / (1024 * 1024), 2)

        # Compute SHA256
        sha = compute_sha256(vpath)

        # Check for duplicates
        is_dup = sha in sha_index
        dup_of = sha_index.get(sha, "")
        if not is_dup:
            sha_index[sha] = video_id

        # Probe media
        meta = probe_video(vpath)

        if meta["is_corrupt"]:
            corrupt_count += 1
        if is_dup:
            duplicate_count += 1

        # Get the ZIP-level folder name
        rel = vpath.relative_to(staging)
        zip_folder = rel.parts[0] if rel.parts else ""

        row = {
            "video_id": video_id,
            "original_path": str(vpath.relative_to(staging)),
            "original_zip_folder": zip_folder,
            "filename": vpath.name,
            "extension": vpath.suffix.lower(),
            "duration_s": meta["duration_s"],
            "fps": meta["fps"],
            "width": meta["width"],
            "height": meta["height"],
            "frame_count": meta["frame_count"],
            "file_size_mb": file_size_mb,
            "sha256": sha,
            "source_category": source_cat,
            "polarity": polarity,
            "negative_kind": negative_kind,
            "review_status": "pending",
            "is_corrupt": meta["is_corrupt"],
            "is_duplicate": is_dup,
            "duplicate_of": dup_of,
            "notes": "",
        }
        rows.append(row)

    # Write CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Inventory written: {args.output}")

    # Summary
    cat_counts = Counter(r["source_category"] for r in rows)
    pol_counts = Counter(r["polarity"] for r in rows)

    print(f"\n{'=' * 60}")
    print("VIDEO INVENTORY SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total videos:    {len(rows)}")
    print(f"  Corrupt:         {corrupt_count}")
    print(f"  Duplicates:      {duplicate_count}")
    print(f"  Unique:          {len(rows) - duplicate_count}")
    print()
    print("  By polarity:")
    for pol in ["POSITIVE", "NEGATIVE", "EDGE", "UNKNOWN"]:
        if pol_counts[pol]:
            print(f"    {pol:12s}: {pol_counts[pol]}")
    print()
    print("  By source category:")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"    {cat:12s}: {cnt}")
    print(f"\n  Output: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
