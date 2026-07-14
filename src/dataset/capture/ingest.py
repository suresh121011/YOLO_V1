"""
src.dataset.capture.ingest — Capture Session Ingest
===================================================

Moves validated phone photos from a staging inbox into the canonical
capture tree (``data/raw/custom_captures`` or the locked eval set root),
producing the provenance manifests the rest of the pipeline consumes.

Canonical tree (flat — merge/QA/split work on it with zero changes):

    <root>/
      images/    h01_kitchen_s001_0007.jpg   (session-prefixed, sequential)
      labels/                                (taxonomy YOLO labels; written by
                                              09_import_annotations --finalize)
      manifests/<session_id>.json            (CaptureSessionManifest each)
      manifest.json                          (aggregate SourceManifest — the
                                              QA license gate reads this)
      LOCKED.json                            (eval sets only, after lock)

Per-image intake: extension/size gates → corruption check → minimum
dimension → intra-session duplicate check → EXIF/GPS strip → post-strip
verification → SHA-256 provenance hash (computed AFTER stripping — the
stripped file is the provenance origin).

Ingest is copy-only: the inbox is never deleted or modified.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from src.dataset.capture.config import CaptureConfig
from src.dataset.capture.consent import load_consent_registry, verify_consent
from src.dataset.capture.exif import inspect_metadata, strip_metadata
from src.dataset.dedup import DedupIndex
from src.dataset.manifest import (
    MANIFEST_FILENAME,
    CaptureSessionManifest,
    SourceManifest,
    utc_now_iso,
)
from src.dataset.sources_config import DedupSettings
from src.utils.dataset_utils import compute_file_hash, find_image_files
from src.utils.image_utils import get_image_dimensions, validate_image

logger = logging.getLogger(__name__)

LOCKED_FILENAME = "LOCKED.json"

#: Fallback license string when dataset_sources.yaml is unavailable.
DEFAULT_CAPTURE_LICENSE = "proprietary (consent-gated; see data/consent/README.md)"

_SUBDIRS = ("images", "labels", "manifests")


@dataclass(frozen=True)
class SessionMeta:
    """Operator-supplied metadata for one capture session."""

    session_id: str
    house_id: str
    room: str
    lighting: str
    capture_device: str
    captured_at: str
    consent_reference: str
    trusted_classes: tuple[str, ...] = ()
    notes: str = ""


@dataclass
class IngestResult:
    """Outcome of one ingest run.

    Attributes:
        session_id:    Session the run belonged to.
        accepted:      Number of images copied (or that would be, in dry-run).
        rejected:      (inbox filename, reason) pairs.
        manifest_path: Session manifest path (None in dry-run mode).
    """

    session_id: str
    accepted: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)
    manifest_path: Path | None = None


def init_captures_tree(root: Path) -> None:
    """Create the canonical (empty) capture tree under ``root``."""
    for sub in _SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    logger.info(f"Capture tree ready at {root}")


def is_eval_locked(root: Path) -> bool:
    """Return True when ``root`` carries a LOCKED.json (immutable eval set)."""
    return (root / LOCKED_FILENAME).exists()


def _content_digest(images_dir: Path) -> tuple[int, str]:
    """SHA-256 digest over the sorted per-file content hashes of a tree."""
    files = find_image_files(images_dir) if images_dir.exists() else []
    lines = [f"{f.name}:{compute_file_hash(f)}" for f in files]
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return len(files), digest


def lock_eval_set(eval_root: Path) -> Path:
    """Freeze an eval set: write LOCKED.json with a content digest.

    A locked set refuses further ingest; changing it means creating a new
    versioned eval root (immutability by construction).

    Args:
        eval_root: Eval set root directory.

    Returns:
        Path of the written LOCKED.json.

    Raises:
        ValueError: If the set is already locked or contains no images.
    """
    if is_eval_locked(eval_root):
        raise ValueError(f"{eval_root} is already locked — create a new eval version instead")

    count, digest = _content_digest(eval_root / "images")
    if count == 0:
        raise ValueError(f"Refusing to lock an empty eval set at {eval_root}")

    lock_path = eval_root / LOCKED_FILENAME
    lock_path.write_text(
        json.dumps(
            {"created_at": utc_now_iso(), "image_count": count, "content_digest": digest},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(f"Eval set locked: {count} images, digest {digest[:12]}… → {lock_path}")
    return lock_path


def ingest_session(
    inbox_dir: Path,
    meta: SessionMeta,
    config: CaptureConfig,
    dest_root: Path,
    license_str: str = DEFAULT_CAPTURE_LICENSE,
    dry_run: bool = False,
) -> IngestResult:
    """Validate and copy one session's inbox images into the capture tree.

    Re-running with the same session ID appends (sequence numbers continue)
    and byte/near-duplicates of already-ingested session images are rejected.

    Args:
        inbox_dir:   Staging directory holding the raw photos (never modified).
        meta:        Operator-supplied session metadata (already validated
                     against the capture config by the CLI).
        config:      Capture configuration.
        dest_root:   Capture tree root (captures_root or eval_root).
        license_str: License recorded in the manifests.
        dry_run:     Validate and report without writing anything.

    Returns:
        :class:`IngestResult` with accepted/rejected breakdown.

    Raises:
        FileNotFoundError: If ``inbox_dir`` does not exist.
        ValueError:        If ``dest_root`` is a locked eval set.
    """
    if is_eval_locked(dest_root):
        raise ValueError(f"{dest_root} is LOCKED — refusing to ingest into a frozen eval set")
    if not inbox_dir.is_dir():
        raise FileNotFoundError(f"Inbox directory not found: {inbox_dir}")

    result = IngestResult(session_id=meta.session_id)
    images_dir = dest_root / "images"
    manifests_dir = dest_root / "manifests"
    if not dry_run:
        init_captures_tree(dest_root)

    # Seed the duplicate index with this session's already-ingested images so
    # re-running the same inbox cannot double-ingest.
    dedup = DedupIndex(DedupSettings())
    existing = sorted(images_dir.glob(f"{meta.session_id}_*")) if images_dir.exists() else []
    for prior in existing:
        dedup.check_and_add(prior)
    seq = len(existing)

    max_bytes = config.image.max_file_mb * 1024 * 1024
    new_hashes: dict[str, str] = {}

    candidates = sorted(p for p in inbox_dir.iterdir() if p.is_file())
    for src in candidates:
        ext = src.suffix.lower()
        if ext not in config.image.allowed_extensions:
            result.rejected.append((src.name, f"disallowed_extension ({ext or 'none'})"))
            continue
        if src.stat().st_size > max_bytes:
            result.rejected.append((src.name, f"file_too_large (> {config.image.max_file_mb} MB)"))
            continue
        valid, message = validate_image(src)
        if not valid:
            result.rejected.append((src.name, f"corrupt_image ({message})"))
            continue
        dims = get_image_dimensions(src)
        if dims is None or min(dims) < config.image.min_dim:
            result.rejected.append((src.name, f"too_small ({dims} < {config.image.min_dim}px)"))
            continue
        duplicate_of = dedup.check_and_add(src)
        if duplicate_of is not None:
            result.rejected.append((src.name, f"duplicate_of ({duplicate_of.name})"))
            continue

        seq += 1
        dest = images_dir / f"{meta.session_id}_{seq:04d}{ext}"
        if dry_run:
            result.accepted += 1
            continue

        if config.image.strip_metadata:
            strip_metadata(src, dest)
            check = inspect_metadata(dest)
            if not check["clean"]:
                dest.unlink()
                seq -= 1
                result.rejected.append((src.name, "metadata_strip_failed"))
                continue
        else:
            shutil.copyfile(src, dest)
        new_hashes[dest.name] = compute_file_hash(dest)
        result.accepted += 1

    if dry_run:
        logger.info(
            f"[dry-run] {meta.session_id}: would accept {result.accepted}, "
            f"reject {len(result.rejected)}"
        )
        return result

    manifest_path = manifests_dir / f"{meta.session_id}.json"
    manifest = _updated_session_manifest(manifest_path, meta, license_str, new_hashes)
    manifest.image_count = len(sorted(images_dir.glob(f"{meta.session_id}_*")))
    manifest.save(manifest_path)
    result.manifest_path = manifest_path

    rebuild_aggregate_manifest(dest_root, license_str)
    logger.info(
        f"{meta.session_id}: accepted {result.accepted}, rejected {len(result.rejected)} "
        f"→ {images_dir}"
    )
    return result


def _updated_session_manifest(
    manifest_path: Path,
    meta: SessionMeta,
    license_str: str,
    new_hashes: dict[str, str],
) -> CaptureSessionManifest:
    """Load-or-create the session manifest and fold in this run's images."""
    if manifest_path.exists():
        manifest = CaptureSessionManifest.load(manifest_path)
        manifest.retrieved_at = utc_now_iso()
    else:
        manifest = CaptureSessionManifest(source="custom_captures")

    manifest.license = license_str
    manifest.url = "local capture session"
    manifest.session_id = meta.session_id
    manifest.house_id = meta.house_id
    manifest.room = meta.room
    manifest.capture_device = meta.capture_device
    manifest.lighting = meta.lighting
    manifest.captured_at = meta.captured_at
    manifest.consent_reference = meta.consent_reference
    manifest.trusted_classes = sorted(set(manifest.trusted_classes) | set(meta.trusted_classes))
    manifest.image_hashes.update(new_hashes)
    if meta.notes:
        manifest.notes = meta.notes
    return manifest


def rebuild_aggregate_manifest(
    root: Path,
    license_str: str = DEFAULT_CAPTURE_LICENSE,
    source_name: str = "custom_captures",
) -> SourceManifest:
    """Rebuild ``<root>/manifest.json`` from the per-session manifests.

    The aggregate is what the QA license gate picks up via its
    ``data/raw/*/manifest.json`` glob, and what the merge stage's source
    root presents as provenance.

    Args:
        root:        Capture tree root.
        license_str: License string for the aggregate.
        source_name: ``source`` field value (config key of the source).

    Returns:
        The saved aggregate :class:`SourceManifest`.
    """
    sessions = load_session_manifests(root)

    class_counts: dict[str, int] = {}
    image_hashes: dict[str, str] = {}
    trusted: set[str] = set()
    for session in sessions:
        for name, count in session.class_counts.items():
            class_counts[name] = class_counts.get(name, 0) + count
        image_hashes.update(session.image_hashes)
        trusted.update(session.trusted_classes)

    aggregate = SourceManifest(
        source=source_name,
        license=license_str,
        url="local capture sessions (see manifests/)",
        query={"sessions": sorted(s.session_id for s in sessions)},
        image_count=sum(s.image_count for s in sessions),
        class_counts=dict(sorted(class_counts.items())),
        trusted_classes=sorted(trusted),
        image_hashes=dict(sorted(image_hashes.items())),
        notes=f"aggregate of {len(sessions)} capture session(s)",
    )
    aggregate.save(root / MANIFEST_FILENAME)
    return aggregate


def load_session_manifests(root: Path) -> list[CaptureSessionManifest]:
    """Load every per-session manifest under ``<root>/manifests/``."""
    manifests_dir = root / "manifests"
    if not manifests_dir.exists():
        return []
    return [CaptureSessionManifest.load(path) for path in sorted(manifests_dir.glob("*.json"))]


def verify_captures_tree(root: Path, config: CaptureConfig) -> list[str]:
    """Re-validate a capture tree end-to-end (the ``--verify-all`` command).

    Read-only. Checks structure, session-ID grammar, manifest↔file hash
    agreement, orphan images/labels, metadata cleanliness (when stripping
    is configured), consent references, and the eval lock digest.

    Args:
        root:   Capture tree root (captures or eval).
        config: Capture configuration.

    Returns:
        List of problems; empty means the tree is verified.
    """
    if not root.exists():
        logger.info(f"{root} does not exist — nothing to verify")
        return []

    problems: list[str] = []
    images_dir = root / "images"
    labels_dir = root / "labels"
    for sub in ("images", "manifests"):
        if not (root / sub).is_dir():
            problems.append(f"missing directory: {root / sub}")
    if problems:
        return problems

    sessions = load_session_manifests(root)
    registry = load_consent_registry(config.consent.registry_path)

    manifest_names: set[str] = set()
    for session in sessions:
        sid = session.session_id
        for problem in config.validate_session_id(sid):
            problems.append(f"[{sid}] {problem}")
        for problem in verify_consent(
            session.consent_reference, session.house_id, config.consent, registry
        ):
            problems.append(f"[{sid}] {problem}")
        for name, expected in session.image_hashes.items():
            manifest_names.add(name)
            path = images_dir / name
            if not path.exists():
                problems.append(f"[{sid}] manifest lists missing image: {name}")
            elif compute_file_hash(path) != expected:
                problems.append(f"[{sid}] hash mismatch (tampered/corrupted): {name}")

    image_files = find_image_files(images_dir)
    known_prefixes = tuple(f"{s.session_id}_" for s in sessions)
    for image in image_files:
        if image.name not in manifest_names:
            problems.append(f"orphan image not in any session manifest: {image.name}")
        if known_prefixes and not image.name.startswith(known_prefixes):
            problems.append(f"image does not match any session prefix: {image.name}")

    if config.image.strip_metadata:
        problems.extend(_verify_metadata_clean(image_files))

    if labels_dir.is_dir():
        image_stems = {f.stem for f in image_files}
        for label in sorted(labels_dir.glob("*.txt")):
            if label.stem not in image_stems:
                problems.append(f"orphan label without image: {label.name}")

    lock_path = root / LOCKED_FILENAME
    if lock_path.exists():
        recorded = json.loads(lock_path.read_text(encoding="utf-8"))
        count, digest = _content_digest(images_dir)
        if count != recorded.get("image_count") or digest != recorded.get("content_digest"):
            problems.append(
                f"LOCKED digest mismatch: eval set changed after locking "
                f"({count} images vs recorded {recorded.get('image_count')})"
            )

    if problems:
        logger.error(f"verify: {len(problems)} problem(s) in {root}")
    else:
        logger.info(f"verify: {root} OK ({len(image_files)} images, {len(sessions)} sessions)")
    return problems


def _verify_metadata_clean(image_files: list[Path]) -> list[str]:
    """Check every image is metadata-free; degrade gracefully without PIL."""
    problems: list[str] = []
    for image in image_files:
        try:
            report = inspect_metadata(image)
        except RuntimeError:
            logger.warning("PIL unavailable — skipping metadata verification")
            return problems
        if not report["clean"]:
            problems.append(f"image carries metadata (EXIF/GPS/text): {image.name}")
    return problems
