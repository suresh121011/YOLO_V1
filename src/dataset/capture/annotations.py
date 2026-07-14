"""
src.dataset.capture.annotations — Annotation Export Import & Validation
=======================================================================

Imports YOLO-format annotation exports (CVAT "YOLO 1.1" primarily, but
any tool that emits ``*.txt`` YOLO labels plus a names file works) into
the capture tree, with session-scoped validation that the whole-dataset
QA cannot provide:

    1. CLASS-ORDER VERIFICATION — the export's names file must match
       configs/data.yaml ID-for-ID. A CVAT task created with a subset or
       reordered label list silently shifts every class ID; the IDs stay
       "valid" so nothing downstream can catch it. This is the one check
       that must never be skipped.
    2. Session coverage — every label must belong to an ingested image of
       the session, and ≥ ``min_labeled_fraction`` of the session's images
       must be labeled (catches wrong-export-for-session).
    3. Expectation check — a session whose ``trusted_classes`` declare a
       class should contain boxes of it (warning otherwise).
    4. Line-level format validation at import time (reusing
       src/utils/annotation_utils) so annotators get feedback per session,
       not weeks later at merge QA.

Flow: ``read_yolo_export`` → ``verify_class_order`` +
``validate_session_labels`` → ``stage_annotations`` (per annotator) →
[M4: dual-annotator compare] → ``finalize_annotations`` (labels land in
``<root>/labels/``, manifests updated).
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from src.dataset.capture.ingest import rebuild_aggregate_manifest
from src.dataset.manifest import (
    VALID_ANNOTATION_STATUSES,
    CaptureSessionManifest,
)
from src.utils.annotation_utils import (
    check_duplicate_lines,
    parse_yolo_line,
    validate_yolo_line,
)

logger = logging.getLogger(__name__)

#: Files in a CVAT YOLO 1.1 export that are metadata, not labels.
_NON_LABEL_FILES = frozenset({"train.txt", "obj.data", "test.txt", "valid.txt"})
#: Candidate names files, in preference order (CVAT, LabelImg, generic).
_NAMES_FILENAMES = ("obj.names", "classes.txt", "names.txt")


@dataclass
class YoloExport:
    """One parsed YOLO-format annotation export.

    Attributes:
        names:  Ordered class names from the export's names file.
        labels: Image stem → raw label lines (unparsed, stripped).
        origin: Where the export came from (for messages).
    """

    names: list[str]
    labels: dict[str, list[str]]
    origin: str = ""


@dataclass
class LabelValidation:
    """Outcome of session-scoped label validation.

    ``problems`` are blocking (format errors, orphans, under-coverage);
    ``warnings`` are advisory (expectation mismatches).
    """

    total_images: int
    labeled_images: int
    labeled_fraction: float
    class_counts: dict[str, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class FinalizeResult:
    """Outcome of finalizing one annotator's staged labels."""

    session_id: str
    annotator: str
    labels_written: int
    class_counts: dict[str, int] = field(default_factory=dict)


def read_yolo_export(export: Path) -> YoloExport:
    """Read a YOLO-format export from a zip archive or directory.

    Tolerant discovery: the names file is searched at any depth
    (obj.names / classes.txt / names.txt) and every other ``*.txt`` except
    known split-list/metadata files is treated as a label file keyed by
    its stem.

    Args:
        export: Path to a ``.zip`` export or an extracted directory.

    Returns:
        Parsed :class:`YoloExport`.

    Raises:
        FileNotFoundError: If ``export`` does not exist.
        ValueError:        If no names file or no label files are found.
    """
    if not export.exists():
        raise FileNotFoundError(f"Annotation export not found: {export}")

    if export.is_dir():
        names_text, label_texts = _read_from_dir(export)
    elif zipfile.is_zipfile(export):
        names_text, label_texts = _read_from_zip(export)
    else:
        raise ValueError(f"Export must be a directory or zip archive: {export}")

    if names_text is None:
        raise ValueError(f"No names file ({', '.join(_NAMES_FILENAMES)}) found in export {export}")
    names = [line.strip() for line in names_text.splitlines() if line.strip()]
    if not label_texts:
        raise ValueError(f"No YOLO label .txt files found in export {export}")

    labels = {
        stem: [line.strip() for line in text.splitlines() if line.strip()]
        for stem, text in sorted(label_texts.items())
    }
    logger.info(f"Export {export.name}: {len(names)} classes, {len(labels)} label files")
    return YoloExport(names=names, labels=labels, origin=str(export))


def _read_from_dir(root: Path) -> tuple[str | None, dict[str, str]]:
    """Collect names/label file contents from an extracted export directory."""
    names_text: str | None = None
    for candidate in _NAMES_FILENAMES:
        matches = sorted(root.rglob(candidate))
        if matches:
            names_text = matches[0].read_text(encoding="utf-8", errors="replace")
            break

    label_texts: dict[str, str] = {}
    for path in sorted(root.rglob("*.txt")):
        if path.name in _NON_LABEL_FILES or path.name in _NAMES_FILENAMES:
            continue
        label_texts[path.stem] = path.read_text(encoding="utf-8", errors="replace")
    return names_text, label_texts


def _read_from_zip(archive: Path) -> tuple[str | None, dict[str, str]]:
    """Collect names/label file contents from a zipped export."""
    names_text: str | None = None
    label_texts: dict[str, str] = {}
    with zipfile.ZipFile(archive) as zf:
        entries = [i for i in zf.infolist() if not i.is_dir()]
        for candidate in _NAMES_FILENAMES:
            match = next((i for i in entries if Path(i.filename).name == candidate), None)
            if match is not None:
                names_text = zf.read(match).decode("utf-8", errors="replace")
                break
        for info in entries:
            name = Path(info.filename).name
            if not name.endswith(".txt") or name in _NON_LABEL_FILES:
                continue
            if name in _NAMES_FILENAMES:
                continue
            label_texts[Path(info.filename).stem] = zf.read(info).decode("utf-8", errors="replace")
    return names_text, label_texts


def verify_class_order(names: list[str], class_names: dict[int, str]) -> list[str]:
    """Verify an export's class list matches the taxonomy ID-for-ID.

    Args:
        names:       Ordered names from the export.
        class_names: Taxonomy mapping (configs/data.yaml, id → name).

    Returns:
        List of problems; ANY entry means every class ID in the export is
        suspect and the import must be aborted (CRITICAL).
    """
    problems: list[str] = []
    if len(names) != len(class_names):
        problems.append(
            f"export defines {len(names)} classes, taxonomy has {len(class_names)} — "
            f"the CVAT task must use the FULL ordered label list from configs/data.yaml"
        )
    for idx in range(min(len(names), len(class_names))):
        expected = class_names[idx]
        if names[idx] != expected:
            problems.append(
                f"class ID {idx}: export says '{names[idx]}', taxonomy says '{expected}'"
            )
    return problems


def validate_session_labels(
    export: YoloExport,
    session_image_stems: set[str],
    class_names: dict[int, str],
    min_labeled_fraction: float,
    trusted_classes: tuple[str, ...] = (),
) -> LabelValidation:
    """Validate an export against one ingested session.

    Args:
        export:               Parsed export (class order already verified).
        session_image_stems:  Stems of the session's ingested images.
        class_names:          Taxonomy mapping (id → name).
        min_labeled_fraction: Required labeled share of session images.
        trusted_classes:      Classes the session declares it captures.

    Returns:
        :class:`LabelValidation` with blocking problems and warnings.
    """
    result = LabelValidation(
        total_images=len(session_image_stems),
        labeled_images=0,
        labeled_fraction=0.0,
    )
    num_classes = len(class_names)

    for stem, lines in export.labels.items():
        if stem not in session_image_stems:
            result.problems.append(
                f"label '{stem}' has no ingested image in this session "
                f"(wrong export or wrong --session?)"
            )
            continue
        result.labeled_images += 1

        for line_num, line in enumerate(lines, start=1):
            ann = parse_yolo_line(line, line_num)
            if ann is None:
                result.problems.append(f"{stem}.txt:{line_num}: malformed YOLO line '{line}'")
                continue
            for error in validate_yolo_line(ann, num_classes):
                result.problems.append(f"{stem}.txt:{line_num}: {error}")
            name = class_names.get(ann.class_id)
            if name is not None:
                result.class_counts[name] = result.class_counts.get(name, 0) + 1
        for first, second in check_duplicate_lines(lines):
            result.problems.append(f"{stem}.txt: duplicate lines {first} and {second}")

    if result.total_images:
        result.labeled_fraction = result.labeled_images / result.total_images
    if result.labeled_fraction < min_labeled_fraction:
        result.problems.append(
            f"only {result.labeled_images}/{result.total_images} session images labeled "
            f"({result.labeled_fraction:.0%}) — minimum is {min_labeled_fraction:.0%}"
        )

    for cls in trusted_classes:
        if result.class_counts.get(cls, 0) == 0:
            result.warnings.append(
                f"session declares trusted class '{cls}' but the export contains "
                f"no '{cls}' boxes"
            )

    result.class_counts = dict(sorted(result.class_counts.items()))
    return result


def stage_annotations(
    export: YoloExport,
    session_id: str,
    annotator: str,
    staging_dir: Path,
) -> Path:
    """Write an export's label files into the per-annotator staging area.

    Layout: ``<staging_dir>/<session_id>/<annotator>/<stem>.txt``.

    Returns:
        The annotator staging directory.
    """
    dest = staging_dir / session_id / annotator
    dest.mkdir(parents=True, exist_ok=True)
    for stem, lines in export.labels.items():
        (dest / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Staged {len(export.labels)} label files → {dest}")
    return dest


def staged_annotators(staging_dir: Path, session_id: str) -> list[str]:
    """List annotators with staged labels for a session."""
    session_dir = staging_dir / session_id
    if not session_dir.is_dir():
        return []
    return sorted(d.name for d in session_dir.iterdir() if d.is_dir())


def update_annotation_status(
    root: Path,
    session_id: str,
    status: str,
    annotator: str | None = None,
    iaa_agreement: float | None = None,
) -> CaptureSessionManifest:
    """Update a session manifest's annotation lifecycle fields.

    Args:
        root:          Capture tree root.
        session_id:    Session to update.
        status:        One of :data:`VALID_ANNOTATION_STATUSES`.
        annotator:     Annotator handle to record (appended if new).
        iaa_agreement: Measured dual-annotator agreement, if available.

    Returns:
        The saved manifest.

    Raises:
        ValueError:        If ``status`` is invalid.
        FileNotFoundError: If the session manifest does not exist
                           (session must be ingested first).
    """
    if status not in VALID_ANNOTATION_STATUSES:
        raise ValueError(f"Invalid annotation status '{status}' — {VALID_ANNOTATION_STATUSES}")
    manifest_path = root / "manifests" / f"{session_id}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest for session '{session_id}' at {manifest_path} — "
            f"ingest the session first (08_ingest_capture_session.py)"
        )
    manifest = CaptureSessionManifest.load(manifest_path)
    manifest.annotation_status = status
    if annotator and annotator not in manifest.annotators:
        manifest.annotators.append(annotator)
    if iaa_agreement is not None:
        manifest.iaa_agreement = iaa_agreement
    manifest.save(manifest_path)
    return manifest


def finalize_annotations(
    staging_dir: Path,
    session_id: str,
    annotator: str,
    root: Path,
    class_names: dict[int, str],
    license_str: str | None = None,
) -> FinalizeResult:
    """Promote one annotator's staged labels to the session's final labels.

    Copies ``<staging>/<session>/<annotator>/*.txt`` into ``<root>/labels/``,
    marks the session ``finalized``, records per-class box counts in the
    session manifest and rebuilds the aggregate manifest.

    Args:
        staging_dir: Annotation staging root.
        session_id:  Session to finalize.
        annotator:   Whose staged labels become final.
        root:        Capture tree root.
        class_names: Taxonomy mapping (id → name) for class counts.
        license_str: Aggregate license override (defaults to existing).

    Returns:
        :class:`FinalizeResult`.

    Raises:
        FileNotFoundError: If the staged labels or session manifest are missing.
    """
    source_dir = staging_dir / session_id / annotator
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"No staged labels for session '{session_id}' by '{annotator}' at {source_dir} — "
            f"run 09_import_annotations.py --stage first"
        )

    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    class_counts: dict[str, int] = {}
    written = 0
    for label_file in sorted(source_dir.glob("*.txt")):
        text = label_file.read_text(encoding="utf-8")
        (labels_dir / label_file.name).write_text(text, encoding="utf-8")
        written += 1
        for line_num, line in enumerate(text.splitlines(), start=1):
            ann = parse_yolo_line(line, line_num)
            if ann is None:
                continue
            name = class_names.get(ann.class_id)
            if name is not None:
                class_counts[name] = class_counts.get(name, 0) + 1

    manifest = update_annotation_status(root, session_id, "finalized", annotator=annotator)
    manifest.class_counts = dict(sorted(class_counts.items()))
    manifest.save(root / "manifests" / f"{session_id}.json")

    if license_str is not None:
        rebuild_aggregate_manifest(root, license_str)
    else:
        rebuild_aggregate_manifest(root)

    logger.info(
        f"Finalized {session_id} by {annotator}: {written} label files, "
        f"{sum(class_counts.values())} boxes"
    )
    return FinalizeResult(
        session_id=session_id,
        annotator=annotator,
        labels_written=written,
        class_counts=class_counts,
    )
