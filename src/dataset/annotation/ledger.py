"""
src.dataset.annotation.ledger — Verification Ledger Schema, IO & Read API
==========================================================================

The verification ledger is the single source of per-(image, class) human-
verified trust expansion (ADR-P5-04): every CVAT-verified cell — whether a
human confirmed boxes present or confirmed the class is truly absent —
becomes one entry here. ``auto_annotate`` (M1) reads it to skip already-
settled cells via :class:`LedgerView`, the canonical reader that
``scripts/dataset/12_auto_annotate.py``'s bootstrap ``read_verified_cells``
now delegates to (byte-compatible, per that function's M1 docstring). M3's
``TrustedListWithLedgerPolicy`` composes :class:`LedgerView` into the
completeness framework.

Conflict semantics (D4): a batch may only add or corroborate a verdict for
one (image, class); overriding an EXISTING different verdict requires an
explicit ``supersedes`` reference to the batch being overridden — anything
else hard-fails. A silent overwrite would hide a re-verification
disagreement a human must resolve, not software. Recording the identical
verdict again (any batch) is a no-op success — re-import must be idempotent.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.dataset.annotation.base import AnnotationError
from src.dataset.manifest import utc_now_iso

logger = logging.getLogger(__name__)

LEDGER_SCHEMA_VERSION = 1
DEFAULT_LEDGER_FILENAME = "verification_ledger.json"

#: The two verdict kinds a human can record for one (image, class) cell.
VALID_VERDICTS = ("present_labeled", "verified_absent")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``object_pairs_hook`` failing on duplicate JSON object keys."""
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise AnnotationError(
                f"Duplicate JSON key '{key}' in ledger — the file is corrupt or was "
                f"hand-edited; append-only edits must go through record_verdict()."
            )
        seen[key] = value
    return seen


def new_ledger(taxonomy_fp: str = "") -> dict[str, Any]:
    """An empty ledger, matching the M1 git-bootstrapped file's shape."""
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "taxonomy_fingerprint": taxonomy_fp,
        "updated_at": "",
        "entries": {},
        "stats": {"images": 0, "cells_verified": 0, "per_class": {}},
    }


def load_ledger(path: Path) -> dict[str, Any]:
    """Load a ledger, rejecting duplicate keys and unsupported schemas.

    A missing file is treated as an empty ledger (bootstrap convenience for
    tests/scratch runs) — production callers rely on the git-committed
    empty ledger always existing at ``configs/annotation.yaml``'s
    ``verification.ledger_path``.

    Raises:
        AnnotationError: On invalid JSON, duplicate keys, or an unsupported
                         schema version.
    """
    if not path.exists():
        return new_ledger()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as e:
        raise AnnotationError(f"Invalid ledger JSON in {path}: {e}") from e
    if not isinstance(raw, dict):
        raise AnnotationError(f"Ledger root must be a JSON object: {path}")
    version = raw.get("schema_version")
    if version != LEDGER_SCHEMA_VERSION:
        raise AnnotationError(
            f"Unsupported ledger schema_version {version!r} in {path} "
            f"(supported: {LEDGER_SCHEMA_VERSION})"
        )
    return raw


def save_ledger(ledger: Mapping[str, Any], path: Path) -> None:
    """Write the ledger as pretty-printed UTF-8 JSON (artifact convention).

    Args:
        ledger: Ledger dict (validate first — this does not re-validate).
        path:   Destination; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_ledger(
    ledger: Mapping[str, Any],
    class_names: Mapping[str, int] | None = None,
    expected_taxonomy_fp: str | None = None,
) -> list[str]:
    """Exhaustively validate a ledger. Returns problems (empty = valid).

    Args:
        ledger:               Ledger dict (from :func:`load_ledger`).
        class_names:          Taxonomy name -> id; when given, every
                              referenced class name must exist in it.
        expected_taxonomy_fp: Live taxonomy fingerprint; ``None`` skips the
                              drift check (an empty ledger has no fingerprint
                              recorded yet — that is not drift).
    """
    problems: list[str] = []
    for key in ("schema_version", "entries", "stats"):
        if key not in ledger:
            problems.append(f"missing required key '{key}'")
    if problems:
        return problems

    recorded_fp = ledger.get("taxonomy_fingerprint")
    if expected_taxonomy_fp is not None and recorded_fp not in ("", expected_taxonomy_fp):
        problems.append(
            f"taxonomy fingerprint drift: ledger {recorded_fp!r} vs live "
            f"{expected_taxonomy_fp!r} — reconcile before importing further verdicts"
        )

    entries = ledger["entries"]
    if not isinstance(entries, Mapping):
        return problems + ["'entries' must be an object"]

    for filename, entry in entries.items():
        if not isinstance(entry, Mapping):
            problems.append(f"entry '{filename}': must be an object")
            continue
        classes = entry.get("classes")
        if not isinstance(classes, Mapping) or not classes:
            problems.append(f"entry '{filename}': 'classes' must be a non-empty object")
            continue
        for class_name, verdict in classes.items():
            where = f"entry '{filename}'/'{class_name}'"
            if class_names is not None and class_name not in class_names:
                problems.append(f"{where}: class not in the taxonomy")
            if not isinstance(verdict, Mapping):
                problems.append(f"{where}: verdict must be an object")
                continue
            status = verdict.get("status")
            if status not in VALID_VERDICTS:
                problems.append(f"{where}: invalid status {status!r} (valid: {VALID_VERDICTS})")
            boxes = verdict.get("boxes", [])
            if not isinstance(boxes, list):
                problems.append(f"{where}: 'boxes' must be a list")
            elif status == "verified_absent" and boxes:
                problems.append(f"{where}: verified_absent but carries {len(boxes)} box(es)")
            elif status == "present_labeled" and not boxes:
                problems.append(f"{where}: present_labeled but carries zero boxes")

    return problems


def record_verdict(
    ledger: dict[str, Any],
    filename: str,
    source: str,
    class_name: str,
    status: str,
    boxes: list[tuple[float, float, float, float]],
    batch_id: str,
    verifier: str,
    method: str,
    cvat_task_ref: str,
    candidate_run: Mapping[str, Any] | None = None,
    supersedes: str | None = None,
) -> None:
    """Insert or update one (image, class) verdict in-place.

    Args:
        ledger:        Ledger dict to mutate (call :func:`save_ledger` after).
        filename:      Image filename (matches ``data/merged`` provenance).
        source:        Provenance source this image belongs to — checked
                       for consistency against any existing entry.
        class_name:    Taxonomy class name being verified.
        status:        ``"present_labeled"`` or ``"verified_absent"``.
        boxes:         Verified boxes (normalized xywhn); must be empty iff
                       ``status == "verified_absent"``.
        batch_id:      Verification batch this verdict came from.
        verifier:      Pseudonymous annotator/reviewer handle.
        method:        Free-text verification method (e.g. ``"cvat"``).
        cvat_task_ref: CVAT task id/URL.
        candidate_run: ``{backend, run_id, candidates_sha256}`` this batch's
                       pre-annotations came from (empty for pure L1 human
                       annotation with no auto-annotation candidates).
        supersedes:    Prior batch_id being intentionally overridden, if
                       this verdict conflicts with an existing one.

    Raises:
        AnnotationError: On an invalid status/boxes combination, a source
                         mismatch against an existing entry, or a
                         conflicting verdict without ``supersedes``.
    """
    if status not in VALID_VERDICTS:
        raise AnnotationError(f"Invalid verdict status '{status}' for {filename}/{class_name}")
    if status == "verified_absent" and boxes:
        raise AnnotationError(
            f"{filename}/{class_name}: verified_absent must carry zero boxes, got {len(boxes)}"
        )
    if status == "present_labeled" and not boxes:
        raise AnnotationError(f"{filename}/{class_name}: present_labeled must carry >=1 box")

    entries = ledger.setdefault("entries", {})
    entry = entries.get(filename)
    now = utc_now_iso()
    if entry is None:
        entry = {
            "source": source,
            "classes": {},
            "batch_id": batch_id,
            "verifier": verifier,
            "method": method,
            "cvat_task_ref": cvat_task_ref,
            "verified_at": now,
            "candidate_run": dict(candidate_run) if candidate_run else {},
            "supersedes": supersedes,
        }
        entries[filename] = entry
    elif entry.get("source") != source:
        raise AnnotationError(
            f"{filename}: ledger attributes it to source '{entry.get('source')}' but this "
            f"batch attributes it to '{source}' — provenance conflict, do not import."
        )

    classes = entry["classes"]
    new_verdict = {"status": status, "boxes": [list(b) for b in boxes]}
    existing = classes.get(class_name)
    if existing is not None and existing != new_verdict:
        if supersedes is None:
            raise AnnotationError(
                f"{filename}/{class_name}: conflicting verdict — existing {existing} vs "
                f"new {new_verdict}. Pass supersedes=<prior batch_id> if this is an "
                f"intentional re-verification; otherwise resolve the disagreement in CVAT "
                f"before importing."
            )
        logger.info(f"{filename}/{class_name}: verdict superseded ({supersedes} -> {batch_id})")
    elif existing is not None:
        return  # identical verdict re-imported — idempotent no-op, entry metadata untouched

    classes[class_name] = new_verdict
    entry["batch_id"] = batch_id
    entry["verifier"] = verifier
    entry["method"] = method
    entry["cvat_task_ref"] = cvat_task_ref
    entry["verified_at"] = now
    if candidate_run:
        entry["candidate_run"] = dict(candidate_run)
    if supersedes is not None:
        entry["supersedes"] = supersedes


def recompute_stats(ledger: dict[str, Any], taxonomy_fp: str) -> None:
    """Refresh ``stats``, ``taxonomy_fingerprint``, and ``updated_at`` in-place."""
    entries = ledger.get("entries", {})
    per_class: dict[str, int] = {}
    cells_verified = 0
    for entry in entries.values():
        for class_name in entry.get("classes", {}):
            per_class[class_name] = per_class.get(class_name, 0) + 1
            cells_verified += 1
    ledger["taxonomy_fingerprint"] = taxonomy_fp
    ledger["updated_at"] = utc_now_iso()
    ledger["stats"] = {
        "images": len(entries),
        "cells_verified": cells_verified,
        "per_class": dict(sorted(per_class.items())),
    }


@dataclass(frozen=True)
class LedgerView:
    """Read-only convenience view over a loaded ledger.

    The canonical reader for every downstream consumer (auto_annotate
    targeting, M3's completeness policy, coverage/quality reports).
    """

    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: Path) -> LedgerView:
        """Load a ledger file into a view (missing file = empty ledger)."""
        return cls(raw=load_ledger(path))

    def verified_cells(self, ids_by_name: Mapping[str, int]) -> dict[str, frozenset[int]]:
        """Filename -> verified class ids (both verdict kinds count as settled).

        Either verdict means a human settled the cell, so auto_annotate's
        targeting must never re-target it (M1's ``read_verified_cells``
        bootstrap behavior, preserved exactly here).

        Raises:
            AnnotationError: If an entry references a class name outside
                             the given taxonomy (drift; caller should not
                             proceed until reconciled).
        """
        cells: dict[str, frozenset[int]] = {}
        for filename, entry in self.raw.get("entries", {}).items():
            names = list((entry.get("classes") or {}).keys())
            unknown = sorted(n for n in names if n not in ids_by_name)
            if unknown:
                raise AnnotationError(
                    f"Ledger entry '{filename}' references classes not in the taxonomy: "
                    f"{unknown} — taxonomy drift; do not auto-annotate until reconciled."
                )
            if names:
                cells[filename] = frozenset(ids_by_name[n] for n in names)
        return cells

    def verified_class_names(self, filename: str) -> frozenset[str]:
        """Classes verified (either verdict) for one image."""
        entry = self.raw.get("entries", {}).get(filename)
        return frozenset((entry or {}).get("classes", {}).keys())

    def entry_source(self, filename: str) -> str | None:
        """The provenance source a ledger entry attributes an image to, if any."""
        entry = self.raw.get("entries", {}).get(filename)
        return entry.get("source") if entry else None

    def all_images(self) -> frozenset[str]:
        """Every filename with at least one verified cell."""
        return frozenset(self.raw.get("entries", {}).keys())

    def taxonomy_fingerprint(self) -> str:
        """Fingerprint recorded at last import (``""`` if never imported).

        An empty string is never taxonomy drift — the M1 git-bootstrapped
        ledger has no entries yet to have been imported against anything
        (M3's completeness drift check treats it as "not applicable").
        """
        return str(self.raw.get("taxonomy_fingerprint", ""))
