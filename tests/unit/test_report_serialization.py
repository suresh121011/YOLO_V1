"""Line-ending determinism for every artifact that DVC hashes.

Why this exists
---------------
DVC hashes the exact bytes on disk. Python's default text mode writes CRLF on
Windows and LF on Linux, so the *same* report acquires two different md5s
depending on which machine produced it. Git compounds it: ``.gitattributes``
declares ``* text=auto eol=lf``, so for the five artifacts that are tracked by
git *and* DVC, every ``git checkout`` rewrote CRLF→LF and silently invalidated
the hash recorded in ``dvc.lock``.

That was not hypothetical — it was measured. ``dvc.lock`` recorded
``d5393708…`` for ``annotation_qa_report.json``, which is exactly the md5 of the
on-disk file with LF re-expanded to CRLF, while the file itself hashed
``4cc0a202…``. The pipeline therefore never reached a clean ``dvc status`` and
``dvc repro`` re-ran those stages forever.

The fix is to write LF explicitly at the source. Git's ``eol=lf`` then has
nothing to convert, so git, DVC and the workspace agree on every platform.

These tests are the regression guard. :class:`TestWritersEmitLf` proves the
behaviour; :class:`TestNoTextModeWritesRemain` is a static sweep that fails when
a *new* writer is added to one of these modules without ``newline=``, which is
how the defect would otherwise creep back.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from src.dataset.annotation.candidates import save_candidates
from src.dataset.annotation.ledger import save_ledger
from src.dataset.completeness import save_completeness
from src.dataset.manifest import SourceManifest
from src.utils.report_utils import save_json_report

pytestmark = pytest.mark.unit

CRLF = b"\r\n"

REPO_ROOT = Path(__file__).resolve().parents[2]

# Modules whose writers feed a DVC output. A text-mode write here corrupts a
# tracked hash, so every write call in them must pin ``newline``.
DVC_OUTPUT_WRITER_MODULES = (
    "src/dataset/annotation/apply.py",
    "src/dataset/annotation/candidates.py",
    "src/dataset/annotation/ledger.py",
    "src/dataset/annotation/verified_import.py",
    "scripts/training/train_yolo.py",
    "src/dataset/completeness.py",
    "src/dataset/manifest.py",
    "src/dataset/merge.py",
    "src/dataset/cross_dataset_salvage.py",
    "src/dataset/remap.py",
    "src/dataset/capture/ingest.py",
    "src/dataset/downloaders/base.py",
    "src/utils/report_utils.py",
    "scripts/qa/run_full_qa.py",
    "scripts/dataset/13_build_verification_batches.py",
    "scripts/dataset/20_ingest_local_zips.py",
)

# A payload with enough structure to span multiple lines — a single-line JSON
# blob would pass even a broken writer, since there is no newline to translate.
_NESTED: dict[str, Any] = {
    "schema_version": 1,
    "rows": [{"id": i, "name": f"item-{i}", "tags": ["a", "b"]} for i in range(5)],
    "nested": {"deep": {"deeper": [1, 2, 3]}},
}


def _assert_lf_only(path: Path) -> None:
    raw = path.read_bytes()
    assert raw, f"{path} is empty — writer produced nothing"
    assert CRLF not in raw, (
        f"{path.name} contains CRLF. DVC hashes raw bytes, so this makes the "
        f"artifact's hash platform-dependent and breaks `dvc status`. "
        f'Pass newline="\\n" to the write call.'
    )
    assert b"\n" in raw, f"{path.name} has no newlines — payload was not multi-line"


class TestWritersEmitLf:
    """Every artifact writer must produce LF bytes, on any OS."""

    def test_save_json_report(self, tmp_path: Path) -> None:
        # Shared helper behind write_all_formats → coverage_report.json,
        # completeness_report.json, dataset_quality_report.json, eval_report.json.
        save_json_report(_NESTED, tmp_path / "report.json")
        _assert_lf_only(tmp_path / "report.json")

    def test_save_ledger(self, tmp_path: Path) -> None:
        # ADR-P5-01 source of truth. Currently empty in-repo, which is the only
        # reason it has not broken yet — an empty ledger has no interior newlines.
        ledger = {
            "schema_version": 1,
            "taxonomy_fingerprint": "abc123",
            "entries": {f"img_{i}.jpg": {"verdict": "accept"} for i in range(4)},
        }
        save_ledger(ledger, tmp_path / "verification_ledger.json")
        _assert_lf_only(tmp_path / "verification_ledger.json")

    def test_save_candidates(self, tmp_path: Path) -> None:
        save_candidates(_NESTED, tmp_path / "candidates.json")
        _assert_lf_only(tmp_path / "candidates.json")

    def test_save_completeness(self, tmp_path: Path) -> None:
        save_completeness(dict(_NESTED), tmp_path / "completeness.json")
        _assert_lf_only(tmp_path / "completeness.json")

    def test_source_manifest_save(self, tmp_path: Path) -> None:
        manifest = SourceManifest(
            source="unit-test",
            license="CC BY 4.0",
            image_count=3,
            class_counts={"person": 4, "chair": 3, "bed": 2},
            trusted_classes=["person", "chair", "bed"],
        )
        manifest.save(tmp_path / "manifest.json")
        _assert_lf_only(tmp_path / "manifest.json")

    def test_write_yolo_label(self, tmp_path: Path) -> None:
        """Labels are the highest-volume artifact — 17,422 files per build.

        Every one of them lives inside a DVC output (``data/raw/*``,
        ``data/interim``, ``data/merged``, ``data/processed/labels``), so a
        CRLF label file makes the whole dataset's hash Windows-specific.
        """
        from src.dataset.downloaders.base import write_yolo_label

        dest = tmp_path / "img.txt"
        write_yolo_label(dest, [(0, 0.5, 0.5, 0.2, 0.2), (3, 0.1, 0.1, 0.05, 0.05)])
        _assert_lf_only(dest)
        # Content must survive untouched — 2 boxes, 5 fields each.
        rows = dest.read_text(encoding="utf-8").strip().splitlines()
        assert len(rows) == 2
        assert all(len(r.split()) == 5 for r in rows)

    def test_verified_labels_overlay(self, tmp_path: Path) -> None:
        """The overlay is `split.source_labels_dir` — every training label passes here.

        A CRLF write in this stage propagates into `data/merged_verified` and
        then `data/processed`, which is how 17,402 CRLF labels reached the split
        even after the merge itself had been fixed.
        """
        from src.dataset.annotation.apply import build_verified_labels_overlay

        merged = tmp_path / "merged"
        verified = tmp_path / "verified"
        out = tmp_path / "out"
        for d in (merged, verified, out):
            d.mkdir()
        # multi-line label: single-line content has no newline to translate
        (merged / "img.txt").write_text(
            "3 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.05 0.05\n", encoding="utf-8", newline="\n"
        )
        build_verified_labels_overlay(
            merged_labels_dir=merged, verified_labels_dir=verified, output_dir=out
        )
        _assert_lf_only(out / "img.txt")

    @pytest.mark.parametrize(
        "writer",
        [save_json_report, save_candidates, save_completeness],
        ids=["json_report", "candidates", "completeness"],
    )
    def test_round_trip_is_lossless(self, writer: Callable[..., Any], tmp_path: Path) -> None:
        """Pinning newline must not alter content — only bytes."""
        out = tmp_path / "rt.json"
        writer(dict(_NESTED), out)
        assert json.loads(out.read_text(encoding="utf-8")) == _NESTED


class TestNoTextModeWritesRemain:
    """Static sweep: a new writer without ``newline=`` fails the build."""

    @staticmethod
    def _offenders(source: str) -> list[str]:
        """Return `write_text(...)` / `open(..., "w")` calls missing ``newline``."""
        found: list[str] = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            kwargs = {kw.arg for kw in node.keywords}

            is_write_text = isinstance(node.func, ast.Attribute) and node.func.attr == "write_text"
            # Any text-mode write: "w", "a", "wt", "at", "w+", "a+", "r+" …
            # Append mode matters as much as write — a salvage path appending one
            # CRLF line to an otherwise-LF label file is what slipped through the
            # first version of this sweep and put 170 CRLF files into data/merged.
            # Binary modes ("wb", "ab") translate nothing and are exempt.
            is_open_text_write = (
                isinstance(node.func, ast.Name)
                and node.func.id == "open"
                and any(
                    isinstance(a, ast.Constant)
                    and isinstance(a.value, str)
                    and "b" not in a.value
                    and any(m in a.value for m in ("w", "a", "+"))
                    for a in node.args[1:2]
                )
            )
            if not (is_write_text or is_open_text_write):
                continue

            # write_text("") for a deliberately empty file has no newline to
            # translate, so it is exempt.
            if is_write_text and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and first.value == "":
                    continue

            if "newline" not in kwargs:
                call = "write_text" if is_write_text else "open(..., text-write mode)"
                found.append(f"line {node.lineno}: {call}")
        return found

    @pytest.mark.parametrize("rel_path", DVC_OUTPUT_WRITER_MODULES)
    def test_module_pins_newline(self, rel_path: str) -> None:
        path = REPO_ROOT / rel_path
        assert path.exists(), f"{rel_path} moved — update DVC_OUTPUT_WRITER_MODULES"
        offenders = self._offenders(path.read_text(encoding="utf-8"))
        assert not offenders, (
            f"{rel_path} writes a DVC-tracked artifact in text mode:\n  "
            + "\n  ".join(offenders)
            + '\nAdd newline="\\n" — otherwise the artifact hashes differently on '
            "Windows and Linux and `dvc status` can never be clean."
        )
