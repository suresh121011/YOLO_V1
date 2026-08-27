"""
src.scenario_engine.csv_view — Export-Only CSV View and Restricted Editable Subset
==================================================================================

CSV is a **view, not a format** (ADR-P6-02).

Lossless round-trip is not achievable and is not attempted. ``optional_objects:
[]`` and ``optional_objects: null`` are both the empty string in CSV, and that
loss is irrecoverable. Excel on Windows opens UTF-8-without-BOM as cp1252, so a
stakeholder who opens the file, changes one risk level and saves has turned
every Devanagari prompt into mojibake — and the mojibake is then the data. This
repository already carries a recorded cp1252 incident and has no stream
hardening anywhere.

So there are two files with different jobs:

``scenarios.view.csv``
    Full read-only projection. UTF-8, LF, ``QUOTE_ALL``, sorted by scenario id,
    explicit column order. Byte-compared in CI, which is what catches a CRLF
    regression on the Windows leg.

``scenarios.editable.csv``
    Deliberately restricted to flat scalars a non-engineer can safely change.
    **Prompt fields are excluded** — that is what keeps Devanagari out of
    Excel's reach. Written with a BOM so Excel decodes it correctly. The
    importer matches on ``scenario_id`` only, cannot create or delete rows, and
    hard-fails on any change outside the allowlist.

Neither file is ever an input to the compiler: it refuses to read any ``.csv``
under ``build/``.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: Full projection, in a fixed order. Includes the brief's requested columns.
VIEW_COLUMNS: tuple[str, ...] = (
    "scenario_id",
    "scenario_name",
    "category",
    "status",
    "detectability",
    "claim_class",
    "risk_level",
    "priority",
    "condition",
    "required_objects",
    "optional_objects",
    "room_context",
    "min_confidence",
    "min_dwell_seconds",
    "clear_after_seconds",
    "cooldown_seconds",
    "max_repeats",
    "max_per_day",
    "patient_facing",
    "next_best_action",
    "caregiver_channel",
    "evidence_sources",
    "evidence_strength",
    "capability_disclaimer",
    "rejection_reason",
    "reviewed_by",
    "reviewed_on",
    "rule_hash",
)

#: The only columns a stakeholder may edit. Prompts are deliberately absent.
EDITABLE_COLUMNS: tuple[str, ...] = (
    "scenario_id",
    "scenario_name",
    "category",
    "risk_level",
    "next_best_action",
    "caregiver_channel",
    "status",
)

#: Separator for list-valued cells. Declared here because CSV has no way to say
#: so in-band; a stakeholder typing ", " instead would break it silently, which
#: is one reason list columns are read-only.
LIST_SEPARATOR = "; "


class CsvViewError(ValueError):
    """Raised when a CSV cannot be produced or an edit is not permitted."""


def _flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | tuple):
        return LIST_SEPARATOR.join(str(item) for item in value)
    return str(value)


def _row_from_scenario(scenario: Mapping[str, Any]) -> dict[str, str]:
    evidence = scenario.get("evidence", []) or []
    row = {
        **{key: _flatten(scenario.get(key)) for key in VIEW_COLUMNS},
        "evidence_sources": LIST_SEPARATOR.join(
            str(item.get("source_id", "")) for item in evidence
        ),
        "evidence_strength": LIST_SEPARATOR.join(
            str(item.get("strength", "")) for item in evidence
        ),
    }
    for column, cell in row.items():
        if "\n" in cell or "\r" in cell:
            raise CsvViewError(
                f"{scenario.get('scenario_id')}: field {column!r} contains a newline. "
                f"Embedded newlines are rejected rather than escaped, because a quoted "
                f"multi-line cell renders as broken rows in `git diff` — destroying the "
                f"diff reviewability that justified a text format in the first place."
            )
    return row


def artifact_rows(artifact: Mapping[str, Any]) -> list[dict[str, str]]:
    """Every scenario in the artifact — runnable, deprecated and rejected."""
    scenarios: list[Mapping[str, Any]] = [
        *artifact.get("scenarios", []),
        *artifact.get("deprecated", []),
        *artifact.get("rejected", []),
    ]
    rows = [_row_from_scenario(scenario) for scenario in scenarios]
    return sorted(rows, key=lambda row: row["scenario_id"])


def _write_csv(
    rows: Sequence[Mapping[str, str]],
    columns: Sequence[str],
    path: Path,
    encoding: str,
) -> Path:
    """Write with the byte spec ADR-P6-02 fixes: LF, QUOTE_ALL, explicit columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(columns),
            extrasaction="ignore",
            lineterminator="\n",
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_view_csv(artifact: Mapping[str, Any], path: Path | str) -> Path:
    """Write the full read-only projection.

    Named ``.view.csv`` deliberately: CSV has no comment syntax in which to say
    "do not edit me", so the filename carries it.
    """
    target = Path(path)
    if not target.name.endswith(".view.csv"):
        raise CsvViewError(
            f"The read-only projection must be named '*.view.csv', got {target.name!r}."
        )
    return _write_csv(artifact_rows(artifact), VIEW_COLUMNS, target, encoding="utf-8")


def write_editable_csv(artifact: Mapping[str, Any], path: Path | str) -> Path:
    """Write the restricted stakeholder-editable subset.

    ``utf-8-sig`` so Excel decodes it correctly. Safe here precisely because the
    prompt columns are excluded, so there is no Devanagari to corrupt.
    """
    rows = [{column: row[column] for column in EDITABLE_COLUMNS} for row in artifact_rows(artifact)]
    return _write_csv(rows, EDITABLE_COLUMNS, Path(path), encoding="utf-8-sig")


def read_editable_csv(path: Path | str) -> dict[str, dict[str, str]]:
    """Read the editable subset back, keyed by scenario id.

    Raises:
        CsvViewError: If a column outside the allowlist is present, or a row has
            no scenario id.
    """
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        unexpected = sorted(set(columns) - set(EDITABLE_COLUMNS))
        if unexpected:
            raise CsvViewError(
                f"{path}: column(s) {unexpected} are not editable. Only "
                f"{list(EDITABLE_COLUMNS)} may be changed via CSV; everything else "
                f"is edited in the scenario YAML."
            )
        rows: dict[str, dict[str, str]] = {}
        for index, row in enumerate(reader):
            scenario_id = (row.get("scenario_id") or "").strip()
            if not scenario_id:
                raise CsvViewError(f"{path}: row {index} has no scenario_id")
            rows[scenario_id] = {k: (v or "").strip() for k, v in row.items()}
    return rows


def diff_editable(
    artifact: Mapping[str, Any],
    edited: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, tuple[str, str]]]:
    """Compare an edited subset against the artifact.

    Returns:
        ``{scenario_id: {column: (before, after)}}`` for changed cells only.

    Raises:
        CsvViewError: If the edit adds or removes a scenario. Creating a
            scenario requires assigning an immutable id and writing a trigger,
            which is a review decision, not a spreadsheet one.
    """
    current = {row["scenario_id"]: row for row in artifact_rows(artifact)}

    added = sorted(set(edited) - set(current))
    removed = sorted(set(current) - set(edited))
    if added:
        raise CsvViewError(
            f"{added} are not existing scenarios. The editable CSV cannot create "
            f"scenarios — ids are immutable and a new row needs a trigger (ADR-P6-02)."
        )
    if removed:
        raise CsvViewError(
            f"{removed} are missing from the edited CSV. The editable CSV cannot delete "
            f"scenarios; deprecate them in the YAML instead so the id stays reserved."
        )

    changes: dict[str, dict[str, tuple[str, str]]] = {}
    for scenario_id, row in edited.items():
        before = current[scenario_id]
        changed = {
            column: (before[column], row[column])
            for column in EDITABLE_COLUMNS
            if column != "scenario_id" and before[column] != row[column]
        }
        if changed:
            changes[scenario_id] = changed
    return changes
