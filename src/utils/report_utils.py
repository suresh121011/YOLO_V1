"""
src.utils.report_utils — CSV, JSON, and Markdown Report Generation
==================================================================

Shared report generation helpers used by the dataset statistics and QA
pipeline scripts. All functions write to disk and return the written path
for logging/chaining purposes.

Output formats:
    JSON     — Machine-readable, consumed by DVC metrics tracking
    CSV      — Spreadsheet-compatible flat table
    Markdown — Human-readable summary for PR review and documentation

Usage:
    from src.utils.report_utils import save_json_report, save_csv_report, save_markdown_report

    save_json_report(data_dict, Path("data/qa_reports/report.json"))
    save_csv_report(rows, headers, Path("data/qa_reports/report.csv"))
    save_markdown_report("QA Report", sections, Path("data/qa_reports/report.md"))
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─── Timestamp ────────────────────────────────────────────────────────────────


def timestamp_str() -> str:
    """Return current UTC time as an ISO 8601 string.

    Returns:
        e.g., "2026-07-13T05:30:00Z"
    """
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ─── JSON Reports ─────────────────────────────────────────────────────────────


def save_json_report(data: dict[str, Any], path: Path) -> Path:
    """Write a dict as a formatted JSON file.

    Args:
        data: Dict to serialize. Must be JSON-serializable.
        path: Output file path. Parent directories are created automatically.

    Returns:
        The written file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"JSON report written: {path}")
    return path


# ─── CSV Reports ──────────────────────────────────────────────────────────────


def save_csv_report(
    rows: list[dict[str, Any]],
    path: Path,
    fieldnames: list[str] | None = None,
) -> Path:
    """Write a list of dicts as a CSV file.

    Args:
        rows:       List of row dicts. All rows should have the same keys.
        path:       Output file path. Parent directories created automatically.
        fieldnames: Column order. If None, keys from the first row are used.

    Returns:
        The written file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        logger.info(f"CSV report written (empty): {path}")
        return path

    headers = fieldnames or list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"CSV report written: {path} ({len(rows)} rows)")
    return path


# ─── Markdown Reports ─────────────────────────────────────────────────────────


def save_markdown_report(
    title: str,
    sections: list[dict[str, Any]],
    path: Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a structured Markdown report.

    Args:
        title:    Report title (H1 heading).
        sections: List of section dicts. Each dict must have:
                    - 'heading': str — section heading (H2)
                    - 'content': str — pre-formatted Markdown content
                  Optionally:
                    - 'table': dict with 'headers' (list) and 'rows' (list of lists)
        path:     Output file path. Parent directories created automatically.
        metadata: Optional metadata shown in the report header (key-value pairs).

    Returns:
        The written file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        f"# {title}",
        "",
        f"*Generated: {timestamp_str()}*",
        "",
    ]

    if metadata:
        for key, val in metadata.items():
            lines.append(f"- **{key}:** {val}")
        lines.append("")

    for section in sections:
        heading = section.get("heading", "")
        content = section.get("content", "")
        table = section.get("table")

        if heading:
            lines.append(f"## {heading}")
            lines.append("")

        if content:
            lines.append(content)
            lines.append("")

        if table:
            lines.append(format_table(table["headers"], table["rows"]))
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Markdown report written: {path}")
    return path


# ─── Formatting Helpers ───────────────────────────────────────────────────────


def format_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Format data as a GitHub-Flavored Markdown table.

    Args:
        headers: Column header strings.
        rows:    List of rows, each row being a list of cell values.

    Returns:
        Multi-line Markdown table string.
    """
    if not headers:
        return ""

    # Convert all cells to strings
    str_rows = [[str(cell) for cell in row] for row in rows]

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    # Build table lines
    def pad(text: str, width: int) -> str:
        return text.ljust(width)

    header_line = "| " + " | ".join(pad(h, widths[i]) for i, h in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    data_lines = [
        "| " + " | ".join(pad(row[i] if i < len(row) else "", widths[i]) for i in range(len(headers))) + " |"
        for row in str_rows
    ]

    return "\n".join([header_line, separator] + data_lines)


def format_severity_badge(severity: str) -> str:
    """Return a Markdown text indicator for a severity level.

    Args:
        severity: One of "CRITICAL", "WARNING", "INFO", "PASS".

    Returns:
        Emoji + severity string (e.g., "🔴 CRITICAL").
    """
    badges = {
        "CRITICAL": "🔴 CRITICAL",
        "WARNING": "🟡 WARNING",
        "INFO": "🔵 INFO",
        "PASS": "✅ PASS",
    }
    return badges.get(severity.upper(), severity)


def format_count_pct(count: int, total: int) -> str:
    """Format a count with its percentage of total.

    Args:
        count: Numerator.
        total: Denominator.

    Returns:
        e.g., "42 (8.4%)" or "0 (0.0%)" if total is 0.
    """
    if total == 0:
        pct = 0.0
    else:
        pct = 100.0 * count / total
    return f"{count} ({pct:.1f}%)"


# ─── Summary Helpers ──────────────────────────────────────────────────────────


def write_all_formats(
    report_data: dict[str, Any],
    csv_rows: list[dict[str, Any]],
    md_title: str,
    md_sections: list[dict[str, Any]],
    output_dir: Path,
    base_name: str,
    csv_fieldnames: list[str] | None = None,
    md_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write a report in all three formats (JSON, CSV, Markdown) at once.

    Args:
        report_data:    Dict for JSON report.
        csv_rows:       List of dicts for CSV report.
        md_title:       Title for Markdown report.
        md_sections:    Section list for Markdown report.
        output_dir:     Directory to write all output files.
        base_name:      Base filename without extension (e.g., "qa_report").
        csv_fieldnames: Column order for CSV. None = auto-detect.
        md_metadata:    Optional metadata for Markdown header.

    Returns:
        Dict with keys "json", "csv", "markdown" mapping to written Paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "json": save_json_report(report_data, output_dir / f"{base_name}.json"),
        "csv": save_csv_report(csv_rows, output_dir / f"{base_name}.csv", csv_fieldnames),
        "markdown": save_markdown_report(md_title, md_sections, output_dir / f"{base_name}.md", md_metadata),
    }

    return paths
