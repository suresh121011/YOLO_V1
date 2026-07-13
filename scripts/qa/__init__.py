"""
scripts.qa — Quality Assurance Pipeline Scripts
================================================

Script inventory:
    check_annotations.py        — Comprehensive annotation QA (15 checks, 3 severities)
                                  Covers: invalid class IDs, bbox validity, missing/duplicate
                                  files, corrupted images, and cross-split leakage detection.

QA outputs → data/qa_reports/
    annotation_qa_report.json   — Machine-readable full report (DVC metrics)
    annotation_qa_report.csv    — Flat issue list for spreadsheet review
    annotation_qa_report.md     — Human-readable Markdown summary
"""
