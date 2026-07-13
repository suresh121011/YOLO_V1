"""
src.dataset.downloaders — Per-Source Acquisition
================================================

Bespoke, requests-based downloaders behind a common interface
(:class:`~src.dataset.downloaders.base.BaseDownloader`). Design rules:

    - Annotations/index first, then per-image fetch — smoke mode is just a
      cap parameter, never a special code path.
    - Every downloader writes ``manifest.json`` (provenance) and
      ``source_classes.json`` (local id → source class name) next to its
      ``images/`` and ``labels/`` output.
    - Downloads are resumable: existing non-empty files are skipped.
    - Missing credentials (e.g. ROBOFLOW_API_KEY) cause a graceful,
      clearly-logged skip — never a crash.
"""

from src.dataset.downloaders.base import BaseDownloader, DownloadSkippedError

__all__ = ["BaseDownloader", "DownloadSkippedError"]
