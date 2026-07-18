"""
src.dataset.release.gates — Release Gates RG1-RG10 (ADR-P5-07)
==================================================================

Releases are gated, not vibes: ``configs/release.yaml`` declares which gate
ids each version track (dataset-v0.5.0 .. dataset-v1.0.0) must pass, and
this module implements the checks. Every ``rgN_*`` function is a pure
function over already-loaded data (dicts, strings, primitives) — no file
I/O of its own — so each gate has a trivial, dependency-free unit test with
a one-line negative case. :func:`evaluate_release` is the thin orchestrator
that loads the real artifacts and calls the gates the requested track
actually requires.

A gate never RECOMPUTES what an earlier stage already established: RG1/RG8
read straight from ``annotation_qa_report.json`` (checks, eval_set,
annotation_sweeps, l4_l5_reports), RG3 reads ``coverage_report.json`` /
``dataset_quality_report.json``. Reuses the same ``GateResult``/report
shape as ``src/training/preflight.py`` (deliberately NOT imported from
there — ``src/dataset`` must never depend on ``src/training``, the reverse
of the real dependency direction; the shape is duplicated, not the module).

A ``MODE`` prerequisite check (not one of RG1-RG10) always runs first: a
release track declares the acquisition ``mode`` (smoke|full) it requires,
and checking a higher track against a lower-mode build is a hard,
unambiguous failure — the plan's "check dataset-v0.5.0 on smoke build
correctly FAILS" acceptance case.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.dataset.capture.ingest import load_session_manifests
from src.dataset.completeness import load_completeness, validate_completeness
from src.dataset.manifest import MANIFEST_FILENAME
from src.dataset.sources_config import SourcesConfig, load_sources_config
from src.utils.config_helpers import load_yaml
from src.utils.dataset_utils import compute_file_hash

GATE_STATUS_PASS = "pass"  # noqa: S105 — gate status literal, not a credential
GATE_STATUS_WARN = "warn"
GATE_STATUS_FAIL = "fail"
GATE_STATUS_SKIPPED = "skipped"

ALL_GATE_IDS: tuple[str, ...] = tuple(f"RG{i}" for i in range(1, 11))


@dataclass(frozen=True)
class GateResult:
    """Outcome of one release gate."""

    gate_id: str
    name: str
    status: str
    details: str

    def format_line(self) -> str:
        """One-line human-readable rendering."""
        badge = {
            GATE_STATUS_PASS: "PASS",
            GATE_STATUS_WARN: "WARN",
            GATE_STATUS_FAIL: "FAIL",
            GATE_STATUS_SKIPPED: "SKIP",
        }[self.status]
        return f"[{badge}] {self.gate_id} {self.name}: {self.details}"


@dataclass(frozen=True)
class ReleaseReport:
    """Aggregated release-gate outcome for one version track."""

    version: str
    required_gate_ids: tuple[str, ...]
    results: tuple[GateResult, ...]

    @property
    def verdict(self) -> str:
        """PASS, WARN, or FAIL — computed only over required gates."""
        required = {
            r.status
            for r in self.results
            if r.gate_id in self.required_gate_ids or r.gate_id == "MODE"
        }
        if GATE_STATUS_FAIL in required:
            return "FAIL"
        if GATE_STATUS_WARN in required:
            return "WARN"
        return "PASS"

    def failures(self) -> list[GateResult]:
        """Required gates (+MODE) that failed."""
        return [
            r
            for r in self.results
            if r.status == GATE_STATUS_FAIL
            and (r.gate_id in self.required_gate_ids or r.gate_id == "MODE")
        ]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return {
            "version": self.version,
            "verdict": self.verdict,
            "required_gate_ids": list(self.required_gate_ids),
            "gates": [
                {"gate_id": r.gate_id, "name": r.name, "status": r.status, "details": r.details}
                for r in self.results
            ],
        }

    def format_lines(self) -> list[str]:
        """Human-readable line per gate plus a verdict line."""
        return [r.format_line() for r in self.results] + [
            f"Release check ({self.version}) verdict: {self.verdict}"
        ]


# ─── MODE prerequisite ─────────────────────────────────────────────────────────


def check_build_mode(actual_mode: str, required_mode: str) -> GateResult:
    """Acquisition mode (configs/dataset_sources.yaml) must match the track."""
    if actual_mode != required_mode:
        return GateResult(
            "MODE",
            "build-mode",
            GATE_STATUS_FAIL,
            f"track requires mode '{required_mode}' but the current build is "
            f"'{actual_mode}' — flip configs/dataset_sources.yaml mode and re-run "
            f"`dvc repro` before checking this release.",
        )
    return GateResult("MODE", "build-mode", GATE_STATUS_PASS, f"build mode '{actual_mode}' matches")


# ─── RG1: QA zero-criticals + artifact sweeps clean ───────────────────────────


def rg1_qa_check(qa_report: Mapping[str, Any] | None) -> GateResult:
    """QA report has zero criticals and zero artifact-sweep findings."""
    if qa_report is None:
        return GateResult(
            "RG1", "qa-check", GATE_STATUS_FAIL, "annotation_qa_report.json not found"
        )

    summary = qa_report.get("summary", {})
    orchestrator = qa_report.get("orchestrator", {})
    critical = (
        int(summary.get("critical_issues", 0)) > 0
        or bool(orchestrator.get("license_critical"))
        or bool(orchestrator.get("eval_overlap_critical"))
    )
    sweep_warnings = int(orchestrator.get("annotation_sweep_warnings", 0)) + int(
        orchestrator.get("l4_l5_report_warnings", 0)
    )
    if critical:
        return GateResult(
            "RG1", "qa-check", GATE_STATUS_FAIL, "QA report has critical issue(s) — see qa_check"
        )
    if sweep_warnings > 0:
        return GateResult(
            "RG1",
            "qa-check",
            GATE_STATUS_FAIL,
            f"{sweep_warnings} artifact-sweep finding(s) must be resolved before release",
        )
    return GateResult("RG1", "qa-check", GATE_STATUS_PASS, "0 criticals, artifact sweeps clean")


# ─── RG2: completeness valid + input hashes fresh ─────────────────────────────


def rg2_completeness_freshness(
    validation_errors: list[str], freshness_errors: list[str]
) -> GateResult:
    """Completeness artifact self-consistency + input-hash freshness.

    Args:
        validation_errors: Output of ``src.dataset.completeness.validate_completeness``.
        freshness_errors:  Recorded input hashes vs. what's on disk right now.
    """
    problems = list(validation_errors) + list(freshness_errors)
    if problems:
        preview = "; ".join(problems[:5])
        return GateResult(
            "RG2",
            "completeness-freshness",
            GATE_STATUS_FAIL,
            f"{len(problems)} problem(s): {preview}",
        )
    return GateResult("RG2", "completeness-freshness", GATE_STATUS_PASS, "valid and fresh")


# ─── RG3: coverage + quality thresholds ────────────────────────────────────────


def rg3_coverage_quality(
    coverage_report: Mapping[str, Any] | None,
    quality_report: Mapping[str, Any] | None,
    min_verified_cells: int,
    min_coverage_score: Mapping[str, float],
) -> GateResult:
    """Coverage/quality reports present; verified-cell + per-class thresholds met."""
    if coverage_report is None or quality_report is None:
        missing = [
            name
            for name, r in (
                ("coverage_report", coverage_report),
                ("quality_report", quality_report),
            )
            if r is None
        ]
        return GateResult(
            "RG3", "coverage-quality", GATE_STATUS_FAIL, f"missing report(s): {', '.join(missing)}"
        )

    problems: list[str] = []
    cells_verified = int(
        (quality_report.get("verification_progress") or {})
        .get("ledger_stats", {})
        .get("cells_verified", 0)
    )
    if cells_verified < min_verified_cells:
        problems.append(f"verified cells {cells_verified} < required {min_verified_cells}")

    per_class = coverage_report.get("per_class") or {}
    for class_name, threshold in sorted(min_coverage_score.items()):
        score = (per_class.get(class_name) or {}).get("coverage_score")
        if score is None:
            problems.append(f"'{class_name}': no coverage_score recorded")
        elif float(score) < threshold:
            problems.append(f"'{class_name}': coverage_score {score} < required {threshold}")

    if problems:
        return GateResult("RG3", "coverage-quality", GATE_STATUS_FAIL, "; ".join(problems))
    return GateResult(
        "RG3",
        "coverage-quality",
        GATE_STATUS_PASS,
        f"{cells_verified} verified cells, {len(min_coverage_score)} per-class threshold(s) met",
    )


# ─── RG4: changelog entry exists ───────────────────────────────────────────────


def rg4_changelog_entry(changelog_text: str, version: str) -> GateResult:
    """``data/DATASET_CHANGELOG.md`` has a heading for this version."""
    heading_prefix = f"## {version}"
    if any(line.strip().startswith(heading_prefix) for line in changelog_text.splitlines()):
        return GateResult("RG4", "changelog-entry", GATE_STATUS_PASS, f"entry found for {version}")
    return GateResult(
        "RG4",
        "changelog-entry",
        GATE_STATUS_FAIL,
        f"no '{heading_prefix}' heading in data/DATASET_CHANGELOG.md — add one before release",
    )


# ─── RG5: working tree clean, HEAD tagged ─────────────────────────────────────


def rg5_working_tree_tagged(
    porcelain_status: str, tags_at_head: list[str], version: str
) -> GateResult:
    """Git working tree clean and HEAD carries the version's tag."""
    problems: list[str] = []
    if porcelain_status.strip():
        problems.append("working tree is not clean (uncommitted changes)")
    if version not in tags_at_head:
        problems.append(f"HEAD is not tagged '{version}' (tags at HEAD: {tags_at_head or 'none'})")
    if problems:
        return GateResult("RG5", "git-tagged", GATE_STATUS_FAIL, "; ".join(problems))
    return GateResult("RG5", "git-tagged", GATE_STATUS_PASS, f"clean tree, tagged '{version}'")


def git_porcelain_status(repo_root: str = ".") -> str:
    """Real ``git status --porcelain`` output (empty string == clean)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
            cwd=repo_root,
        )
        return out.stdout
    except (OSError, subprocess.SubprocessError):
        return "<git status unavailable>"


def git_tags_at_head(repo_root: str = ".") -> list[str]:
    """Real tag names pointing at HEAD."""
    try:
        out = subprocess.run(
            ["git", "tag", "--points-at", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
            cwd=repo_root,
        )
        return [line for line in out.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


# ─── RG6: dvc push executed + dvc status -c clean ─────────────────────────────


def rg6_dvc_push_verified(dvc_status_cache_output: str) -> GateResult:
    """``dvc status -c --quiet`` prints nothing when the cache/remote are in sync."""
    if dvc_status_cache_output.strip():
        return GateResult(
            "RG6",
            "dvc-push-verified",
            GATE_STATUS_FAIL,
            f"`dvc status -c` is not clean — run `dvc push`: "
            f"{dvc_status_cache_output.strip()[:200]}",
        )
    return GateResult("RG6", "dvc-push-verified", GATE_STATUS_PASS, "cache and remote in sync")


def dvc_status_cache(repo_root: str = ".") -> str:
    """Real ``dvc status -c --quiet`` output."""
    try:
        out = subprocess.run(
            ["dvc", "status", "-c", "--quiet"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            cwd=repo_root,
        )
        return out.stdout
    except (OSError, subprocess.SubprocessError) as e:
        return f"<dvc status unavailable: {e}>"


# ─── RG7: license gate ─────────────────────────────────────────────────────────


def rg7_license_gate(
    license_entries: Sequence[Mapping[str, Any]],
    allow_noncommercial: bool,
    roboflow_slug_licenses: Mapping[str, str],
) -> GateResult:
    """Noncommercial sources enumerated + gate honored; Roboflow slug licenses recorded."""
    noncommercial_with_data = sorted(
        str(e["source"])
        for e in license_entries
        if e.get("noncommercial") and int(e.get("image_count", 0)) > 0
    )
    if noncommercial_with_data and not allow_noncommercial:
        return GateResult(
            "RG7",
            "license-gate",
            GATE_STATUS_FAIL,
            f"noncommercial source(s) {noncommercial_with_data} contributed data but "
            f"allow_noncommercial is false",
        )
    roboflow_present = any(
        str(e.get("source")) == "roboflow" and int(e.get("image_count", 0)) > 0
        for e in license_entries
    )
    if roboflow_present and not roboflow_slug_licenses:
        return GateResult(
            "RG7",
            "license-gate",
            GATE_STATUS_FAIL,
            "Roboflow contributed data but no per-slug licenses are recorded",
        )
    return GateResult(
        "RG7",
        "license-gate",
        GATE_STATUS_PASS,
        f"noncommercial sources: {noncommercial_with_data or 'none'}; "
        f"allow_noncommercial={allow_noncommercial}",
    )


# ─── RG8: split leakage + eval-set overlap ─────────────────────────────────────


def rg8_split_eval_leakage(qa_checks: Mapping[str, Any], eval_set: Mapping[str, Any]) -> GateResult:
    """Zero train/val/test leakage and zero eval-set image/house overlap."""
    problems: list[str] = []
    for check_name in ("train_val_leakage", "train_test_leakage"):
        count = int((qa_checks.get(check_name) or {}).get("count", 0))
        if count > 0:
            problems.append(f"{check_name}: {count}")

    overlap = eval_set.get("overlap") or {}
    if overlap.get("available"):
        exact = int(overlap.get("exact_overlap_count", 0))
        near = int(overlap.get("near_overlap_count", 0))
        if exact or near:
            problems.append(f"eval overlap: {exact} exact, {near} near-duplicate")

    house = eval_set.get("house_exclusivity") or {}
    if house.get("available"):
        shared = house.get("shared_houses") or []
        if shared:
            problems.append(f"houses in both training and eval: {shared}")

    if problems:
        return GateResult("RG8", "split-eval-leakage", GATE_STATUS_FAIL, "; ".join(problems))
    return GateResult("RG8", "split-eval-leakage", GATE_STATUS_PASS, "zero leakage, zero overlap")


# ─── RG9: capture targets ───────────────────────────────────────────────────────


def rg9_capture_targets(
    total_custom_images: int, houses: int, min_custom_images: int, min_houses: int
) -> GateResult:
    """Custom-capture image/house counts vs. this track's targets."""
    problems: list[str] = []
    if total_custom_images < min_custom_images:
        problems.append(f"custom images {total_custom_images} < required {min_custom_images}")
    if houses < min_houses:
        problems.append(f"houses {houses} < required {min_houses}")
    if problems:
        return GateResult("RG9", "capture-targets", GATE_STATUS_FAIL, "; ".join(problems))
    return GateResult(
        "RG9",
        "capture-targets",
        GATE_STATUS_PASS,
        f"{total_custom_images} custom images across {houses} house(s)",
    )


# ─── RG10: A/B benchmark + eval evidence ───────────────────────────────────────


def rg10_ab_benchmark_eval(eval_report_exists: bool, ab_benchmark_exists: bool) -> GateResult:
    """v1.0 acceptance evidence: A/B benchmark run + locked eval-set evaluation."""
    missing = [
        name
        for name, present in (
            ("eval_report", eval_report_exists),
            ("ab_benchmark", ab_benchmark_exists),
        )
        if not present
    ]
    if missing:
        return GateResult(
            "RG10", "ab-benchmark-eval", GATE_STATUS_FAIL, f"missing evidence: {', '.join(missing)}"
        )
    return GateResult(
        "RG10", "ab-benchmark-eval", GATE_STATUS_PASS, "A/B benchmark + eval reports present"
    )


# ─── Orchestrator ───────────────────────────────────────────────────────────────


def load_release_config(release_yaml_path: Path) -> dict[str, Any]:
    """Load ``configs/release.yaml``'s ``releases:`` section.

    Raises:
        ValueError: If the file has no ``releases:`` section.
    """
    raw = load_yaml(release_yaml_path)
    releases = raw.get("releases")
    if not isinstance(releases, dict) or not releases:
        raise ValueError(f"{release_yaml_path} has no non-empty 'releases:' section")
    return releases


def _collect_license_entries(raw_root: Path, sources_cfg: SourcesConfig) -> list[dict[str, Any]]:
    """Per-source (source, license, noncommercial, image_count) from raw manifests."""
    entries: list[dict[str, Any]] = []
    for manifest_path in sorted(raw_root.glob(f"*/{MANIFEST_FILENAME}")):
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_name = str(data.get("source", manifest_path.parent.name))
        source_cfg = sources_cfg.sources.get(source_name)
        entries.append(
            {
                "source": source_name,
                "license": data.get("license", ""),
                "noncommercial": bool(source_cfg.noncommercial) if source_cfg else False,
                "image_count": data.get("image_count", 0),
            }
        )
    return entries


def _rg2_errors(completeness_path: Path, data_yaml_path: Path) -> tuple[list[str], list[str]]:
    """(validation_errors, freshness_errors) for RG2, or a single not-found error."""
    if not completeness_path.exists():
        return ([f"{completeness_path.as_posix()} not found"], [])
    completeness = load_completeness(completeness_path)
    validation_errors = validate_completeness(completeness, data_yaml_path=data_yaml_path)
    freshness_errors: list[str] = []
    inputs = completeness.get("inputs", {})
    for name in ("merged_manifest", "split_summary"):
        record = inputs.get(name)
        if not isinstance(record, dict) or "path" not in record or "sha256" not in record:
            freshness_errors.append(f"{name}: no hash recorded in completeness.json")
            continue
        path = Path(str(record["path"]))
        if not path.exists():
            freshness_errors.append(f"{name}: {path.as_posix()} missing on disk")
        elif compute_file_hash(path) != record["sha256"]:
            freshness_errors.append(f"{name}: {path.as_posix()} changed since generation")
    return validation_errors, freshness_errors


def _load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def evaluate_release(
    version: str,
    release_yaml_path: Path = Path("configs/release.yaml"),
    sources_yaml_path: Path = Path("configs/dataset_sources.yaml"),
    data_yaml_path: Path = Path("configs/data.yaml"),
    completeness_path: Path = Path("data/processed/completeness.json"),
    qa_report_path: Path = Path("data/qa_reports/annotation_qa_report.json"),
    coverage_report_path: Path = Path("data/qa_reports/coverage_report.json"),
    quality_report_path: Path = Path("data/qa_reports/dataset_quality_report.json"),
    changelog_path: Path = Path("data/DATASET_CHANGELOG.md"),
    raw_root: Path = Path("data/raw"),
    captures_root: Path = Path("data/raw/custom_captures"),
    eval_report_path: Path = Path("data/qa_reports/eval_report.json"),
    ab_benchmark_dir: Path = Path("data/qa_reports/ab_benchmark"),
    repo_root: str = ".",
) -> ReleaseReport:
    """Evaluate a release version's declared gates against the current build.

    Only the gates the track actually requires (``configs/release.yaml``
    ``releases.<version>.gates``) are computed — a v0.5.0 check never needs
    (or expects) RG9/RG10 evidence that legitimately doesn't exist yet. The
    MODE prerequisite always runs.

    Raises:
        ValueError: If ``version`` is not declared in ``release_yaml_path``.
    """
    releases = load_release_config(release_yaml_path)
    if version not in releases:
        raise ValueError(
            f"Unknown release version '{version}' — not declared in "
            f"{release_yaml_path.as_posix()} (known: {sorted(releases)})"
        )
    track = releases[version]
    required_gate_ids = tuple(track.get("gates", []))

    def needs(gate_id: str) -> bool:
        return gate_id in required_gate_ids

    sources_cfg = load_sources_config(sources_yaml_path)
    results: list[GateResult] = [check_build_mode(sources_cfg.mode, str(track.get("mode", "full")))]

    qa_report = _load_json(qa_report_path)

    if needs("RG1"):
        results.append(rg1_qa_check(qa_report))

    if needs("RG2"):
        validation_errors, freshness_errors = _rg2_errors(completeness_path, data_yaml_path)
        results.append(rg2_completeness_freshness(validation_errors, freshness_errors))

    if needs("RG3"):
        coverage_report = _load_json(coverage_report_path)
        quality_report = _load_json(quality_report_path)
        min_coverage_score = {
            str(k): float(v) for k, v in (track.get("min_coverage_score") or {}).items()
        }
        results.append(
            rg3_coverage_quality(
                coverage_report,
                quality_report,
                int(track.get("min_verified_cells", 0)),
                min_coverage_score,
            )
        )

    if needs("RG4"):
        changelog_text = (
            changelog_path.read_text(encoding="utf-8") if changelog_path.exists() else ""
        )
        results.append(rg4_changelog_entry(changelog_text, version))

    if needs("RG5"):
        results.append(
            rg5_working_tree_tagged(
                git_porcelain_status(repo_root), git_tags_at_head(repo_root), version
            )
        )

    if needs("RG6"):
        results.append(rg6_dvc_push_verified(dvc_status_cache(repo_root)))

    if needs("RG7"):
        license_entries = _collect_license_entries(raw_root, sources_cfg)
        roboflow_slugs = {
            str(k): str(v) for k, v in (track.get("roboflow_slug_licenses") or {}).items()
        }
        results.append(
            rg7_license_gate(license_entries, sources_cfg.allow_noncommercial, roboflow_slugs)
        )

    if needs("RG8"):
        qa_checks = (qa_report or {}).get("checks", {})
        eval_set = (qa_report or {}).get("eval_set", {})
        results.append(rg8_split_eval_leakage(qa_checks, eval_set))

    if needs("RG9"):
        sessions = load_session_manifests(captures_root)
        total_images = sum(s.image_count for s in sessions)
        houses = len({s.house_id for s in sessions if s.house_id})
        results.append(
            rg9_capture_targets(
                total_images,
                houses,
                int(track.get("min_custom_images", 0)),
                int(track.get("min_houses", 0)),
            )
        )

    if needs("RG10"):
        results.append(rg10_ab_benchmark_eval(eval_report_path.exists(), ab_benchmark_dir.exists()))

    return ReleaseReport(
        version=version, required_gate_ids=required_gate_ids, results=tuple(results)
    )
