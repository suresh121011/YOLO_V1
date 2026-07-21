"""
src.dataset.completeness — Per-Image Label-Completeness Artifact
================================================================

Phase-4: builds and validates ``data/processed/completeness.json`` — the
single artifact mapping every processed (post-split) image to the taxonomy
classes its source annotates exhaustively. The missing-annotation mitigation
trainer masks the classification loss with exactly this information.

Data flow:
    merged_manifest.json (label_completeness + image_provenance)
        + configs/dataset_sources.yaml (top-level ``completeness:`` policies)
        + configs/data.yaml (taxonomy)
        + data/processed/images/{train,val,test} (what actually trains)
        + capture session manifests (per-session trusted classes)
    → build_completeness() → completeness.json

Failure philosophy: every ambiguity (unknown image, unmapped source, unknown
class name, config/manifest drift, duplicate keys) raises
:class:`CompletenessError` — a wrong mask silently corrupts supervision, so
the generator never guesses. See docs/06_training_engineering/ADR-P4-03.

Schema evolution follows the manifest convention: consumers ignore unknown
keys; additive fields do not bump ``COMPLETENESS_SCHEMA_VERSION``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.dataset.annotation.ledger import LedgerView
from src.dataset.completeness_policies import (
    CompletenessError,
    CompletenessPolicyProvider,
    PolicyContext,
    get_policy_provider,
    registered_policy_modes,
)
from src.dataset.manifest import MergedManifest
from src.utils.config_helpers import get_class_names_from_data_yaml, load_data_config, load_yaml
from src.utils.dataset_utils import compute_file_hash, find_image_files
from src.utils.report_utils import timestamp_str

logger = logging.getLogger(__name__)

COMPLETENESS_SCHEMA_VERSION = 1
COMPLETENESS_FILENAME = "completeness.json"
GENERATOR_SCRIPT = "scripts/dataset/11_generate_completeness.py"

VALID_SPLITS: tuple[str, ...] = ("train", "val", "test")

#: Splits whose images must be covered for training (test is validated too
#: when present, but train/val are what the trainer touches).
TRAINING_SPLITS: tuple[str, ...] = ("train", "val")


# ─── Fingerprints & hashing ───────────────────────────────────────────────────


def taxonomy_fingerprint(nc: int, names: Mapping[int, str]) -> str:
    """Return a stable fingerprint of the class taxonomy.

    The fingerprint changes iff ``nc`` or any (id, name) pair changes, letting
    consumers detect taxonomy drift between artifact generation and training.

    Args:
        nc:    Number of classes.
        names: Class id → name mapping.

    Returns:
        ``sha256:<hex>`` over the canonical JSON of {nc, ordered names}.
    """
    canonical = json.dumps(
        {"nc": nc, "names": [[i, names[i]] for i in sorted(names)]},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git_commit() -> str:
    """Return the current short git commit hash, or 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``object_pairs_hook`` that fails on duplicate JSON object keys."""
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise CompletenessError(
                f"Duplicate JSON key '{key}' in completeness artifact — the file is "
                f"corrupt or was hand-edited; regenerate it via "
                f"`dvc repro generate_completeness`."
            )
        seen[key] = value
    return seen


def load_completeness(path: Path) -> dict[str, Any]:
    """Load a completeness artifact, rejecting duplicate JSON keys.

    Args:
        path: Path to completeness.json.

    Returns:
        Parsed artifact dict.

    Raises:
        FileNotFoundError:  If the file does not exist.
        CompletenessError:  On invalid JSON, non-object root, or duplicate keys.
    """
    if not path.exists():
        raise FileNotFoundError(f"Completeness artifact not found: {path.absolute()}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as e:
        raise CompletenessError(f"Invalid JSON in completeness artifact {path}: {e}") from e
    if not isinstance(raw, dict):
        raise CompletenessError(f"Completeness artifact root must be a JSON object: {path}")
    return raw


def save_completeness(artifact: dict[str, Any], path: Path) -> None:
    """Write a completeness artifact as pretty-printed UTF-8 JSON.

    Args:
        artifact: Artifact dict (from :func:`build_completeness`).
        path:     Destination path. Parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(f"Completeness artifact written: {path}")


# ─── Builder ──────────────────────────────────────────────────────────────────


def _scan_processed_images(processed_images_root: Path) -> dict[str, str]:
    """Map every processed image basename to its split, rejecting duplicates.

    Args:
        processed_images_root: data/processed/images (contains split subdirs).

    Returns:
        Image basename → split name.

    Raises:
        CompletenessError: If no images are found or a basename appears twice
                           (within or across splits).
    """
    split_by_name: dict[str, str] = {}
    for split in VALID_SPLITS:
        for img in find_image_files(processed_images_root / split):
            if img.name in split_by_name:
                raise CompletenessError(
                    f"Duplicate image filename '{img.name}' found in splits "
                    f"'{split_by_name[img.name]}' and '{split}' — completeness lookup "
                    f"is keyed by basename, which must be unique. Re-run the split stage."
                )
            split_by_name[img.name] = split
    if not split_by_name:
        raise CompletenessError(
            f"No processed images found under {processed_images_root} — "
            f"run the split stage first (dvc repro split_train_val_test)."
        )
    return split_by_name


def _hash_input(path: Path, extra_keys: tuple[str, ...] = ()) -> dict[str, Any]:
    """Record path + sha256 (+ selected JSON fields) for one input file."""
    record: dict[str, Any] = {
        "path": path.as_posix(),
        "sha256": compute_file_hash(path),
    }
    if extra_keys:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key in extra_keys:
                    if key in data:
                        record[key] = data[key]
        except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
            logger.warning(f"Could not read {path} for input metadata; recording hash only")
    return record


def build_completeness(
    merged_manifest_path: Path,
    processed_images_root: Path,
    split_summary_path: Path,
    data_yaml_path: Path,
    sources_yaml_path: Path,
    capture_manifests_dir: Path | None,
) -> dict[str, Any]:
    """Build the per-image completeness artifact from pipeline outputs.

    Args:
        merged_manifest_path:  data/merged/merged_manifest.json.
        processed_images_root: data/processed/images (split subdirs).
        split_summary_path:    data/processed/split_report/split_summary.json.
        data_yaml_path:        configs/data.yaml (taxonomy).
        sources_yaml_path:     configs/dataset_sources.yaml (must contain the
                               top-level ``completeness:`` policy section).
        capture_manifests_dir: data/raw/custom_captures/manifests, or None.

    Returns:
        The artifact dict (see module docstring / ADR-P4-03 for the schema).

    Raises:
        CompletenessError:  On any ambiguity — unknown image, unmapped source,
                            unknown policy mode, class-name mismatch, duplicate
                            filenames or policy keys.
        FileNotFoundError:  If a required input file is missing.
        ValueError:         If configs/data.yaml is structurally invalid.
    """
    data_cfg = load_data_config(data_yaml_path)
    names = get_class_names_from_data_yaml(data_cfg)
    nc = int(data_cfg["nc"])
    if len(names) != nc or sorted(names) != list(range(nc)):
        raise CompletenessError(
            f"Taxonomy in {data_yaml_path} is inconsistent: nc={nc} but names cover "
            f"ids {sorted(names)} — expected exactly 0..{nc - 1}."
        )
    class_ids_by_name = {name: cid for cid, name in names.items()}

    merged = MergedManifest.load(merged_manifest_path)

    sources_cfg = load_yaml(sources_yaml_path)
    completeness_cfg = sources_cfg.get("completeness")
    if not isinstance(completeness_cfg, dict) or not isinstance(
        completeness_cfg.get("policies"), dict
    ):
        raise CompletenessError(
            f"{sources_yaml_path} has no top-level 'completeness.policies' section. "
            f"Declare an explicit policy mode per source "
            f"(modes: {registered_policy_modes()}) — completeness semantics are never "
            f"inferred from source names."
        )
    policy_modes: dict[str, str] = {str(k): str(v) for k, v in completeness_cfg["policies"].items()}

    split_by_name = _scan_processed_images(processed_images_root)

    # Attribute every processed image to its provenance source.
    provenance = merged.image_provenance
    unknown_images = sorted(n for n in split_by_name if n not in provenance)
    if unknown_images:
        preview = ", ".join(unknown_images[:10])
        raise CompletenessError(
            f"{len(unknown_images)} processed image(s) missing from the merged "
            f"manifest's image_provenance (first 10: {preview}) — data/processed and "
            f"data/merged are out of sync; re-run `dvc repro`."
        )

    sources_used = sorted({provenance[n] for n in split_by_name})

    # M3: load + validate the human-verification ledger, only when a source
    # actually opts into it (explicit per-source, never inferred — D9).
    uses_ledger = any(
        policy_modes.get(source) == "trusted_list_with_ledger" for source in sources_used
    )
    ledger_view: LedgerView | None = None
    ledger_input_record: dict[str, Any] | None = None
    if uses_ledger:
        ledger_path_str = completeness_cfg.get("ledger_path")
        if not ledger_path_str:
            raise CompletenessError(
                f"A source uses policy 'trusted_list_with_ledger' but {sources_yaml_path} "
                f"has no 'completeness.ledger_path' — declare it alongside "
                f"'completeness.policies'."
            )
        ledger_path = Path(str(ledger_path_str))
        ledger_view = LedgerView.load(ledger_path)

        live_fp = taxonomy_fingerprint(nc, names)
        recorded_fp = ledger_view.taxonomy_fingerprint()
        if recorded_fp and recorded_fp != live_fp:
            raise CompletenessError(
                f"Ledger taxonomy fingerprint drift: ledger recorded {recorded_fp!r}, live "
                f"taxonomy is {live_fp!r} — reconcile before generating completeness."
            )
        for filename in sorted(ledger_view.all_images()):
            if filename not in provenance:
                raise CompletenessError(
                    f"Ledger entry '{filename}' is absent from the merged manifest's "
                    f"image_provenance — re-run merge_datasets or reconcile the ledger."
                )
            declared_source = ledger_view.entry_source(filename)
            actual_source = provenance[filename]
            if declared_source != actual_source:
                raise CompletenessError(
                    f"Ledger entry '{filename}' is attributed to source '{declared_source}' "
                    f"but the merged manifest attributes it to '{actual_source}' — "
                    f"provenance drift; reconcile before generating completeness."
                )
        if ledger_path.exists():
            ledger_input_record = _hash_input(ledger_path, ("updated_at",))

    # Resolve policies per source through the provider registry.
    policies: dict[str, tuple[int, ...]] = {}
    policy_mode_by_key: dict[str, str] = {}
    providers: dict[str, CompletenessPolicyProvider] = {}
    contexts: dict[str, PolicyContext] = {}
    for source in sources_used:
        if source not in policy_modes:
            raise CompletenessError(
                f"Source '{source}' appears in the merged dataset but has no entry "
                f"under 'completeness.policies' in {sources_yaml_path}. Every source "
                f"feeding training must declare a policy mode "
                f"({registered_policy_modes()}) — unsupported datasets are rejected."
            )
        source_cfg = (sources_cfg.get("sources") or {}).get(source) or {}
        config_trusted = source_cfg.get("trusted_classes")
        manifest_trusted = merged.label_completeness.get(source)
        ctx = PolicyContext(
            source=source,
            manifest_trusted_classes=(
                tuple(manifest_trusted) if manifest_trusted is not None else None
            ),
            config_trusted_classes=(tuple(config_trusted) if config_trusted is not None else None),
            class_ids_by_name=class_ids_by_name,
            nc=nc,
            capture_manifests_dir=capture_manifests_dir,
            verification_ledger=ledger_view,
        )
        provider = get_policy_provider(policy_modes[source])
        resolved = provider.resolve_policies(ctx)
        collisions = sorted(set(resolved) & set(policies))
        if collisions:
            raise CompletenessError(
                f"Duplicate policy key(s) {collisions} — sources resolved overlapping "
                f"policy identifiers; policy keys must be globally unique."
            )
        policies.update(resolved)
        policy_mode_by_key.update({key: provider.mode for key in resolved})
        providers[source] = provider
        contexts[source] = ctx

    # Attribute every image to a policy key.
    images: dict[str, dict[str, str]] = {}
    for name in sorted(split_by_name):
        source = provenance[name]
        key = providers[source].policy_key_for_image(contexts[source], name)
        if key not in policies:  # pragma: no cover - providers guarantee this
            raise CompletenessError(
                f"Image '{name}' resolved to policy '{key}' which was never defined — "
                f"provider '{providers[source].mode}' is inconsistent."
            )
        images[name] = {"policy": key, "split": split_by_name[name]}

    # Stats.
    by_split = Counter(entry["split"] for entry in images.values())
    by_policy = Counter(entry["policy"] for entry in images.values())
    class_trusted_image_counts: Counter[str] = Counter()
    trusted_total = 0
    for entry in images.values():
        ids = policies[entry["policy"]]
        trusted_total += len(ids)
        for cid in ids:
            class_trusted_image_counts[names[cid]] += 1

    artifact: dict[str, Any] = {
        "schema_version": COMPLETENESS_SCHEMA_VERSION,
        "generated_at": timestamp_str(),
        "generator": {"script": GENERATOR_SCRIPT, "git_commit": _git_commit()},
        "taxonomy": {
            "nc": nc,
            "names": {str(cid): names[cid] for cid in sorted(names)},
            "fingerprint": taxonomy_fingerprint(nc, names),
        },
        "inputs": {
            "merged_manifest": _hash_input(merged_manifest_path, ("created_at",)),
            "split_summary": _hash_input(split_summary_path, ("seed", "timestamp", "strategy")),
            "dataset_sources_mode": str(sources_cfg.get("mode", "unknown")),
            **({"ledger": ledger_input_record} if ledger_input_record is not None else {}),
        },
        "policies": {
            key: {"mode": policy_mode_by_key[key], "trusted_class_ids": list(ids)}
            for key, ids in sorted(policies.items())
        },
        "images": images,
        "stats": {
            "images_total": len(images),
            "by_split": dict(sorted(by_split.items())),
            "by_policy": dict(sorted(by_policy.items())),
            "class_trusted_image_counts": dict(sorted(class_trusted_image_counts.items())),
            "mean_trusted_classes_per_image": (
                round(trusted_total / len(images), 3) if images else 0.0
            ),
        },
    }
    logger.info(
        f"Completeness built: {len(images)} images, {len(policies)} policies, "
        f"sources={sources_used}"
    )
    return artifact


# ─── Validator ────────────────────────────────────────────────────────────────


def validate_completeness(
    artifact: dict[str, Any],
    data_yaml_path: Path | None = None,
) -> list[str]:
    """Validate an artifact's self-consistency (and taxonomy vs live config).

    Args:
        artifact:       Parsed artifact (from :func:`load_completeness`).
        data_yaml_path: When given, the embedded taxonomy is additionally
                        checked against the live configs/data.yaml.

    Returns:
        List of human-readable error strings; empty means valid. Warnings
        (e.g. unused policies) are NOT errors — see :func:`find_unused_policies`.
    """
    errors: list[str] = []

    schema = artifact.get("schema_version")
    if not isinstance(schema, int) or schema > COMPLETENESS_SCHEMA_VERSION or schema < 1:
        errors.append(
            f"Unsupported schema_version {schema!r} (supported: 1..{COMPLETENESS_SCHEMA_VERSION})"
        )

    # Taxonomy block.
    taxonomy = artifact.get("taxonomy")
    nc = 0
    names: dict[int, str] = {}
    if not isinstance(taxonomy, dict):
        errors.append("Missing or invalid 'taxonomy' block")
    else:
        raw_nc = taxonomy.get("nc")
        raw_names = taxonomy.get("names")
        if not isinstance(raw_nc, int) or raw_nc <= 0:
            errors.append(f"taxonomy.nc must be a positive int, got {raw_nc!r}")
        elif not isinstance(raw_names, dict):
            errors.append("taxonomy.names must be a dict of id → name")
        else:
            nc = raw_nc
            try:
                names = {int(k): str(v) for k, v in raw_names.items()}
            except (TypeError, ValueError):
                errors.append("taxonomy.names keys must be integer-like")
            if names and sorted(names) != list(range(nc)):
                errors.append(
                    f"taxonomy.names ids must be exactly 0..{nc - 1}, got {sorted(names)}"
                )
            if names and sorted(names) == list(range(nc)):
                expected = taxonomy_fingerprint(nc, names)
                if taxonomy.get("fingerprint") != expected:
                    errors.append(
                        "taxonomy.fingerprint does not match the embedded names — "
                        "artifact is corrupt; regenerate it"
                    )
        if data_yaml_path is not None and not errors:
            live_cfg = load_data_config(data_yaml_path)
            live_names = get_class_names_from_data_yaml(live_cfg)
            live_fp = taxonomy_fingerprint(int(live_cfg["nc"]), live_names)
            if taxonomy.get("fingerprint") != live_fp:
                errors.append(
                    f"Taxonomy drift: artifact fingerprint {taxonomy.get('fingerprint')} "
                    f"!= live {data_yaml_path} fingerprint {live_fp}. "
                    f"Re-run `dvc repro generate_completeness`."
                )

    # Policies block.
    policies = artifact.get("policies")
    if not isinstance(policies, dict) or not policies:
        errors.append("Missing or empty 'policies' block")
        policies = {}
    known_modes = set(registered_policy_modes())
    for key, policy in policies.items():
        if not isinstance(policy, dict):
            errors.append(f"Policy '{key}' must be an object")
            continue
        mode = policy.get("mode")
        ids = policy.get("trusted_class_ids")
        if mode not in known_modes:
            errors.append(
                f"Policy '{key}' has unregistered mode {mode!r} (known: {sorted(known_modes)})"
            )
        if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
            errors.append(f"Policy '{key}' trusted_class_ids must be a list of ints")
            continue
        if nc and any(i < 0 or i >= nc for i in ids):
            errors.append(f"Policy '{key}' has class ids outside [0, {nc}): {ids}")
        if sorted(set(ids)) != ids:
            errors.append(f"Policy '{key}' trusted_class_ids must be sorted and unique: {ids}")
        if mode == "verified_absence_all" and nc and len(ids) != nc:
            errors.append(
                f"Policy '{key}' (verified_absence_all) must trust all {nc} classes, has {len(ids)}"
            )
        if mode == "trusted_list" and not ids:
            errors.append(f"Policy '{key}' (trusted_list) must trust at least one class")

    # Images block.
    images = artifact.get("images")
    if not isinstance(images, dict) or not images:
        errors.append("Missing or empty 'images' block")
        images = {}
    orphans: list[str] = []
    for name, entry in images.items():
        if not isinstance(entry, dict):
            errors.append(f"Image '{name}' entry must be an object")
            continue
        if entry.get("policy") not in policies:
            orphans.append(name)
        if entry.get("split") not in VALID_SPLITS:
            errors.append(f"Image '{name}' has invalid split {entry.get('split')!r}")
    if orphans:
        preview = ", ".join(sorted(orphans)[:10])
        errors.append(
            f"{len(orphans)} image(s) reference undefined policies "
            f"(orphan references; first 10: {preview})"
        )

    # Stats sanity (cheap self-consistency only).
    stats = artifact.get("stats")
    if isinstance(stats, dict) and isinstance(images, dict) and images:
        if stats.get("images_total") != len(images):
            errors.append(
                f"stats.images_total={stats.get('images_total')} != len(images)={len(images)}"
            )

    return errors


def find_unused_policies(artifact: dict[str, Any]) -> list[str]:
    """Return policy keys defined in the artifact but referenced by no image.

    Unused policies are warnings, not errors: a finalized capture session may
    legitimately have all its images deduplicated away at merge time.
    """
    policies = artifact.get("policies") or {}
    images = artifact.get("images") or {}
    referenced = {entry.get("policy") for entry in images.values() if isinstance(entry, dict)}
    return sorted(set(policies) - referenced)


def summarize_completeness(artifact: dict[str, Any]) -> dict[str, Any]:
    """Produce report-ready summary rows for one artifact.

    Args:
        artifact: Parsed, validated artifact.

    Returns:
        Dict with 'policy_rows' (one dict per policy, CSV-friendly),
        'unused_policies', and the artifact's 'stats' block.
    """
    taxonomy = artifact.get("taxonomy") or {}
    names = {int(k): str(v) for k, v in (taxonomy.get("names") or {}).items()}
    nc = int(taxonomy.get("nc") or 0)
    by_policy = (artifact.get("stats") or {}).get("by_policy") or {}

    policy_rows: list[dict[str, Any]] = []
    for key, policy in sorted((artifact.get("policies") or {}).items()):
        ids = policy.get("trusted_class_ids") or []
        policy_rows.append(
            {
                "policy": key,
                "mode": policy.get("mode", ""),
                "images": by_policy.get(key, 0),
                "trusted_count": len(ids),
                "untrusted_count": max(nc - len(ids), 0),
                "trusted_classes": " ".join(names.get(i, f"?{i}") for i in ids),
            }
        )
    return {
        "policy_rows": policy_rows,
        "unused_policies": find_unused_policies(artifact),
        "stats": artifact.get("stats") or {},
    }
