# Phase-5 M5 Release Runbook — Releases as Code

> Operational SOP for cutting a dataset release (`dataset-v0.5.0` ..
> `dataset-v1.0.0`). Tooling reference: `docs/07_dataset_production/README.md`.
> Architecture/design: ADR-P5-07 (releases as code).

---

## 0. Overview

Every release is a git tag + an immutable `data/releases/dataset-vX.Y.Z/
release_manifest.json` + a `dvc.lock` snapshot. Nothing about a release is
hand-typed: `18_make_release.py` evaluates `configs/release.yaml`'s declared
gates for that version's track against the **current build**, and only
writes the manifest when the verdict is not `FAIL`.

```
check   → evaluate gates, print the report, exit 1 on FAIL. No side effects.
make    → same evaluation; on PASS/WARN, writes the release manifest.
verify  → re-load an existing manifest, confirm no gate recorded FAIL,
          compare the current dvc.lock hash against the one at release time.
```

The gate list per track (`releases.<version>.gates` in `configs/release.yaml`)
determines what actually blocks that version — `check dataset-v0.5.0` never
evaluates RG9/RG10 (custom captures, A/B evidence) because that track
doesn't require them yet; `check dataset-v1.0.0` does.

---

## 1. Check the current build

```bash
python scripts/dataset/18_make_release.py check dataset-v0.5.0
```

Fix every reported `FAIL` before proceeding — most map directly to an
earlier milestone's tooling:

| Gate | What it means when it fails |
|:---|:---|
| `MODE` | `configs/dataset_sources.yaml` mode doesn't match the track (flip to `full`, re-run `dvc repro`) |
| RG1 | `qa_check` has a critical, or an M3/M4 artifact sweep found something — see `annotation_qa_report.json` |
| RG2 | `completeness.json` is invalid or its recorded input hashes are stale — re-run `dvc repro generate_completeness` |
| RG3 | `coverage_report.json`/`dataset_quality_report.json` missing, or a per-class `min_coverage_score` threshold isn't met yet |
| RG4 | `data/DATASET_CHANGELOG.md` has no `## <version>` heading — add one (§2) |
| RG5 | Working tree isn't clean, or HEAD isn't tagged `<version>` (§3) |
| RG6 | `dvc status -c` isn't clean — run `dvc push` |
| RG7 | A noncommercial source contributed data while `allow_noncommercial: false`, or Roboflow contributed data with no recorded slug licenses |
| RG8 | Train/val/test leakage, or the locked eval set overlaps train-facing images/houses |
| RG9 | Custom-capture image/house counts below the track's `min_custom_images`/`min_houses` |
| RG10 | A/B benchmark or locked-eval-set evaluation reports missing (v1.0 evidence) |

---

## 2. Add the changelog entry

Append a `## <version> — <date>` section to `data/DATASET_CHANGELOG.md`
(mirror the existing `dataset-v0.1.0-smoke` entry's shape: per-source
acceptance counts, split summary, QA verdict, known follow-ups).

---

## 3. Tag the release

RG5 requires the tag to already exist at check/make time — tag **before**
running `make`, not after:

```bash
git tag dataset-v0.5.0
```

(Working tree must be clean at this point — commit everything the release
depends on first.)

---

## 4. Make the release

```bash
python scripts/dataset/18_make_release.py make dataset-v0.5.0
```

On a non-FAIL verdict this writes
`data/releases/dataset-v0.5.0/release_manifest.json` — gate results,
per-input artifact sha256 hashes, dataset counts, license summary, and
reproducibility metadata (python/dvc versions, split seed, param-file
hashes). It never recomputes anything those artifacts already established
(mirrors `quality.py`'s L5 discipline) — a wrong number here means an
earlier stage's artifact was wrong, not a bug in `make` itself.

---

## 5. Pin and push

```bash
dvc commit -f record_release
git add dvc.lock data/releases data/DATASET_CHANGELOG.md
git commit -m "release: dataset-v0.5.0"
git push --tags
dvc push
```

`record_release` is a **frozen**, no-deps DVC stage (mirrors the human-loop
annotation stages, `dvc.yaml` header note) — `dvc repro` never touches it;
humans run `make` directly and `dvc commit -f` records the result.

---

## 6. Verify (post-hoc sanity check)

```bash
python scripts/dataset/18_make_release.py verify dataset-v0.5.0
```

Confirms the manifest parses, matches the requested version, and records
no `FAIL` gate (a `make` never writes one — this catches hand-edited or
corrupted manifests). Also compares the current `dvc.lock` hash against the
one recorded at release time — a mismatch is expected on any later commit
and only means "you're not currently checked out at this release", not a
release defect.

---

## 7. Rollback

A failed `make` attempt leaves no trace (the manifest is only written on a
non-FAIL verdict) — nothing to roll back. To undo an already-recorded
release: delete the git tag, `git revert` the release commit, and
`dvc checkout` to restore `data/releases/` to its prior state. Earlier
releases are never invalidated by a later one — a failed v1.0 attempt
leaves v0.9 fully consumable (plan §Rollback strategy).

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|:---|:---|:---|
| `MODE` fails immediately | `configs/dataset_sources.yaml` still `mode: smoke` | Flip to `full`, `dvc repro`, re-check |
| RG5 fails even after tagging | Tagged before the last commit, or working tree has untracked changes | `git tag -f` at the correct commit; `git status` must be fully clean |
| RG6 flaky / "dvc status unavailable" | `dvc` not on `PATH` for the invoking shell | Prepend `.venv/Scripts` (or the venv's bin dir) to `PATH` before running |
| RG3 fails with "no coverage_score recorded" | A `min_coverage_score` class in `release.yaml` was never targeted by `auto_annotate` | Check `configs/annotation.yaml`'s `targeting.priority_classes` covers it |
| `verify` reports a FAILed gate | Manifest was hand-edited, or `make` was bypassed | Re-run `make` from a clean, passing build — never hand-edit a release manifest |
| `Unknown release version` | Version string doesn't match a `configs/release.yaml` key exactly | Check for typos — versions are exact string keys, not parsed semver |

---

Previous: [README.md](./README.md), [verification_runbook.md](./verification_runbook.md)

Related: [risk_register.md](../01_executive_implementation_plan/risk_register.md) (R30–R38)
