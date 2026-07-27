# Dataset Pipeline Reproduction Log

Executed clean-machine reproduction tests for the Phase-2 dataset pipeline
(WP3.0 exit criterion G0). Newest first.

---

## 2026-07-14 — Clean-machine rebuild-from-sources (Linux, partial by network policy)

| | |
|---|---|
| **Machine** | Ephemeral Linux container (x86_64, Python 3.11.15) — no prior repo state, no DVC cache, no dataset |
| **Commit** | `wp3.0-platform-remediation` @ `b1c1a84` (fresh `git clone`, separate working copy) |
| **Variant** | Rebuild-from-sources (`dvc repro`), smoke mode. The `dvc pull` variant was **pending** at the time of this run — it required the first `dvc push` from the machine holding the data (see §6 runbook). **That blocker cleared 2026-07-27**: the cache is now on `localstore` and on S3 (46,337 objects / 6.21 GB, `dvc status -c -r storage` in sync). The pull-variant was then **partially executed 2026-07-27** — see "Pull-variant fetch check" below. |
| **Deps** | `dvc[s3] 3.67.1`, pyyaml, requests, pillow, numpy, opencv-python-headless in a fresh `.venv`. *Deviation:* full `requirements.txt` (ultralytics/torch/sounddevice) not installed — unused by the dataset DAG (training stage frozen). |

### Result: PASS (for every stage the environment's network policy allowed)

| Stage | Outcome |
|---|---|
| `download_openimages` | ✅ 60/60 images via `dvc repro`; CSV index streamed; 304 validation candidates (deterministic) |
| `download_roboflow` | ✅ graceful skip (`datasets: []`), empty out recorded — matches lock |
| `download_coco`, `collect_negatives` | ⛔ blocked: `images.cocodataset.org` denied by the sandbox egress policy (CONNECT 403) — **environment limitation, not a pipeline defect** |
| `download_wider_face` | ⛔ blocked: `huggingface.co` unreachable from the sandbox |
| `remap_classes` → `merge_datasets` → `split_train_val_test` → `qa_check` | ✅ all exit 0 over the acquired subset (*deviation:* the three blocked sources set `enabled: false` locally in the test clone only; empty raw dirs created to satisfy DVC deps) |

### QA metric from the fresh build

```
total_images: 57   total_boxes: 79   critical: 0   warnings: 0
train/val leakage: PASS   train/test leakage: PASS   license violation: false
image quality: 57 scanned, 6 blurry, 0 low-light
```

### Cross-platform consistency vs the original Windows smoke build

- **Acquisition is deterministic:** identical 60 Open Images selected; image bytes
  identical (17.2 MB output dir differs by only ~178 bytes — all in label files).
- **Filtering is deterministic:** 57/60 accepted with the same 3 rejected as the
  Windows run (committed QA report: openimages accepted 57, filtered 3).
- **Blur detection consistent:** 6 blurry flagged = the 6 Open Images blur samples
  in the committed report.
- The QA report now records `data_dir: data/processed` (portable relative path) —
  the WP3.0 path-leak fix confirmed in a production run.

### Findings

1. **F1 — `dvc.lock` is line-ending sensitive (cross-platform).** Label files are
   written in text mode, so Windows produces CRLF and Linux LF; script *dependency*
   hashes flip the same way via git EOL translation. Rebuilds on a different OS
   therefore rewrite `dvc.lock` even when content is semantically identical.
   *Recommendation (Phase-3 backlog):* write labels/sidecars with `newline="\n"`
   and pin `*.py eol=lf` (+ generated `*.txt` label policy) in `.gitattributes`;
   until then, treat the lock as OS-specific and regenerate on one canonical OS.
2. **F2 — Sandbox egress policy blocks 2 of 4 source hosts** (`images.cocodataset.org`,
   `huggingface.co`); `storage.googleapis.com` and `open-images-dataset.s3.amazonaws.com`
   are allowed. A **full** rebuild-from-sources needs unrestricted egress (any dev
   machine) or a widened environment network policy.
3. **F3 — Disabled sources don't create their DVC outs.** A source with
   `enabled: false` exits 0 without creating its `data/raw/<source>` out, so
   `dvc repro` fails on the missing directory. Harmless today (all sources
   enabled) but worth fixing when source toggling becomes routine.

### Verdict

The pipeline reproduces end-to-end on a clean Linux machine for everything the
network allowed, with deterministic acquisition/filtering and zero QA issues.
G0 is **partially satisfied**; full satisfaction requires either
(a) the pending `dvc push` from the data-holding machine followed by
`dvc pull && dvc repro qa_check` on a clean machine, or
(b) one full rebuild-from-sources on a machine with unrestricted egress.

### Commands (as run)

```bash
git clone --branch wp3.0-platform-remediation <repo> repro-test && cd repro-test
python -m venv .venv && .venv/bin/pip install "dvc[s3]>=3.50,<4.0" pyyaml requests pillow numpy opencv-python-headless
dvc repro download_openimages download_roboflow            # network-allowed acquisition
# local-only deviation: coco/wider_face/negatives enabled:false + mkdir empty raw dirs
dvc repro --single-item remap_classes
dvc repro --single-item merge_datasets
dvc repro --single-item split_train_val_test
dvc repro --single-item qa_check
```

---

## Pull-variant fetch check — 2026-07-27

Records what was actually executed. This is **not** the full clean-machine gate
(`dvc pull && dvc repro qa_check`); read the limitations before citing it.

| | |
|---|---|
| **Method** | `git clone --depth 1` from GitHub into a scratch directory (no DVC cache, no `.dvc/config.local`), then `dvc pull -r storage data/processed/split_report` |
| **Commit** | `main` @ `cc1cf34` |
| **Result** | **PASS** — `3 files fetched and 2 files added` |
| **Content verified** | `split_summary.json`: `total_images` 17,888 · `total_labels` 17,888 · `leakage_count` 0 · splits 14,323 / 1,773 / 1,792 · **0 CRLF bytes** |

### What this establishes

1. The `storage` remote in the tracked `.dvc/config` resolves from a clean
   checkout. Credentials came from the standard AWS chain with no
   project-specific setup, which validates the decision to set no `profile` key.
2. S3 is **readable**, not merely writable — every earlier check confirmed
   upload only.
3. The dataset is retrievable by a checkout that never built it, which is the
   substance of risk C-1.
4. The LF fix survives workspace → cache → S3 → fresh clone byte-identically.

### Limitations — why this is not the full gate

- **Same machine, same OS, same AWS credentials.** A genuinely independent host
  (different OS, different IAM principal) is still untested.
- **One small output pulled**, not the full 6.2 GB — the fetch mechanism was
  what was unproven; a complete pull would incur real egress to demonstrate the
  same thing.
- **`dvc repro qa_check` was not run** in the clone, so end-to-end pipeline
  reproduction from a pull remains unverified.

The full gate stays open. This closes the narrower question of whether the
remote is fetchable at all, which until now was assumed rather than tested.
