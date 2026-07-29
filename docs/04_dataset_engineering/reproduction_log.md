# Dataset Pipeline Reproduction Log

Executed clean-machine reproduction tests for the Phase-2 dataset pipeline
(WP3.0 exit criterion G0). **Newest first** — and, as of 2026-07-29, that is
actually true: the file previously said "newest first" while listing entries
oldest-first, so the reader's eye landed on the *weakest* evidence.

---

## 2026-07-29 — `dataset-v0.6.0` release ceremony (F6/F7/F9)

The release this log's evidence was gathered for. Cut on `main`, gates
`RG1`–`RG8` per [ADR-P5-13](../07_dataset_production/adr/ADR-P5-13-v060-local-capture-release-track.md).

| | |
|---|---|
| **Release** | `dataset-v0.6.0` — 24,352 images / 104,289 boxes, `mode: full` |
| **Manifest** | `data/releases/dataset-v0.6.0/release_manifest.json` (git-tracked) |
| **Gates** | MODE, RG1–RG8 all **PASS**; verdict **PASS** |
| **Reproducibility block** | python 3.14.3 · dvc 3.67.1 · seed 42 · 5 param-file sha256s |
| **Artifact hashes** | completeness, qa_report, coverage_report, quality_report, merged_manifest, ledger — all recorded |

### Two defects caught during the ceremony, both fixed before anything shipped

1. **The manifest recorded `"dvc": "unknown"`** — in the `reproducibility` block
   whose only job is to say what produced the release. `_dvc_version()` shelled
   out to a bare `dvc`, which is not on PATH when the release script runs as
   `.venv/Scripts/python.exe 18_make_release.py`, and swallowed the `OSError`.
   Same root cause as the RG6 misattribution fixed the same day; this was the
   other call site. The first manifest and its tag were local and unpushed, so
   the ceremony was re-run rather than shipping a release record that could not
   name its own toolchain.
2. **`record_release`'s out was DVC-cached, not git-tracked.** `dvc commit -f
   record_release` added `/releases` to `data/.gitignore`, which would have made
   the ~4 KB release record unreadable to anyone at the tag without AWS
   credentials. Every sibling audit artifact in `dvc.yaml` is `cache: false`
   (`annotation_qa_report`, `completeness_report`, `coverage_report`,
   `dataset_quality_report`, `verification_ledger`, `eval_report`) and the
   release runbook §5 already said `git add data/releases`, which cannot work
   while the out is cached. Fixed in `dvc.yaml`; recorded in
   [ADR-P5-07](../07_dataset_production/adr/ADR-P5-07-releases-as-code.md).
   `verify_lock_objects.py` now skips **6** `cache: false` outs, up from 5.

### Known limitations shipped with this release

Enumerated in full in `data/DATASET_CHANGELOG.md`. The load-bearing one:
`completeness.json` over-claims trusted classes for **69.5%** of images
(source-level union instead of per-slug), so **this release must not be trained
on with `missing_annotation_mitigation` enabled**. Deferred to `v0.7.0` with
evidence in [ADR-P5-14](../07_dataset_production/adr/ADR-P5-14-per-slug-completeness-policy.md)
and pinned by `tests/unit/test_completeness_policy_granularity.py`.

### Ordering wart, recorded not fixed

`make` requires RG5 green (clean tree, tag at HEAD) and then writes
`data/releases`, dirtying the tree — so the manifest can never describe the
commit that contains it. Concretely for this release:

| | |
|---|---|
| Manifest's recorded `git_commit` | `956d538` — the gated state: dataset, changelog, ADRs, code |
| Tag `dataset-v0.6.0` points at | `9982a78` — that state **plus** the manifest and the `record_release` freeze |

The two differ only by the manifest and its lock entry; no dataset-defining
content changes between them. Re-running `make` to close the gap does not
converge — each regeneration dirties the tree again, one commit behind forever.

The same structure makes `manifest.dvc_lock_sha256` permanently unequal to the
`dvc.lock` at the tag, because the recorded hash predates the
`dvc commit -f record_release` that pins the manifest. `verify` used to warn
*"expected unless you've checked out this release's tag"*, which told the
operator the opposite at the one moment they were most likely to run it; the
message now states the invariant instead of implying a match that cannot occur.

---

## 2026-07-29 — Phase F2/F3: full clean-container pull + `dvc repro qa_check` (Linux)

**The full clean-machine gate (route (a)), executed.** Every limitation recorded
against the 2026-07-27 fetch check is closed: different OS, complete pull, and
`dvc repro qa_check` actually run in the clone.

| | |
|---|---|
| **Machine** | `python:3.12-slim` container on Docker Linux — no repo state, no DVC cache, no `.dvc/config.local` (so the cache defaults to `.dvc/cache`, genuinely cold) |
| **Commit** | `main` @ `f8e469f` (public `git clone --depth 1`, no credentials) |
| **Variant** | Pull-from-remote (`dvc pull -r storage`) + `dvc repro --single-item -f qa_check` |
| **Pull** | **67,734 files fetched / 197,861 added** (~7.4 GB) from S3 |

### Result

| Check | Outcome |
|---|---|
| Dep hashes vs a clean checkout (53 file deps present) | ✅ every recorded hash equals this Linux checkout's bytes — **except** `train_yolo11n` (see finding 2) |
| `download_coco` dep hash — the PR #20 regression | ✅ **no `changed deps`**; the CRLF entry is gone |
| Counts vs committed `split_summary.json` | ✅ 24,352 images / 24,352 labels / leakage 0; train 20,588 · val 1,882 · test 1,882, images == labels in each |
| CRLF on Linux | ✅ 0 across both 24,352-file label trees (`data/processed/labels`, `data/merged/labels`) |
| `dvc repro --single-item -f qa_check` | ✅ rc=0 |
| Regenerated QA report vs the Windows-built one | ✅ identical — 24,352 / 24,352 / **104,289 boxes** / 0 critical, leakage 0/0, sweeps 0/0 |
| `dvc pull` exit code | ❌ **rc=1** (see finding 1) |

**Verdict: the reproducibility claim is proven; one non-data defect found.** A
machine that has never built this dataset reconstructs it byte-for-byte from git
+ S3 alone and regenerates the QA report exactly. The build had existed on one
laptop since 2026-07-14; that gate is now closed.

### Findings

1. **`dvc pull` exits 1 on a clean clone** — `Checkout failed for following
   targets: data/releases`. Not data loss: all 67,734 objects transferred and the
   dataset checked out completely (every check above ran on the pulled data).
   The cause is two stages **declared in `dvc.yaml` but absent from `dvc.lock`**
   because they have never run — `record_release` (`outs: data/releases`) and
   `evaluate_yolo11n` (`outs: data/qa_reports/eval_report.json`), both
   `frozen: true` human-loop stages. `dvc pull` still tries to materialise their
   outs and fails. This matters because the documented gate in this directory's
   README is "`dvc pull && dvc repro qa_check` must succeed", and the first half
   returns non-zero today. F6/F7 (`make dataset-v0.6.0` + `dvc commit -f
   record_release`) removes the `record_release` half; the `evaluate_yolo11n`
   half rides with RG10 training evidence and will keep `dvc pull` at rc=1 until
   then. Recorded, not worked around — suppressing the exit code is exactly the
   false-green pattern this project keeps finding.
2. **`train_yolo11n`'s recorded dep hashes match no committed content.**
   `scripts/training/train_yolo.py` (`708e5442…`) and
   `configs/training/yolo11n_config.yaml` (`6023a982…`) match **neither** the LF
   **nor** the CRLF form of HEAD — so that training run used script content that
   no longer exists in the repo and is not reproducible. Distinct from the PR #20
   CRLF bug. `dvc status` hides it on *every* platform ("stage … is frozen. Its
   dependencies are not going to be shown"), so it surfaces only by reading
   `dvc.lock` directly. Harmless for the dataset — no dataset stage depends on it
   — but it must be cleared before RG10 evidence is recorded, by re-running
   training on current scripts, **not** by re-stamping the hashes.
3. **`qa_check` can never be lock-idempotent.** `annotation_qa_report.json`
   embeds a `timestamp`, so re-running the stage always yields a new out hash and
   rewrites `dvc.lock`. Expected, and *not* OS-dependence — the report content
   compared identical — but it means a re-run always dirties the lock.

### What is still untested

- **Route (b), full rebuild-from-sources on an independent host.** Unchanged from
  2026-07-14: needs unrestricted egress to all four source hosts.
- **A different IAM principal.** Credentials came from the same AWS chain.

### How the checks were designed (and where they were too coarse)

The script's own verdict line printed FAIL. Three of its four failing checks were
the script's fault, not the repo's, and are recorded here so the next run does not
re-litigate them:

- Grepping `dvc status` for `changed deps` **before** the pull matches entries
  reading `deleted: data/raw/…` — those are absent *data*, not disagreeing
  *hashes*. The precise check is the arithmetic one (recompute each recorded md5
  from the checkout's bytes), which is why it was added.
- The same grep post-pull matches `ingest_local_zips` (`deleted: Dataset`, the
  local ZIP inbox) and `dataset_quality_report`
  (`deleted: data/raw/custom_captures/manifests`). Both paths are in neither git
  nor the DVC remote, so a clean clone legitimately lacks them. Windows shows the
  same two stages clean only because those paths exist there.
- Only finding 1 above was a genuine repo defect.

### Commands (as run)

```bash
docker run --rm -v <scratch>:/work:ro -v C:\Users\haris\.aws:/root/.aws:ro python:3.12-slim \
  bash -c 'apt-get install -y git; bash /work/phase_f23_main.sh'
# inside: git clone --depth 1 --branch main <repo> /rc
#         dvc pull -r storage
#         dvc repro --single-item -f qa_check
```

> The `~/.aws` mount must be a **Windows-style path**, and Git Bash needs
> `MSYS_NO_PATHCONV=1`, or it rewrites `~/.aws` to `C:/Program Files/Git/root/.aws`
> and the container starts with no credentials. That produced one bogus FAIL
> before it was diagnosed. Invoking `docker` from PowerShell (as above) avoids the
> rewrite entirely.

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

> **Superseded 2026-07-29.** All three limitations above were closed by the
> Phase F2/F3 run — different OS, full pull, `dvc repro qa_check` executed. The
> figures here (17,888 images; 6.2 GB) describe the pre-Phase-E build and are
> left as-written because this is a dated record, not current-state docs.

---

## 2026-07-14 — Clean-machine rebuild-from-sources (Linux, partial by network policy)

| | |
|---|---|
| **Machine** | Ephemeral Linux container (x86_64, Python 3.11.15) — no prior repo state, no DVC cache, no dataset |
| **Commit** | `wp3.0-platform-remediation` @ `b1c1a84` (fresh `git clone`, separate working copy) |
| **Variant** | Rebuild-from-sources (`dvc repro`), smoke mode. The `dvc pull` variant was **pending** at the time of this run — it required the first `dvc push` from the machine holding the data (see §6 runbook). **That blocker cleared 2026-07-27**: the cache is now on `localstore` and on S3 (46,337 objects / 6.21 GB at that date, `dvc status -c -r storage` in sync). The pull-variant was then **partially executed 2026-07-27** — see "Pull-variant fetch check" above — and **fully executed 2026-07-29**. |
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
   **✅ Closed 2026-07-29.** Both halves were implemented (`newline="\n"` writers;
   `.gitattributes` `* text=auto eol=lf` + `*.py text eol=lf`) — but the
   recommendation as written would not have caught the residual case. One lock
   entry had been recorded under a dep's **CRLF** md5 and stayed there: git
   normalises on `add`, so the drifted working copy reported clean, and the one
   machine whose bytes matched the lock was the machine that wrote it. It took a
   Linux checkout to expose a hash **no** clean checkout on **any** OS could
   produce. Fixed in PR #20 (`c481cfc`) and guarded by
   `tests/unit/test_dvc_lock_portability.py`.
2. **F2 — Sandbox egress policy blocks 2 of 4 source hosts** (`images.cocodataset.org`,
   `huggingface.co`); `storage.googleapis.com` and `open-images-dataset.s3.amazonaws.com`
   are allowed. A **full** rebuild-from-sources needs unrestricted egress (any dev
   machine) or a widened environment network policy.
   *Still open* — the 2026-07-29 run validates the **pull** variant, which does not
   need source egress. A full rebuild-from-sources on an independent host is
   untested.
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

> **Update 2026-07-29:** route (a) was executed — see the Phase F2/F3 entry.
> Route (b) remains untested.

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
