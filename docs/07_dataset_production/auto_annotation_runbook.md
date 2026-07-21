# Phase-5 M1 Auto-Annotation Runbook — `12_auto_annotate.py`

> Operational SOP for generating L2 candidate labels. Architecture/design:
> ADR-P5-01 (candidate isolation), ADR-P5-02 (backend + determinism).

---

## 0. Environment provisioning

The primary backend (`yolo_world`) needs a CUDA-enabled torch + ultralytics
+ the pinned CLIP text encoder — none of this is in `requirements.txt` (kept
optional, ADR-P5-11):

```bash
pip install -r requirements.txt -r requirements-annotation.txt
```

Verify the `.venv` actually resolves to a CUDA build before relying on GPU
throughput — on Windows especially, a bare `python`/`dvc` on `PATH` can
silently resolve to a *different*, CPU-only interpreter than the project's
`.venv/Scripts/python.exe`:

```bash
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If this ever prints `False`, either the wrong interpreter is being invoked
or the venv genuinely needs a CUDA-matched torch reinstall — do not proceed
to a full-scale run until it prints `True` (the throughput budget, ≥4 img/s,
assumes GPU inference).

## 1. Weight pinning

`configs/annotation.yaml` requires a non-empty `weights_sha256` for every
enabled backend and the optional refinement pass — this is a hard-fail, not
a warning (ADR-P5-02: determinism-within-machine).

```bash
.venv/Scripts/python.exe -c "
from ultralytics import YOLO
YOLO('models/annotators/yolov8x-worldv2.pt')   # downloads if missing
"
.venv/Scripts/python.exe -c "
from ultralytics import SAM
SAM('models/annotators/mobile_sam.pt')
"
```

Then compute and record each digest:

```bash
.venv/Scripts/python.exe -c "
import hashlib
from pathlib import Path
for f in ['models/annotators/yolov8x-worldv2.pt', 'models/annotators/mobile_sam.pt']:
    print(f, hashlib.sha256(Path(f).read_bytes()).hexdigest())
"
```

Paste the digests into `configs/annotation.yaml`'s `weights_sha256` fields.
Weight files themselves stay out of git/DVC (`.gitignore` — reproducible via
download, not something to version).

## 2. Prompts and targeting

`auto_annotation.backends.yolo_world.prompts` lists text prompts per
taxonomy class — an **empty list means the class is never targeted** (L2
scope honesty, ADR-P5-02): custom-capture-only classes (`wet_floor`,
`medicine_strip`, `gas_cylinder`, `walking_stick`, `support_handle`,
`passport`, `stove`) are deliberately absent here — open-vocab models
cannot deliver usable precision on them and junk candidates would flood
human verification (risk R30). `targeting.priority_classes` ranks
verification batches (M2) but does not affect which cells get annotated.

## 3. Run

```bash
.venv/Scripts/python.exe scripts/dataset/12_auto_annotate.py --verify-determinism
```

`--limit N` caps targeted images for a smoke run. `--backend <name>` (may
repeat) overrides the config's enabled-backend selection. Writes one
`data/annotation/candidates/<backend>/candidates.json` per backend —
**never touches `labels/`** (ADR-P5-01).

`--verify-determinism` re-annotates the artifact's first 20 images
(including the refinement pass, if enabled) and diffs — a non-empty diff
exits 1. Two independent process invocations should both pass; if not,
suspect TF32/cuDNN-benchmark nondeterminism before suspecting the model
itself (`_setup_determinism()` already disables both and pins
`CUBLAS_WORKSPACE_CONFIG`).

## 4. DVC

```bash
dvc repro auto_annotate
dvc push
```

Normal (non-frozen) stage — deps include the git-tracked verification
ledger, so masking shrinks exactly as M2 verification grows (re-running
after an import targets fewer cells automatically).

---

Previous: [README.md](./README.md)

Next: [verification_runbook.md](./verification_runbook.md)
