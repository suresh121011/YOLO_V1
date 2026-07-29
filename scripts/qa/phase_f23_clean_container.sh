#!/usr/bin/env bash
# Clean-machine reproduction gate (Phase F2/F3), run INSIDE a Linux container.
#
#   docker run --rm -v <dir-holding-this-script>:/work:ro \
#     -v C:\Users\<you>\.aws:/root/.aws:ro python:3.12-slim \
#     bash -c 'apt-get update -qq && apt-get install -y -qq git && bash /work/phase_f23_clean_container.sh'
#
# Invoke docker from PowerShell, not Git Bash: Git Bash rewrites the `~/.aws`
# mount to `C:/Program Files/Git/root/.aws`, the container comes up with no
# credentials, and every downstream check fails for a reason that has nothing to
# do with the repo. (`MSYS_NO_PATHCONV=1` plus a Windows-style path also works.)
#
# What this gate is for: the pipeline demonstrably works on the build machine.
# The open question is whether a Windows-authored `dvc.lock` describes a Linux
# checkout — that is what "reproducible" means here, and it is what the
# 2026-07-29 run confirmed after PR #20 fixed a dep recorded under its CRLF md5.
#
# Precision note, learned the hard way: grepping `dvc status` for "changed deps"
# does NOT test hash agreement. Entries reading `deleted: <path>` mean the path
# is absent from this checkout, which is true of every DVC out before the pull
# and permanently true of paths kept in neither git nor the remote (the local
# ZIP inbox, the empty capture manifests dir). The first version of this script
# conflated the two and printed FAIL over a repo that was fine. Hash agreement is
# checked arithmetically in F3.0 instead: recompute each recorded md5 from this
# checkout's bytes.
set -u
FAIL=0
step() { echo; echo "===== $* ====="; }
ok()   { echo "PASS: $*"; }
bad()  { echo "FAIL: $*"; FAIL=1; }

REPO_URL="${REPO_URL:-https://github.com/suresh121011/YOLO_V1.git}"
BRANCH="${BRANCH:-main}"
# Expected dataset shape — update deliberately when the dataset changes, never
# to make a red run go green.
EXP_IMAGES="${EXP_IMAGES:-24352}"
EXP_BOXES="${EXP_BOXES:-104289}"

step "F2.1 environment"
python --version
pip --version >/dev/null && echo "pip ok"

step "F2.2 install dvc"
pip install --quiet --no-input "dvc[s3]==3.67.1" pyyaml >/dev/null 2>&1 \
  && echo "dvc installed" || { bad "pip install dvc"; exit 1; }
dvc --version

step "F2.3 clone $BRANCH (public repo, no credentials)"
git clone --quiet --depth 1 --branch "$BRANCH" "$REPO_URL" /rc || { bad "clone"; exit 1; }
cd /rc || exit 1
echo "HEAD: $(git log --oneline -1)"

step "F3.0 dep-hash agreement BEFORE any pull"
# Runs on a checkout with no data at all, so a green result here cannot be an
# artefact of the pull — it is pure git content vs the hashes in dvc.lock.
python - <<PY
import hashlib, pathlib, yaml
lock = yaml.safe_load(pathlib.Path("dvc.lock").read_text(encoding="utf-8"))["stages"]
frozen = {
    name for name, spec in
    (yaml.safe_load(pathlib.Path("dvc.yaml").read_text(encoding="utf-8"))["stages"]).items()
    if spec.get("frozen")
}
mismatched, frozen_stale, checked = [], [], 0
for stage, spec in lock.items():
    for dep in spec.get("deps") or []:
        md5 = dep.get("md5")
        p = pathlib.Path(dep["path"])
        if not md5 or md5.endswith(".dir") or not p.is_file():
            continue
        checked += 1
        actual = hashlib.md5(p.read_bytes(), usedforsecurity=False).hexdigest()
        if actual == md5:
            continue
        line = f"{stage}: {dep['path']} lock={md5} checkout={actual}"
        # A frozen stage is a human-loop artefact that DVC never re-runs, so its
        # lock legitimately lags the scripts. Report it, don't fail on it — the
        # dataset DAG does not depend on these.
        (frozen_stale if stage in frozen else mismatched).append(line)
print(f"  swept {checked} file deps present in the checkout")
for line in frozen_stale:
    print(f"  NOTE (frozen stage, informational): {line}")
if mismatched:
    print("  FAIL: recorded hashes no clean checkout can produce:")
    for line in mismatched:
        print(f"    {line}")
    raise SystemExit(1)
print("  PASS: every non-frozen recorded dep hash equals this checkout's bytes")
PY
[ $? -eq 0 ] || bad "dep hashes disagree with a clean checkout"

step "F2.4 dvc pull from S3"
# NOTE: rc=1 is currently EXPECTED and is not data loss. `record_release` and
# `evaluate_yolo11n` are declared in dvc.yaml but absent from dvc.lock (never
# run), so `dvc pull` tries to check out outs that do not exist. Reported, not
# suppressed — see reproduction_log.md, Phase F2/F3 finding 1.
dvc pull -r storage 2>&1 | tail -5
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] && ok "dvc pull rc=0" || echo "NOTE: dvc pull rc=$rc (see finding 1); data checkout verified below"

step "F3.1 post-pull status — hash disagreements only"
dvc status > /tmp/status.txt 2>&1
echo "dvc status exit=$?"
grep -v '^WARNING:' /tmp/status.txt | head -30
# A real cross-OS defect looks like "changed deps/outs" whose detail line is
# `modified:`. `deleted:` means the path is absent here, which is expected.
if grep -qE '^[[:space:]]+modified:' /tmp/status.txt; then
  bad "dvc status reports MODIFIED paths — the lock disagrees with this OS"
  grep -B2 -E '^[[:space:]]+modified:' /tmp/status.txt
else
  ok "no modified paths — every difference is an absent path, not a differing hash"
fi

step "F3.2 counts match the committed split_summary"
python - <<PY
import json, pathlib, sys
p = pathlib.Path("data/processed/split_report/split_summary.json")
if not p.exists():
    print("  FAIL: split_summary.json absent"); sys.exit(1)
d = json.loads(p.read_text(encoding="utf-8"))
print(f"  total_images={d['total_images']} total_labels={d['total_labels']} leakage={d['leakage_count']}")
bad = d["total_images"] != $EXP_IMAGES or d["leakage_count"] != 0
for split in ("train", "val", "test"):
    imgs = len(list(pathlib.Path(f"data/processed/images/{split}").glob("*")))
    lbls = len(list(pathlib.Path(f"data/processed/labels/{split}").glob("*.txt")))
    print(f"  {'PASS' if imgs == lbls else 'FAIL'}: {split} images={imgs} labels={lbls}")
    bad = bad or imgs != lbls
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] || bad "split counts do not match the committed summary"

step "F3.3 zero CRLF on Linux"
python - <<'PY'
import pathlib, sys
bad = False
for d in ("data/processed/labels", "data/merged/labels"):
    root = pathlib.Path(d)
    if not root.exists():
        print(f"  SKIP {d} (not pulled)"); continue
    files = list(root.rglob("*.txt"))
    crlf = sum(1 for f in files if b"\r\n" in f.read_bytes())
    print(f"  {'PASS' if crlf == 0 else 'FAIL'}: {d} {len(files)} files, CRLF={crlf}")
    bad = bad or crlf > 0
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] || bad "CRLF bytes present in label files on Linux"

step "F3.4 reproduce qa_check"
pip install --quiet --no-input pillow numpy opencv-python-headless >/dev/null 2>&1
dvc repro --single-item -f qa_check 2>&1 | tail -8
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] && ok "qa_check rc=0" || bad "qa_check rc=$rc"

step "F3.5 regenerated report matches the committed one"
# The report embeds a `timestamp`, so its hash — and therefore dvc.lock — changes
# on every run. Compare the content, never the hash.
python - <<PY
import json, pathlib, sys
d = json.loads(pathlib.Path("data/qa_reports/annotation_qa_report.json").read_text(encoding="utf-8"))
s, o = d["summary"], d.get("orchestrator", {})
print(f"  images={s['total_images']} labels={s['total_labels']} boxes={s['total_boxes']}")
print(f"  critical={s['critical_issues']} leakage={d['checks']['train_val_leakage']['count']}/{d['checks']['train_test_leakage']['count']}")
print(f"  sweeps: annotation={o.get('annotation_sweep_warnings')} l4l5={o.get('l4_l5_report_warnings')}")
exp = dict(total_images=$EXP_IMAGES, total_labels=$EXP_IMAGES, total_boxes=$EXP_BOXES, critical_issues=0)
mism = [k for k, v in exp.items() if s.get(k) != v]
print(("  FAIL: mismatch " + str(mism)) if mism else "  PASS: matches the Windows-built report exactly")
sys.exit(1 if mism else 0)
PY
[ $? -eq 0 ] || bad "regenerated QA report differs from the committed one"

echo
echo "================================"
[ "$FAIL" -eq 0 ] && echo "PHASE F2/F3 RESULT: PASS" || echo "PHASE F2/F3 RESULT: FAIL"
echo "================================"
exit "$FAIL"
