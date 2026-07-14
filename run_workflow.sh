#!/bin/bash
# Elderly Assistant System — dataset pipeline entry point.
#
# The DVC DAG (dvc.yaml) is the single orchestration path:
#   download → remap → merge → split → QA
# This wrapper exists only for discoverability; it is equivalent to
# running `dvc repro` directly. The training stage stays frozen until
# Phase-5; inference tooling lives in scripts/inference/.

set -euo pipefail

echo "============================================================"
echo " Elderly Assistant System — Dataset Pipeline (dvc repro)"
echo "============================================================"

dvc repro "$@"

echo ""
echo "QA metric:"
dvc metrics show

echo ""
echo "Pipeline execution finished."
