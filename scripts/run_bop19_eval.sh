#!/usr/bin/env bash
# Official BOP19 evaluation (AR = mean of VSD, MSSD, MSPD recall) of the YCB-V comparison
# results, using the bop_toolkit in external/bop_toolkit (own venv, headless EGL rendering).
#
# Prerequisites (one-off):
#   git clone https://github.com/thodan/bop_toolkit external/bop_toolkit
#   python3.9 -m venv external/bop_toolkit/.venv && external/bop_toolkit/.venv/bin/python -m pip install -e external/bop_toolkit
#   a BOP_PATH root with ycbv/{models_eval,test,test_targets_bop19.json,...} (symlinks are fine)
#
# Usage: scripts/run_bop19_eval.sh [BOP_PATH] [results.csv]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BOP_PATH="${1:-$HOME/Munka/Label_Factory_SFM_2/BOP_toolkit_root}"
CSV="${2:-$ROOT/evaluation/ycbv/bop_results_labelfactory_all.csv}"
WORK="$ROOT/evaluation/bop_toolkit"
TK="$ROOT/external/bop_toolkit"
mkdir -p "$WORK/results" "$WORK/eval"
cp "$CSV" "$WORK/results/labelfactory_ycbv-test.csv"
rm -rf "$WORK/eval/labelfactory_ycbv-test"
# The toolkit starts its workers with the 'python' found on PATH, so the venv must come first.
# Headless rendering: NVIDIA EGL vendor (the default vendor fails to initialise without a display).
PATH="$TK/.venv/bin:$PATH" BOP_PATH="$BOP_PATH" BOP_RESULTS_PATH="$WORK/results" \
BOP_EVAL_PATH="$WORK/eval" \
__EGL_VENDOR_LIBRARY_FILENAMES="${EGL_VENDOR_FILE:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}" \
env -u DISPLAY python "$TK/scripts/eval_bop19_pose.py" --renderer_type=vispy \
  --result_filenames=labelfactory_ycbv-test.csv --num_workers="${WORKERS:-6}" | tee "$WORK/eval_log.txt"
cat "$WORK/eval/labelfactory_ycbv-test/scores_bop19.json"
