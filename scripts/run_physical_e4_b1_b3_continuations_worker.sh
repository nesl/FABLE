#!/bin/bash
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base='/media/brianw/Extreme SSD2/fable_results'
b1="$base/physical_e4_b1_continuation_20260826"
b3="$base/physical_e4_b3_continuation_20260826"
final="$repo/final/physical_e4_consolidated_20260826"
master="$base/physical_e4_b1_b3_continuations_20260826.log"
cd "$repo"

cleanup() {
  .venv/bin/python scripts/physical_condition_control.py restore-network \
    --identity-file /tmp/fable_deploy_key --target rpi_to_jetson --execute >>"$master" 2>&1 || true
  .venv/bin/python scripts/physical_condition_control.py clear-compute \
    --identity-file /tmp/fable_deploy_key --target physical_jetson --execute >>"$master" 2>&1 || true
}
trap cleanup EXIT INT TERM
export FABLE_CONFIRM_EXPERIMENT=YES

mkdir -p "$b1" "$b3" "$final"
.venv/bin/python scripts/run_physical_e4_multitrace.py \
  --manifest evaluation/manifests/adaptation/physical_e4_b1_continuation.json \
  --output "$b1" --hard-cell-timeout 600 --execute >>"$master" 2>&1
echo "B1_RETURN=$?" >>"$master"
cleanup
.venv/bin/python scripts/run_physical_e4_multitrace.py \
  --manifest evaluation/manifests/adaptation/physical_e4_b3_continuation.json \
  --output "$b3" --hard-cell-timeout 600 --execute >>"$master" 2>&1
echo "B3_RETURN=$?" >>"$master"
cleanup
.venv/bin/python scripts/report_physical_e4_consolidated.py --output "$final" \
  >"$final/report.stdout.json" 2>"$final/report.stderr.log"
echo "REPORT_RETURN=$?" >>"$master"
