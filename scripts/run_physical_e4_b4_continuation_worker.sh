#!/bin/bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output='/media/brianw/Extreme SSD2/fable_results/physical_e4_b4_continuation_20260826'
log="$output/campaign.log"
mkdir -p "$output"
cd "$repo"

cleanup() {
  .venv/bin/python scripts/physical_condition_control.py restore-network \
    --identity-file /tmp/fable_deploy_key --target rpi_to_jetson --execute \
    >>"$log" 2>&1 || true
  .venv/bin/python scripts/physical_condition_control.py clear-compute \
    --identity-file /tmp/fable_deploy_key --target physical_jetson --execute \
    >>"$log" 2>&1 || true
}
trap cleanup EXIT INT TERM

export FABLE_CONFIRM_EXPERIMENT=YES
flock -n "$output/campaign.lock" \
  .venv/bin/python scripts/run_physical_e4_multitrace.py \
    --manifest evaluation/manifests/adaptation/physical_e4_b4_continuation.json \
    --output "$output" --hard-cell-timeout 600 --execute \
  2>&1 | tee -a "$log"
