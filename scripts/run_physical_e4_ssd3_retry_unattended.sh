#!/bin/bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output='/media/brianw/Extreme SSD2/fable_results/physical_e4_ssd3_retry_20260825'
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
    --manifest evaluation/manifests/adaptation/physical_e4_multitrace.json \
    --output "$output" --hard-cell-timeout 600 --execute \
    --experiment-id 20241008-route-convoy-5-r016 \
    --experiment-id 20241008-route-convoy-6-r018 \
    --experiment-id 20241008-route-convoy-7-r019 \
    --experiment-id 20241008-route-convoy-8-r020 \
    --experiment-id 20241008-route-convoy-9-r021 \
    --experiment-id 20241008-route-convoy-10-r022 \
    --experiment-id 20241008-vehicle-convergence-1-r004 \
    --experiment-id 20241008-vehicle-convergence-2-r005 \
    --experiment-id 20241008-vehicle-convergence-3-r006 \
    --experiment-id 20241008-vehicle-convergence-4-r007 \
    --experiment-id 20241008-vehicle-convergence-5-r008 \
  2>&1 | tee -a "$log"
