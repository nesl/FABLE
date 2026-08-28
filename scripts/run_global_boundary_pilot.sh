#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
experiment="20260415-pass-follow-clear-convoy-3-car-convoy-ce1-9-r009"
condition="$repo_root/evaluation/manifests/adaptation/rq3a_single_disconnect_full/disconnect_${experiment}.json"
topology="$repo_root/netwaggle/configs/site_evaluation_29node.json"
output_root="$repo_root/evaluation/results/global_boundary_pilot_20260807"

run_cell() {
  local baseline="$1"
  local condition_name="$2"
  local output="$output_root/$condition_name/$baseline"
  local -a condition_args=()
  if [[ "$condition_name" == "disconnect" ]]; then
    condition_args=(--condition-trace "$condition")
  fi
  timeout --signal=INT --kill-after=20s 210 \
    "$repo_root/.venv/bin/python" "$repo_root/scripts/run_full_ce_suite.py" \
      --output-dir "$output" \
      --experiment-id "$experiment" \
      --baseline "$baseline" \
      --playback-mode realtime \
      --max-seconds 120 \
      --ready-seconds 45 \
      --netwaggle-topology "$topology" \
      --require-netwaggle-bindings \
      --drop-offline-evidence \
      --close-live-evidence-at-replay-end \
      --allow-raw-to-trusted-site-edge \
      "${condition_args[@]}"
}

run_cell B1_STATIC_WHOLE_EVENT nominal
run_cell FABLE nominal
run_cell B1_STATIC_WHOLE_EVENT disconnect
run_cell FABLE disconnect

