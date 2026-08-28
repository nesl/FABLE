#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
experiment="20260415-pass-follow-clear-convoy-3-car-convoy-ce1-9-r009"
condition="$repo_root/evaluation/manifests/adaptation/rq3a_single_disconnect_full/disconnect_${experiment}_decisive.json"
topology="$repo_root/netwaggle/configs/site_evaluation_29node.json"
default_output="/media/brianw/Extreme SSD2/fable_results/b1_fable_disconnect_pilot_20260807"
output_root="${FABLE_PILOT_OUTPUT_ROOT:-$default_output}"

if [[ ! -S /run/netwaggle/fable-control.sock ]]; then
  echo "missing NetWaggle control socket: /run/netwaggle/fable-control.sock" >&2
  exit 2
fi
if [[ ! -d "$(dirname "$output_root")" || ! -w "$(dirname "$output_root")" ]]; then
  echo "output parent unavailable or not writable: $(dirname "$output_root")" >&2
  exit 2
fi

run_cell() {
  local baseline="$1"
  local condition_name="$2"
  local -a condition_args=()
  if [[ "$condition_name" == "SENSOR_DISCONNECT" ]]; then
    condition_args=(--condition-trace "$condition")
  fi
  timeout --signal=INT --kill-after=20s 210 \
    "$repo_root/.venv/bin/python" "$repo_root/scripts/run_full_ce_suite.py" \
      --output-dir "$output_root/$condition_name/$baseline" \
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

for condition_name in N0 SENSOR_DISCONNECT; do
  for baseline in B1_STATIC_WHOLE_EVENT FABLE; do
    run_cell "$baseline" "$condition_name"
  done
done
