#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
experiment="20260415-pass-follow-clear-convoy-3-car-convoy-ce1-9-r009"
condition="$repo_root/evaluation/manifests/adaptation/rq3a_single_disconnect_full/disconnect_${experiment}_post_eof.json"
topology="$repo_root/netwaggle/configs/site_evaluation_29node.json"
output_root="${FABLE_PILOT_OUTPUT_ROOT:-/media/brianw/Extreme SSD2/fable_results/b1_fable_post_eof_disconnect_20260807}"

test -S /run/netwaggle/fable-control.sock || {
  echo "missing NetWaggle control socket" >&2
  exit 2
}
test -w "$(dirname "$output_root")" || {
  echo "output parent is not writable: $(dirname "$output_root")" >&2
  exit 2
}

for baseline in B1_STATIC_WHOLE_EVENT FABLE; do
  timeout --signal=INT --kill-after=20s 320 \
    "$repo_root/.venv/bin/python" "$repo_root/scripts/run_full_ce_suite.py" \
      --output-dir "$output_root/$baseline" \
      --experiment-id "$experiment" \
      --baseline "$baseline" \
      --playback-mode realtime \
      --max-seconds 210 \
      --ready-seconds 45 \
      --netwaggle-topology "$topology" \
      --require-netwaggle-bindings \
      --drop-offline-evidence \
      --close-live-evidence-at-replay-end \
      --allow-raw-to-trusted-site-edge \
      --condition-trace "$condition"
done
