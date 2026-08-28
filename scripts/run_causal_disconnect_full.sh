#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="$repo_root/evaluation/manifests/adaptation/causal_disconnect_full/causal_disconnect_full_830.jsonl"
topology="$repo_root/netwaggle/configs/site_evaluation_29node.json"
default_output="/media/brianw/Extreme SSD2/fable_results/causal_disconnect_full_830_v1_$(date +%Y%m%d)"
output="${FABLE_CAUSAL_DISCONNECT_OUTPUT_ROOT:-$default_output}"

test -S /run/netwaggle/fable-control.sock || {
  echo "missing NetWaggle control socket" >&2
  exit 2
}
test -r "$manifest" || {
  echo "missing causal-disconnect manifest: $manifest" >&2
  exit 2
}
mkdir -p "$output"

cd "$repo_root"
exec .venv/bin/python scripts/run_planned_ce_campaign.py \
  --manifest "$manifest" \
  --output-dir "$output" \
  --max-seconds 900 \
  --ready-seconds 45 \
  --mobile-root "/media/brianw/Extreme SSD3" \
  --netwaggle-topology "$topology" \
  --require-netwaggle-bindings \
  --drop-offline-evidence \
  --close-live-evidence-at-replay-end \
  --allow-raw-to-trusted-site-edge \
  --execution-order ce-round-robin \
  --condition-order disturbed-first
