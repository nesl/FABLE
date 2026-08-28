#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="$repo_root/evaluation/manifests/adaptation/rq3a_single_disconnect_full/rq3_network_single_disconnect_full_90.jsonl"
default_output="/media/brianw/Extreme SSD2/fable_results/rq3_network_single_disconnect_full_$(date +%Y%m%d)"
output="${FABLE_RQ3_OUTPUT_ROOT:-$default_output}"
topology="$repo_root/netwaggle/configs/site_evaluation_29node.json"
mobile_root="/media/brianw/Extreme SSD3"

if [[ ! -S /run/netwaggle/fable-control.sock ]]; then
    echo "missing NetWaggle control socket: /run/netwaggle/fable-control.sock" >&2
    exit 2
fi
if [[ ! -r "$manifest" ]]; then
    echo "missing campaign manifest: $manifest" >&2
    exit 2
fi
if [[ ! -d "$(dirname "$output")" || ! -w "$(dirname "$output")" ]]; then
    echo "output parent is unavailable or not writable: $(dirname "$output")" >&2
    echo "set FABLE_RQ3_OUTPUT_ROOT to an available output directory" >&2
    exit 2
fi

cd "$repo_root"
exec .venv/bin/python scripts/run_planned_ce_campaign.py \
    --manifest "$manifest" \
    --output-dir "$output" \
    --max-seconds 600 \
    --ready-seconds 45 \
    --mobile-root "$mobile_root" \
    --netwaggle-topology "$topology" \
    --require-netwaggle-bindings \
    --drop-offline-evidence \
    --close-live-evidence-at-replay-end \
    --allow-raw-to-trusted-site-edge \
    --execution-order ce-round-robin
