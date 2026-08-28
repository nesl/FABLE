#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="$repo_root/evaluation/manifests/adaptation/rq3_post_eof_disconnect_full/rq3_network_post_eof_full_830.jsonl"
default_output="/media/brianw/Extreme SSD2/fable_results/rq3_network_post_eof_full_830_v1_$(date +%Y%m%d)"
output="${FABLE_RQ3_OUTPUT_ROOT:-$default_output}"
topology="$repo_root/netwaggle/configs/site_evaluation_29node.json"
mobile_root="/media/brianw/Extreme SSD3"

test -S /run/netwaggle/fable-control.sock || {
    echo "missing NetWaggle control socket" >&2
    exit 2
}
test -r "$manifest" || {
    echo "missing post-EOF manifest: $manifest" >&2
    exit 2
}
test -w "$(dirname "$output")" || {
    echo "output parent is unavailable or not writable: $(dirname "$output")" >&2
    exit 2
}

cd "$repo_root"
exec .venv/bin/python scripts/run_planned_ce_campaign.py \
    --manifest "$manifest" \
    --output-dir "$output" \
    --max-seconds 900 \
    --ready-seconds 45 \
    --mobile-root "$mobile_root" \
    --netwaggle-topology "$topology" \
    --require-netwaggle-bindings \
    --drop-offline-evidence \
    --close-live-evidence-at-replay-end \
    --allow-raw-to-trusted-site-edge \
    --execution-order ce-round-robin \
    --condition-order disturbed-first
