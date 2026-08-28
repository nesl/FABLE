#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
identity_file="${FABLE_PHYSICAL_IDENTITY_FILE:-/tmp/fable_deploy_key}"
output_dir="${FABLE_PHYSICAL_OUTPUT_DIR:-${repo_root}/evaluation/results/physical_causal_orin7}"
experiment_id="${FABLE_PHYSICAL_EXPERIMENT_ID:-20241008-route-convoy-2-r013}"

if [[ "${FABLE_CONFIRM_EXPERIMENT:-}" != "YES" ]]; then
  echo "Prepared only; set FABLE_CONFIRM_EXPERIMENT=YES to run the causal physical experiment." >&2
  exit 2
fi

cd "${repo_root}"
exec .venv/bin/python scripts/run_full_ce_suite.py \
  --experiment-id "${experiment_id}" \
  --output-dir "${output_dir}" \
  --baseline FABLE \
  --playback-mode realtime \
  --max-seconds 240 \
  --ready-seconds 60 \
  --replay-node orin7 \
  --maximum-replay-nodes 1 \
  --stage-physical-rpi \
  --execute-physical-rpi \
  --physical-rpi-replay-node orin7 \
  --physical-rpi-host rpi \
  --physical-jetson-host jetson \
  --physical-rpi-identity-file "${identity_file}"
