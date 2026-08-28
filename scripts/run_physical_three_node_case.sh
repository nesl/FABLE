#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
identity_file="${FABLE_PHYSICAL_IDENTITY_FILE:-/tmp/fable_deploy_key}"
output_dir="${FABLE_PHYSICAL_OUTPUT_DIR:-${repo_root}/evaluation/results/physical_three_node_route_convoy_r012}"
physical_replay_node="${FABLE_PHYSICAL_REPLAY_NODE:-orin7}"

if [[ ! -f "${identity_file}" ]]; then
  echo "Missing SSH identity: ${identity_file}" >&2
  echo "Set FABLE_PHYSICAL_IDENTITY_FILE to a key authorized on both devices." >&2
  exit 2
fi

cd "${repo_root}"
exec .venv/bin/python scripts/run_full_ce_suite.py \
  --experiment-id 20241008-route-convoy-1-r012 \
  --output-dir "${output_dir}" \
  --baseline FABLE \
  --playback-mode realtime \
  --max-seconds 240 \
  --ready-seconds 60 \
  --replay-node orin1 \
  --replay-node orin4 \
  --replay-node orin7 \
  --maximum-replay-nodes 3 \
  --stage-physical-rpi \
  --execute-physical-rpi \
  --physical-rpi-replay-node "${physical_replay_node}" \
  --physical-rpi-host rpi \
  --physical-jetson-host jetson \
  --physical-rpi-identity-file "${identity_file}"
