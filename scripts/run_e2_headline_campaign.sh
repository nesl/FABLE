#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${1:-/media/brianw/Extreme SSD2/fable_results/e2_headline_campaign_20260820}"
repetitions="${2:-10}"
snapshot="${repo_root}/final/e2_headline_readiness_20260820/live_runtime_snapshot/s0000-r0000-01a02110-f2ef-704b-b4ee-766a619ac00d.json"

mkdir -p "${output_root}"
cd "${repo_root}"
for repetition in $(seq 1 "${repetitions}"); do
  result_dir="${output_root}/repetition-$(printf '%02d' "${repetition}")"
  if [[ -s "${result_dir}/pilot.json" ]]; then
    continue
  fi
  mkdir -p "${result_dir}"
  started="$(date -Iseconds)"
  if timeout 600s .venv/bin/python scripts/run_e2_contention_pilot.py \
      --output "${result_dir}" \
      --hypotheses 1,2,4,8,16 \
      --beam-width 8 \
      --oracle-max-hypotheses 4 \
      --runtime-snapshot "${snapshot}" \
      >"${result_dir}/campaign.log" 2>&1; then
    status="COMPLETE"
  else
    status="FAILED"
  fi
  printf '%s,%s,%s,%s\n' "${repetition}" "${status}" "${started}" "$(date -Iseconds)" \
    >>"${output_root}/progress.csv"
  [[ "${status}" == "COMPLETE" ]] || exit 1
done
