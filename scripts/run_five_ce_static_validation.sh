#!/usr/bin/env bash
set -u

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1

output_root="evaluation/results/rq1_calibrated_template_five_ce_validation_20260806"
traces=(
  20241008-route-convoy-14-r026
  20260414-three-visit-stalking-stalking-30-r030
  20241009-two-vehicle-chase-18-r021
  20241008-vehicle-convergence-1-r004
  20250812-vehicle-rendezvous-brianjulian-1-r026
)
baselines=(B0_PRODUCE_ALL B1_STATIC_WHOLE_EVENT)

for trace in "${traces[@]}"; do
  for baseline in "${baselines[@]}"; do
    echo "RUNNING ${trace} ${baseline}"
    .venv/bin/python scripts/run_full_ce_suite.py \
      --output-dir "${output_root}/${baseline}" \
      --baseline "${baseline}" \
      --max-seconds 300 \
      --ready-seconds 30 \
      --playback-mode realtime \
      --mobile-root "/media/brianw/Extreme SSD3" \
      --experiment-id "${trace}" || true
  done
done
