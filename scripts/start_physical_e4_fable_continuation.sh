#!/bin/bash
set -euo pipefail

session=fable-e4-fable
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output='/media/brianw/Extreme SSD2/fable_results/physical_e4_fable_continuation_20260826'
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 2
fi
tmux new-session -d -s "$session" \
  "cd '$repo' && exec ./scripts/run_physical_e4_fable_continuation_worker.sh"
echo "started $session"
echo "monitor: tmux attach -t $session"
echo "log: $output/campaign.log"
