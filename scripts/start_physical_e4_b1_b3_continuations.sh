#!/bin/bash
set -euo pipefail
session=fable-e4-b1-b3
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 2
fi
tmux new-session -d -s "$session" "cd '$repo' && exec ./scripts/run_physical_e4_b1_b3_continuations_worker.sh"
echo "started $session"
echo "monitor: tmux attach -t $session"
