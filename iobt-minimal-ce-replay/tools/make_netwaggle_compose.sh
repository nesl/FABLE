#!/usr/bin/env bash
set -euo pipefail

REPLAY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NETWAGGLE_ROOT="$(cd "$REPLAY_ROOT/../netwaggle" && pwd)"

COMPOSE_IN="${1:-$REPLAY_ROOT/compose.replay.yaml}"
COMPOSE_OUT="${2:-$REPLAY_ROOT/compose.netwaggle.yaml}"
NODE_MAP="${3:-$NETWAGGLE_ROOT/configs/fable_single_host.json}"

python3 "$NETWAGGLE_ROOT/scripts/make_netwaggle_compose.py" \
  --compose-in "$COMPOSE_IN" \
  --compose-out "$COMPOSE_OUT" \
  --node-map "$NODE_MAP"
