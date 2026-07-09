#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOPOLOGY="${1:-$ROOT/configs/fable_single_host.json}"
PROFILE="${2:-$ROOT/configs/profiles/good_network.json}"

sudo PYTHONPATH="$ROOT" python3 -m netwaggle.runner \
  --topology "$TOPOLOGY" \
  --profile "$PROFILE" \
  --no-cli \
  --hold
