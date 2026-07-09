#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOPOLOGY="${1:-$ROOT/configs/fable_single_host.json}"

sudo PYTHONPATH="$ROOT" python3 -m netwaggle.cleanup --topology "$TOPOLOGY"
