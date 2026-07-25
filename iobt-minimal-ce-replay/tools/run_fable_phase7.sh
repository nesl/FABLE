#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose \
  -f compose.server.yaml \
  -f compose.replay.yaml \
  -f compose.fable.yaml \
  -f compose.fable.phase7.yaml \
  up --build "$@"
