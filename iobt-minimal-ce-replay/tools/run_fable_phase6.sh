#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose \
  -f compose.server.yaml \
  -f compose.replay.yaml \
  -f compose.fable.yaml \
  up --build "$@"
