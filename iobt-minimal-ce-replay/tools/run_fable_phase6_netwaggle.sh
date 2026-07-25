#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose \
  -f compose.server.yaml \
  -f compose.netwaggle.yaml \
  -f compose.fable.yaml \
  -f compose.fable.netwaggle.yaml \
  up --build "$@"
