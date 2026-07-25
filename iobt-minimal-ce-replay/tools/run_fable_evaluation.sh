#!/usr/bin/env bash
set -euo pipefail

: "${FABLE_EVAL_RUN_ID:?set FABLE_EVAL_RUN_ID}"
: "${FABLE_EVAL_TRACE_ID:?set FABLE_EVAL_TRACE_ID}"
: "${FABLE_EVAL_REQUEST_ID:?set FABLE_EVAL_REQUEST_ID}"
: "${FABLE_EVAL_BASELINE:?set FABLE_EVAL_BASELINE}"

exec docker compose \
  -f compose.server.yaml \
  -f compose.replay.yaml \
  -f compose.fable.yaml \
  -f compose.fable.phase7.yaml \
  -f compose.fable.phase8.yaml \
  -f compose.fable.evaluation.yaml \
  up --build
