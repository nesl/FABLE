#!/usr/bin/env bash
# Convenience wrapper for starting replay control against the host-side broker.
# The replay containers themselves connect to MQTT through NetWaggle.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SCENARIO=""
START="0"
END="30"
PLAYBACK_MODE="max"
SYNC_DELAY="2"

usage() {
  cat <<USAGE
usage: $0 --scenario SCENARIO_ID [options]

Options:
  --start SEC          Replay start offset. Default: $START
  --end SEC            Replay end offset. Default: $END
  --playback-mode MODE max, realtime, or scaled. Default: $PLAYBACK_MODE
  --sync-delay SEC     Sync delay. Default: $SYNC_DELAY
  -h, --help           Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario) SCENARIO="$2"; shift 2 ;;
    --start) START="$2"; shift 2 ;;
    --end) END="$2"; shift 2 ;;
    --playback-mode) PLAYBACK_MODE="$2"; shift 2 ;;
    --sync-delay) SYNC_DELAY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

[[ -n "$SCENARIO" ]] || { usage; fail "--scenario is required"; }

cd "$REPLAY_DIR"
"$PYTHON_BIN" tools/replay_control.py \
  --mqtt-host localhost \
  --scenario "$SCENARIO" \
  --start "$START" \
  --end "$END" \
  --playback-mode "$PLAYBACK_MODE" \
  --sync-delay "$SYNC_DELAY"
