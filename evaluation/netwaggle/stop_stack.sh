#!/usr/bin/env bash
# Stop NetWaggle runner/collectors, Docker replay/server stacks, and clean stale
# Mininet/veth state.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

STOP_SERVER=1
TOPOLOGY="$DEFAULT_TOPOLOGY"
RUN_DIR=""

usage() {
  cat <<USAGE
usage: $0 [options]

Options:
  --keep-server      Stop replay/NetWaggle but leave MQTT broker + UI running.
  --topology PATH    Topology JSON. Default: $DEFAULT_TOPOLOGY
  --run-dir PATH     Run directory. Default: $CURRENT_LINK
  -h, --help         Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-server) STOP_SERVER=0; shift ;;
    --topology) TOPOLOGY="$(readlink -f "$2")"; shift 2 ;;
    --run-dir) RUN_DIR="$(readlink -f "$2")"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

if [[ -z "$RUN_DIR" && -L "$CURRENT_LINK" ]]; then
  RUN_DIR="$(readlink -f "$CURRENT_LINK")"
fi

sudo -v || true

if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
  log "Collecting final debug snapshot before shutdown"
  "$SCRIPT_DIR/collect_debug_snapshot.sh" --run-dir "$RUN_DIR" --topology "$TOPOLOGY" --no-tar >/dev/null 2>&1 || true
fi

if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
  safe_kill_pid_file "$RUN_DIR/mqtt_trace.pid"
  safe_kill_pid_file "$RUN_DIR/metrics.pid"
  safe_kill_pid_glob "$RUN_DIR/latency_probe_*.pid"
  safe_kill_pid_file "$RUN_DIR/netwaggle_runner.pid"
else
  log "No run directory found; skipping PID-file cleanup."
fi

cd "$REPLAY_DIR"
if [[ -f compose.netwaggle.yaml ]]; then
  log "Stopping compose.netwaggle.yaml"
  docker compose -f compose.netwaggle.yaml down --remove-orphans || true
fi
if (( STOP_SERVER )); then
  log "Stopping compose.server.yaml"
  docker compose -f compose.server.yaml down --remove-orphans || true
fi

log "Cleaning NetWaggle/Mininet state"
(
  cd "$NETWAGGLE_DIR"
  sudo PYTHONPATH=. "$PYTHON_BIN" -m netwaggle.cleanup --topology "$TOPOLOGY" || true
)
sudo mn -c >/dev/null 2>&1 || true
if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
  touch "$RUN_DIR/.closed" || true
  write_run_event "$RUN_DIR" "stop_stack_done keep_server=$((1-STOP_SERVER))" || true
fi
log "Stopped."
