#!/usr/bin/env bash
# Terminal 1: run only the host-side MQTT broker and web UI in the foreground.
# This keeps UI/server logs visible while NetWaggle and replay run elsewhere.
# The same log stream is also saved under runs/netwaggle/current/logs/.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BUILD=1
NEW_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build) BUILD=0; shift ;;
    --new-run)
      rm -f "$CURRENT_LINK"
      NEW_RUN=1
      shift ;;
    -h|--help)
      cat <<USAGE
usage: $0 [--no-build] [--new-run]

Runs compose.server.yaml in the foreground. Open http://localhost:8080.
All logs/config snapshots are saved under runs/netwaggle/current/.
Use --new-run to force a fresh run directory before starting Terminal 1.
USAGE
      exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

RUN_DIR="$(ensure_run_dir server_ui_mqtt)"
write_run_event "$RUN_DIR" "terminal1_start build=$BUILD new_run=$NEW_RUN"

cd "$REPLAY_DIR"
save_compose_rendered "$RUN_DIR" compose.server.yaml server
save_docker_ps "$RUN_DIR" before_server

cleanup() {
  local rc=$?
  write_run_event "$RUN_DIR" "terminal1_exit rc=$rc"
  save_compose_logs "$RUN_DIR" compose.server.yaml server
  save_container_inspect_for_compose "$RUN_DIR" compose.server.yaml server
  save_docker_ps "$RUN_DIR" after_server
  save_webui_api_state "$RUN_DIR"
  exit "$rc"
}
trap cleanup EXIT

args=(-f compose.server.yaml up --timestamps)
if (( BUILD )); then args+=(--build); fi

log "Terminal 1: starting MQTT + web UI. Open http://localhost:8080"
log "Run directory: $RUN_DIR"
log "Saved live log: $RUN_DIR/logs/terminal1_server_ui_mqtt.log"
log "Stop with Ctrl-C. Final compose logs/API snapshots will be saved automatically."

set +e
docker compose "${args[@]}" 2>&1 | tee -a "$RUN_DIR/logs/terminal1_server_ui_mqtt.log"
exit_code=${PIPESTATUS[0]}
set -e
exit "$exit_code"
