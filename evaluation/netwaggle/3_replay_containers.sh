#!/usr/bin/env bash
# Terminal 3: run replay/detector/CE containers in the foreground with verbose
# publish logging enabled. Use the web UI to start/stop replay; this terminal
# shows data/status messages emitted by replay services and saves the same logs
# under runs/netwaggle/current/logs/.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BUILD=1
REGENERATE_COMPOSE=0
SERVICES=()

usage() {
  cat <<USAGE
usage: $0 [options] [service ...]

Options:
  --regenerate-compose     Regenerate compose.netwaggle.yaml before starting.
  --no-build               Do not build images.
  --quiet-publish-logs     Disable [publish:net]/[publish:local] log lines.
  -h, --help               Show this help.

If no services are supplied, starts all non-anchor replay/detector/CE services.
Use the web UI at http://localhost:8080 to start replay.
All live logs and final Docker/Compose snapshots are saved under runs/netwaggle/current/.
USAGE
}

export IOBT_LOG_NET_PUBLISH="${IOBT_LOG_NET_PUBLISH:-true}"
export IOBT_LOG_LOCAL_PUBLISH="${IOBT_LOG_LOCAL_PUBLISH:-true}"
export IOBT_LOG_NET_PUBLISH_EVERY_N="${IOBT_LOG_NET_PUBLISH_EVERY_N:-1}"
export IOBT_LOG_LOCAL_PUBLISH_EVERY_N="${IOBT_LOG_LOCAL_PUBLISH_EVERY_N:-30}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --regenerate-compose) REGENERATE_COMPOSE=1; shift ;;
    --no-build) BUILD=0; shift ;;
    --quiet-publish-logs)
      export IOBT_LOG_NET_PUBLISH=false
      export IOBT_LOG_LOCAL_PUBLISH=false
      shift ;;
    -h|--help) usage; exit 0 ;;
    --*) fail "Unknown argument: $1" ;;
    *) SERVICES+=("$1"); shift ;;
  esac
done

RUN_DIR="$(ensure_run_dir replay_containers)"
write_run_event "$RUN_DIR" "terminal3_start build=$BUILD regenerate=$REGENERATE_COMPOSE net_publish=$IOBT_LOG_NET_PUBLISH local_publish=$IOBT_LOG_LOCAL_PUBLISH"

cd "$REPLAY_DIR"
if (( REGENERATE_COMPOSE )) || [[ ! -f compose.netwaggle.yaml ]]; then
  if [[ ! -f compose.replay.yaml ]]; then
    "$PYTHON_BIN" setup/generate_replay_compose.py --device orin11 --compose-out compose.replay.yaml
  fi
  "$PYTHON_BIN" ../netwaggle/scripts/make_netwaggle_compose.py \
    --compose-in compose.replay.yaml \
    --compose-out compose.netwaggle.yaml \
    --node-map "$DEFAULT_TOPOLOGY"
fi

save_compose_rendered "$RUN_DIR" compose.netwaggle.yaml netwaggle
save_docker_ps "$RUN_DIR" before_replay

if [[ ${#SERVICES[@]} -eq 0 ]]; then
  mapfile -t SERVICES < <(docker compose -f compose.netwaggle.yaml config --services | grep -v '^netwaggle-node-')
fi

BUILD_ARGS=()
if (( BUILD )); then BUILD_ARGS+=(--build); fi

cleanup() {
  local rc=$?
  write_run_event "$RUN_DIR" "terminal3_exit rc=$rc"
  save_compose_logs "$RUN_DIR" compose.netwaggle.yaml replay
  save_container_inspect_for_compose "$RUN_DIR" compose.netwaggle.yaml replay
  save_docker_ps "$RUN_DIR" after_replay
  save_webui_api_state "$RUN_DIR"
  "$SCRIPT_DIR/collect_debug_snapshot.sh" --run-dir "$RUN_DIR" --no-tar >/dev/null 2>&1 || true
  exit "$rc"
}
trap cleanup EXIT

log "Terminal 3: starting replay/detector/CE services: ${SERVICES[*]}"
log "Run directory: $RUN_DIR"
log "Saved live log: $RUN_DIR/logs/terminal3_replay_containers.log"
log "Publish logs: net=$IOBT_LOG_NET_PUBLISH every=$IOBT_LOG_NET_PUBLISH_EVERY_N; local=$IOBT_LOG_LOCAL_PUBLISH every=$IOBT_LOG_LOCAL_PUBLISH_EVERY_N"
log "Start replay from the web UI. This terminal should then show [publish:local] raw IPC and [publish:net] MQTT messages."
log "Stop with Ctrl-C. Final logs, docker inspect, web UI message buffers, and timelines will be saved automatically."

set +e
docker compose -f compose.netwaggle.yaml up --timestamps "${BUILD_ARGS[@]}" "${SERVICES[@]}" 2>&1 | tee -a "$RUN_DIR/logs/terminal3_replay_containers.log"
exit_code=${PIPESTATUS[0]}
set -e
exit "$exit_code"
