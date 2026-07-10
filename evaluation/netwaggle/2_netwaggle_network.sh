#!/usr/bin/env bash
# Terminal 2: start NetWaggle anchors, Mininet runner, and UI latency probes.
# The Mininet runner stays in the background so this terminal can also start
# probes and tail NetWaggle logs in one place.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PROFILE_ARG="good_network"
TOPOLOGY="$DEFAULT_TOPOLOGY"
REGENERATE_COMPOSE=0
BUILD=1
PROBES=1
PROBE_INTERVAL="1.0"
RUN_ID=""
MQTT_TRACE=1
NEW_RUN=0

usage() {
  cat <<USAGE
usage: $0 [options]

Options:
  --profile NAME_OR_PATH       Network profile. Default: good_network.
                               Useful: smoke_fixed_latency, good_network,
                               constrained_bandwidth, high_latency_cloud.
  --topology PATH              Topology JSON. Default: $DEFAULT_TOPOLOGY
  --regenerate-compose         Regenerate compose.netwaggle.yaml.
  --no-build                   Do not build anchor images.
  --no-probes                  Do not start /netwaggle/probe latency probes.
  --probe-interval SEC         Probe interval. Default: 1.0.
  --run-id ID                  Run directory name.
  --new-run                    Force a fresh run directory.
  --no-mqtt-trace              Do not start host-side MQTT trace capture.
  -h, --help                   Show this help.

Run this after Terminal 1 is up. Stop with Ctrl-C.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE_ARG="$2"; shift 2 ;;
    --topology) TOPOLOGY="$(readlink -f "$2")"; shift 2 ;;
    --regenerate-compose) REGENERATE_COMPOSE=1; shift ;;
    --no-build) BUILD=0; shift ;;
    --no-probes) PROBES=0; shift ;;
    --probe-interval) PROBE_INTERVAL="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --new-run) NEW_RUN=1; rm -f "$CURRENT_LINK"; shift ;;
    --no-mqtt-trace) MQTT_TRACE=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

PROFILE="$(resolve_profile "$PROFILE_ARG")"
PNAME="$(profile_name "$PROFILE")"
if [[ -n "$RUN_ID" ]]; then
  export NETWAGGLE_RUN_ID="$RUN_ID"
elif (( NEW_RUN )); then
  rm -f "$CURRENT_LINK"
fi
RUN_DIR="$(ensure_run_dir "${PNAME}")"
RUN_ID="$(basename "$RUN_DIR")"
write_run_event "$RUN_DIR" "terminal2_start profile=$PNAME probes=$PROBES mqtt_trace=$MQTT_TRACE"

need_cmd docker
need_cmd sudo
need_cmd "$PYTHON_BIN"
ensure_system_python_mininet >/dev/null
if (( PROBES || MQTT_TRACE )); then ensure_python_package paho.mqtt paho-mqtt >/dev/null; fi
sudo -v

cd "$REPLAY_DIR"
if (( REGENERATE_COMPOSE )) || [[ ! -f compose.netwaggle.yaml ]]; then
  if [[ ! -f compose.replay.yaml ]]; then
    log "compose.replay.yaml missing; generating for orin11"
    "$PYTHON_BIN" setup/generate_replay_compose.py --device orin11 --compose-out compose.replay.yaml
  fi
  log "Generating compose.netwaggle.yaml"
  "$PYTHON_BIN" ../netwaggle/scripts/make_netwaggle_compose.py \
    --compose-in compose.replay.yaml \
    --compose-out compose.netwaggle.yaml \
    --node-map "$TOPOLOGY"
fi

save_compose_rendered "$RUN_DIR" compose.netwaggle.yaml netwaggle
cp "$TOPOLOGY" "$RUN_DIR/topology.json"
cp "$PROFILE" "$RUN_DIR/profile.json"

mapfile -t ANCHORS < <(anchor_names "$TOPOLOGY")
BUILD_ARGS=()
if (( BUILD )); then BUILD_ARGS+=(--build); fi
log "Starting NetWaggle anchor containers: ${ANCHORS[*]}"
docker compose -f compose.netwaggle.yaml up -d "${BUILD_ARGS[@]}" "${ANCHORS[@]}"
save_docker_ps "$RUN_DIR" after_anchors
save_container_inspect_for_compose "$RUN_DIR" compose.netwaggle.yaml netwaggle_after_anchors

cleanup() {
  log "Stopping NetWaggle terminal resources"
  safe_kill_pid_file "$RUN_DIR/netwaggle_runner.pid" || true
  safe_kill_pid_glob "$RUN_DIR/latency_probe_*.pid" || true
  safe_kill_pid_file "$RUN_DIR/tail.pid" || true
  safe_kill_pid_file "$RUN_DIR/mqtt_trace.pid" || true
  (cd "$NETWAGGLE_DIR" && sudo PYTHONPATH=. "$PYTHON_BIN" -m netwaggle.cleanup --topology "$TOPOLOGY") || true
  sudo mn -c >/dev/null 2>&1 || true
  write_run_event "$RUN_DIR" "terminal2_exit"
  save_docker_ps "$RUN_DIR" after_netwaggle
  save_webui_api_state "$RUN_DIR"
}
trap cleanup EXIT INT TERM


if (( MQTT_TRACE )); then
  log "Starting host-side MQTT trace capture"
  (
    cd "$NETWAGGLE_DIR"
    "$PYTHON_BIN" scripts/mqtt_trace.py \
      --host localhost \
      --out "$RUN_DIR/mqtt_trace.jsonl"
  ) > "$RUN_DIR/logs/mqtt_trace.log" 2>&1 &
  echo $! > "$RUN_DIR/mqtt_trace.pid"
fi

log "Starting Mininet/NetWaggle runner with profile=$PNAME"
(
  cd "$NETWAGGLE_DIR"
  sudo PYTHONPATH=. "$PYTHON_BIN" -m netwaggle.runner \
    --topology "$TOPOLOGY" \
    --profile "$PROFILE" \
    --no-cli \
    --hold
) > "$RUN_DIR/netwaggle_runner.log" 2>&1 &
echo $! > "$RUN_DIR/netwaggle_runner.pid"

log "Waiting for netwaggle0 on ${ANCHORS[0]}"
if ! wait_for_netwaggle_if "${ANCHORS[0]}" 90; then
  tail -n 100 "$RUN_DIR/netwaggle_runner.log" >&2 || true
  fail "Timed out waiting for netwaggle0 on ${ANCHORS[0]}"
fi

if (( PROBES )); then
  log "Starting latency probes for UI NetWaggle panel"
  "$SCRIPT_DIR/start_latency_probes.sh" \
    --topology "$TOPOLOGY" \
    --profile "$PROFILE" \
    --run-dir "$RUN_DIR" \
    --interval "$PROBE_INTERVAL"
fi

cat > "$RUN_DIR/status.txt" <<STATUS
Run: $RUN_ID
Profile: $PROFILE
Topology: $TOPOLOGY
Web UI: http://localhost:8080
MQTT from nodes: 10.255.0.1:1883
Terminal 2 owns NetWaggle runner, MQTT trace, and probes.
STATUS

log "NetWaggle is up. Web UI should show the NetWaggle Network panel."
log "Run directory: $RUN_DIR"
log "Useful checks in another shell if needed:"
log "  docker exec netwaggle-node-orin11 ip route"
log "  docker exec netwaggle-node-orin11 ping -c 3 10.255.0.1"
log "Tailing NetWaggle/probe logs below. Stop with Ctrl-C."

tail -n +1 -F \
  "$RUN_DIR/netwaggle_runner.log" \
  "$RUN_DIR/logs/mqtt_trace.log" \
  "$RUN_DIR"/latency_probe_*.log &
echo $! > "$RUN_DIR/tail.pid"
wait "$(cat "$RUN_DIR/netwaggle_runner.pid")"
