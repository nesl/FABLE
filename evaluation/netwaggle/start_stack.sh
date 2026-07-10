#!/usr/bin/env bash
# Start the host MQTT/UI, NetWaggle anchor containers, Mininet runner, and
# optionally the full replay services.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PROFILE_ARG="good_network"
TOPOLOGY="$DEFAULT_TOPOLOGY"
DEVICE="orin11"
REGENERATE_COMPOSE=0
ANCHORS_ONLY=0
BUILD=1
START_TRACE=1
START_METRICS=0
START_PROBES=1
PROBE_INTERVAL="1.0"
RUN_ID=""

usage() {
  cat <<USAGE
usage: $0 [options]

Options:
  --profile NAME_OR_PATH       Network profile to use. Default: good_network.
                               Examples: good_network, smoke_fixed_latency,
                               constrained_bandwidth, high_latency_cloud.
  --topology PATH              Topology JSON. Default: $DEFAULT_TOPOLOGY
  --device NAME                Device used if compose.replay.yaml must be regenerated. Default: orin11.
  --regenerate-compose         Regenerate compose.replay.yaml and compose.netwaggle.yaml.
  --anchors-only               Start only MQTT/UI + NetWaggle anchor containers + Mininet.
                               Useful for smoke tests without replay services.
  --no-build                   Do not pass --build to docker compose.
  --no-trace                   Do not start MQTT JSONL tracing.
  --metrics                    Start interface/qdisc metrics collector.
  --no-probes                  Do not start /netwaggle/probe latency probes.
  --probe-interval SEC         Probe interval for UI latency panel. Default: 1.0.
  --run-id ID                  Run directory name. Default: timestamp_profile.
  -h, --help                   Show this help.

Examples:
  $0 --profile smoke_fixed_latency --anchors-only
  $0 --profile good_network --regenerate-compose
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE_ARG="$2"; shift 2 ;;
    --topology) TOPOLOGY="$(readlink -f "$2")"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --regenerate-compose) REGENERATE_COMPOSE=1; shift ;;
    --anchors-only) ANCHORS_ONLY=1; shift ;;
    --no-build) BUILD=0; shift ;;
    --no-trace) START_TRACE=0; shift ;;
    --metrics) START_METRICS=1; shift ;;
    --no-probes) START_PROBES=0; shift ;;
    --probe-interval) PROBE_INTERVAL="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

PROFILE="$(resolve_profile "$PROFILE_ARG")"
PNAME="$(profile_name "$PROFILE")"
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date '+%Y%m%d_%H%M%S')_${PNAME}"
fi
RUN_DIR="$RUN_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "$CURRENT_LINK"

log "Run directory: $RUN_DIR"
log "Profile: $PROFILE"

need_cmd docker
need_cmd sudo
need_cmd "$PYTHON_BIN"
ensure_system_python_mininet >/dev/null
if (( START_TRACE || START_PROBES )); then ensure_python_package paho.mqtt paho-mqtt >/dev/null; fi

sudo -v

cd "$REPLAY_DIR"
if (( REGENERATE_COMPOSE )) || [[ ! -f compose.replay.yaml ]]; then
  log "Generating compose.replay.yaml for device=$DEVICE"
  "$PYTHON_BIN" setup/generate_replay_compose.py --device "$DEVICE" --compose-out compose.replay.yaml
fi
if (( REGENERATE_COMPOSE )) || [[ ! -f compose.netwaggle.yaml ]]; then
  log "Generating compose.netwaggle.yaml"
  "$PYTHON_BIN" ../netwaggle/scripts/make_netwaggle_compose.py \
    --compose-in compose.replay.yaml \
    --compose-out compose.netwaggle.yaml \
    --node-map "$TOPOLOGY"
fi

log "Validating compose.netwaggle.yaml"
docker compose -f compose.netwaggle.yaml config > "$RUN_DIR/compose.netwaggle.rendered.yaml"
cp "$TOPOLOGY" "$RUN_DIR/topology.json"
cp "$PROFILE" "$RUN_DIR/profile.json"

log "Starting host-side MQTT broker and web UI"
BUILD_ARGS=()
if (( BUILD )); then BUILD_ARGS+=(--build); fi
docker compose -f compose.server.yaml up -d "${BUILD_ARGS[@]}"

log "Starting NetWaggle anchor containers"
mapfile -t ANCHORS < <(anchor_names "$TOPOLOGY")
docker compose -f compose.netwaggle.yaml up -d "${BUILD_ARGS[@]}" "${ANCHORS[@]}"

log "Starting NetWaggle Mininet runner"
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
  log "NetWaggle runner log follows:"
  tail -n 80 "$RUN_DIR/netwaggle_runner.log" >&2 || true
  fail "Timed out waiting for netwaggle0 on ${ANCHORS[0]}"
fi

log "Starting MQTT trace"
if (( START_TRACE )); then
  (
    cd "$NETWAGGLE_DIR"
    "$PYTHON_BIN" scripts/mqtt_trace.py --host localhost --out "$RUN_DIR/mqtt_trace.jsonl"
  ) > "$RUN_DIR/mqtt_trace.log" 2>&1 &
  echo $! > "$RUN_DIR/mqtt_trace.pid"
fi

if (( START_METRICS )); then
  log "Starting NetWaggle metrics collector"
  (
    cd "$NETWAGGLE_DIR"
    sudo PYTHONPATH=. "$PYTHON_BIN" -m netwaggle.metrics \
      --topology "$TOPOLOGY" \
      --out "$RUN_DIR/link_stats.jsonl"
  ) > "$RUN_DIR/metrics.log" 2>&1 &
  echo $! > "$RUN_DIR/metrics.pid"
fi

if (( START_PROBES )); then
  log "Starting NetWaggle latency probes for the web UI"
  "$SCRIPT_DIR/start_latency_probes.sh" \
    --topology "$TOPOLOGY" \
    --profile "$PROFILE" \
    --run-dir "$RUN_DIR" \
    --interval "$PROBE_INTERVAL"
fi

if (( ANCHORS_ONLY )); then
  log "Anchors-only mode: replay services were not started."
else
  log "Starting replay services"
  docker compose -f compose.netwaggle.yaml up -d "${BUILD_ARGS[@]}"
fi

cat > "$RUN_DIR/status.txt" <<STATUS
NetWaggle run: $RUN_ID
Profile: $PROFILE
Topology: $TOPOLOGY
Replay dir: $REPLAY_DIR
NetWaggle dir: $NETWAGGLE_DIR
Web UI: http://localhost:8080
MQTT broker from host: localhost:1883
MQTT broker from NetWaggle nodes: 10.255.0.1:1883
Anchors only: $ANCHORS_ONLY
Latency probes: $START_PROBES
Probe interval: $PROBE_INTERVAL
STATUS

log "Started. Web UI: http://localhost:8080"
log "Logs are in: $RUN_DIR"
log "The Web UI NetWaggle panel should update from /netwaggle/probe messages."
log "Try: $SCRIPT_DIR/smoke_fixed_latency.sh"
