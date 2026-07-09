#!/usr/bin/env bash
# Send synthetic MQTT traffic through a NetWaggle node namespace. The messages
# use topics already displayed by the web UI, so this is a quick end-to-end test
# without starting a real replay scenario.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

NODE_CONTAINER="netwaggle-node-orin11"
MQTT_HOST="$DEFAULT_MQTT_HOST"
MQTT_PORT="$DEFAULT_MQTT_PORT"
COUNT=5
INTERVAL=1
PROFILE_LABEL="smoke_fixed_latency"
PING_FIRST=1

usage() {
  cat <<USAGE
usage: $0 [options]

Options:
  --node CONTAINER      Anchor namespace to publish from. Default: $NODE_CONTAINER
  --mqtt-host IP        MQTT host as seen by NetWaggle nodes. Default: $MQTT_HOST
  --mqtt-port PORT      MQTT port. Default: $MQTT_PORT
  --count N             Number of message groups. Default: $COUNT
  --interval SEC        Delay between groups. Default: $INTERVAL
  --profile-label NAME  Label embedded in payload. Default: $PROFILE_LABEL
  --no-ping             Skip gateway ping before publishing.
  -h, --help            Show this help.

Recommended setup:
  ./evaluation/netwaggle/start_stack.sh --profile smoke_fixed_latency --anchors-only
  ./evaluation/netwaggle/smoke_fixed_latency.sh
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --node) NODE_CONTAINER="$2"; shift 2 ;;
    --mqtt-host) MQTT_HOST="$2"; shift 2 ;;
    --mqtt-port) MQTT_PORT="$2"; shift 2 ;;
    --count) COUNT="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --profile-label) PROFILE_LABEL="$2"; shift 2 ;;
    --no-ping) PING_FIRST=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

need_cmd docker
sudo -v
container_running "$NODE_CONTAINER" || fail "$NODE_CONTAINER is not running. Run start_stack.sh first."
wait_for_netwaggle_if "$NODE_CONTAINER" 5 || fail "$NODE_CONTAINER does not have netwaggle0. Is netwaggle.runner running?"

if (( PING_FIRST )); then
  log "Pinging $MQTT_HOST from $NODE_CONTAINER. With smoke_fixed_latency, RTT should be roughly 200 ms plus overhead."
  nsenter_net "$NODE_CONTAINER" ping -c 5 "$MQTT_HOST" || true
fi

log "Publishing $COUNT synthetic message groups through $NODE_CONTAINER -> NetWaggle -> MQTT."
log "Open http://localhost:8080 and watch the event log, YOLO, Audio, and CE panels."

for i in $(seq 1 "$COUNT"); do
  ts="$(date +%s.%N)"
  base="{\"source\":\"netwaggle_smoke\",\"profile\":\"$PROFILE_LABEL\",\"node\":\"$NODE_CONTAINER\",\"seq\":$i,\"sent_ts\":$ts}"

  docker run --rm --network "container:$NODE_CONTAINER" eclipse-mosquitto:2 \
    mosquitto_pub -h "$MQTT_HOST" -p "$MQTT_PORT" \
    -t /debug/orin11/analytics/yolo/status \
    -m "$base"

  docker run --rm --network "container:$NODE_CONTAINER" eclipse-mosquitto:2 \
    mosquitto_pub -h "$MQTT_HOST" -p "$MQTT_PORT" \
    -t /debug/orin11/audio_detector/status \
    -m "$base"

  docker run --rm --network "container:$NODE_CONTAINER" eclipse-mosquitto:2 \
    mosquitto_pub -h "$MQTT_HOST" -p "$MQTT_PORT" \
    -t /complex_events/demo \
    -m "{\"source\":\"netwaggle_smoke\",\"profile\":\"$PROFILE_LABEL\",\"event\":\"synthetic_netwaggle_latency_probe\",\"seq\":$i,\"sent_ts\":$ts,\"note\":\"This is synthetic CE traffic, not replay output.\"}"

  docker run --rm --network "container:$NODE_CONTAINER" eclipse-mosquitto:2 \
    mosquitto_pub -h "$MQTT_HOST" -p "$MQTT_PORT" \
    -t /replay/status/zed/orin11 \
    -m "{\"source\":\"netwaggle_smoke\",\"scenario\":\"synthetic\",\"sim_time\":$i,\"frame\":$i,\"avg_loudness\":null,\"sent_ts\":$ts}"

  sleep "$INTERVAL"
done

log "Smoke messages sent. Latest MQTT trace, if enabled: $CURRENT_LINK/mqtt_trace.jsonl"
