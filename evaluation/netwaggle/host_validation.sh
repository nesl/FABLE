#!/usr/bin/env bash
# Validate that NetWaggle anchor containers are attached, can reach the gateway,
# and can publish MQTT messages through the emulated namespace.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TOPOLOGY="$DEFAULT_TOPOLOGY"
MQTT_HOST="$DEFAULT_MQTT_HOST"
MQTT_PORT="$DEFAULT_MQTT_PORT"
NODE_CONTAINER="netwaggle-node-orin11"
PEER_CONTAINER="netwaggle-node-cloud1"

usage() {
  cat <<USAGE
usage: $0 [options]

Options:
  --topology PATH       Topology JSON. Default: $DEFAULT_TOPOLOGY
  --node CONTAINER      Source anchor container. Default: $NODE_CONTAINER
  --peer CONTAINER      Peer anchor container to ping. Default: $PEER_CONTAINER
  --mqtt-host IP        MQTT host as seen by NetWaggle nodes. Default: $MQTT_HOST
  --mqtt-port PORT      MQTT port. Default: $MQTT_PORT
  -h, --help            Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --topology) TOPOLOGY="$(readlink -f "$2")"; shift 2 ;;
    --node) NODE_CONTAINER="$2"; shift 2 ;;
    --peer) PEER_CONTAINER="$2"; shift 2 ;;
    --mqtt-host) MQTT_HOST="$2"; shift 2 ;;
    --mqtt-port) MQTT_PORT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

need_cmd docker
sudo -v
container_running "$NODE_CONTAINER" || fail "$NODE_CONTAINER is not running. Run start_stack.sh first."
container_running "$PEER_CONTAINER" || fail "$PEER_CONTAINER is not running. Run start_stack.sh first."

log "Checking netwaggle0 in $NODE_CONTAINER"
nsenter_net "$NODE_CONTAINER" ip addr show netwaggle0
nsenter_net "$NODE_CONTAINER" ip route show

log "Pinging NetWaggle MQTT/gateway $MQTT_HOST from $NODE_CONTAINER"
nsenter_net "$NODE_CONTAINER" ping -c 5 "$MQTT_HOST"

peer_ip="$(nsenter_net "$PEER_CONTAINER" ip -o -4 addr show netwaggle0 | awk '{print $4}' | cut -d/ -f1 | head -1)"
if [[ -n "$peer_ip" ]]; then
  log "Pinging peer $PEER_CONTAINER at $peer_ip"
  nsenter_net "$NODE_CONTAINER" ping -c 3 "$peer_ip"
fi

log "Publishing validation MQTT message through $NODE_CONTAINER namespace"
docker run --rm --network "container:$NODE_CONTAINER" eclipse-mosquitto:2 \
  mosquitto_pub -h "$MQTT_HOST" -p "$MQTT_PORT" \
  -t /debug/netwaggle/analytics/yolo/status \
  -m "{\"source\":\"netwaggle_host_validation\",\"node\":\"$NODE_CONTAINER\",\"ts\":$(date +%s.%N)}"

log "Validation message published. Open http://localhost:8080 and check the event log / YOLO status panel."
