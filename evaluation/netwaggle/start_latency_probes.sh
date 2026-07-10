#!/usr/bin/env bash
# Start timestamped NetWaggle latency probes from each logical node namespace.
# These probes publish /netwaggle/probe/<node> through Mininet to MQTT so the
# existing web UI can show that emulated latency is active during replay.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TOPOLOGY="$DEFAULT_TOPOLOGY"
PROFILE="$DEFAULT_PROFILE"
RUN_DIR=""
MQTT_HOST="$DEFAULT_MQTT_HOST"
MQTT_PORT="$DEFAULT_MQTT_PORT"
INTERVAL="1.0"
COUNT="0"
PAYLOAD_BYTES="0"
NODES_FILTER=""

usage() {
  cat <<USAGE
usage: $0 [options]

Options:
  --topology PATH        Topology JSON. Default: $TOPOLOGY
  --profile NAME_OR_PATH Profile JSON or profile name. Default: good_network
  --run-dir PATH         Directory for logs/PIDs. Default: runs/netwaggle/current
  --mqtt-host IP         MQTT host as seen by NetWaggle nodes. Default: $MQTT_HOST
  --mqtt-port PORT       MQTT port. Default: $MQTT_PORT
  --interval SEC         Probe interval. Default: $INTERVAL
  --count N              Probe count per node. 0 means forever. Default: $COUNT
  --payload-bytes N      Optional padding bytes per probe. Default: $PAYLOAD_BYTES
  --nodes CSV            Limit to logical node names, e.g. orin11,cloud1
  -h, --help             Show this help.

Recommended:
  ./evaluation/netwaggle/start_stack.sh --profile smoke_fixed_latency --anchors-only
  ./evaluation/netwaggle/start_latency_probes.sh --profile smoke_fixed_latency
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --topology) TOPOLOGY="$(readlink -f "$2")"; shift 2 ;;
    --profile) PROFILE="$(resolve_profile "$2")"; shift 2 ;;
    --run-dir) RUN_DIR="$(mkdir -p "$2" && cd "$2" && pwd)"; shift 2 ;;
    --mqtt-host) MQTT_HOST="$2"; shift 2 ;;
    --mqtt-port) MQTT_PORT="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --count) COUNT="$2"; shift 2 ;;
    --payload-bytes) PAYLOAD_BYTES="$2"; shift 2 ;;
    --nodes) NODES_FILTER="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  if [[ -L "$CURRENT_LINK" ]]; then
    RUN_DIR="$(readlink -f "$CURRENT_LINK")"
  else
    RUN_DIR="$RUN_ROOT/manual_latency_probes_$(date '+%Y%m%d_%H%M%S')"
  fi
fi
mkdir -p "$RUN_DIR"
PROFILE_NAME="$(profile_name "$PROFILE")"

need_cmd docker
need_cmd sudo
need_cmd "$PYTHON_BIN"
ensure_python_package paho.mqtt paho-mqtt >/dev/null
sudo -v

log "Publishing active NetWaggle profile to /netwaggle/profile"
(
  cd "$NETWAGGLE_DIR"
  "$PYTHON_BIN" scripts/publish_profile.py \
    --topology "$TOPOLOGY" \
    --profile "$PROFILE" \
    --mqtt-host localhost \
    --mqtt-port 1883 \
    --topic /netwaggle/profile \
    --retain
) > "$RUN_DIR/netwaggle_profile_published.json" 2> "$RUN_DIR/netwaggle_profile_publish.log" || \
  log "WARNING: failed to publish /netwaggle/profile; see $RUN_DIR/netwaggle_profile_publish.log"

log "Starting NetWaggle latency probes every ${INTERVAL}s; logs/PIDs in $RUN_DIR"

# Print TSV: name anchor configured_one_way_ms.
mapfile -t NODE_ROWS < <("$PYTHON_BIN" - "$TOPOLOGY" "$PROFILE" "$NODES_FILTER" <<'PY'
import json, math, re, sys
from pathlib import Path

topo_path, prof_path, filt = sys.argv[1:4]
topo = json.load(open(topo_path, 'r', encoding='utf-8'))
prof = json.load(open(prof_path, 'r', encoding='utf-8'))
allow = {x.strip() for x in filt.split(',') if x.strip()}

def delay_ms(v):
    if v is None: return 0.0
    if isinstance(v, (int, float)): return float(v)
    m = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)\s*(us|µs|ms|s)?', str(v).strip().lower())
    if not m: return 0.0
    n = float(m.group(1)); unit = m.group(2) or 'ms'
    return n/1000 if unit in {'us','µs'} else n*1000 if unit == 's' else n

links = {}
for source in (topo.get('links', []), prof.get('links', [])):
    for l in source:
        a = l.get('from') or l.get('src'); b = l.get('to') or l.get('dst')
        if not a or not b: continue
        key = tuple(sorted((str(a), str(b))))
        old = links.get(key, {}).copy(); old.update(l); links[key] = old

g = {}
for l in links.values():
    a = str(l.get('from') or l.get('src')); b = str(l.get('to') or l.get('dst'))
    g.setdefault(a, []).append((b, delay_ms(l.get('delay'))))
    g.setdefault(b, []).append((a, delay_ms(l.get('delay'))))

def dijkstra(src, dst):
    dist = {src: 0.0}; unseen = set(g)
    while unseen:
        cur = min(unseen, key=lambda n: dist.get(n, math.inf))
        if dist.get(cur, math.inf) == math.inf: break
        unseen.remove(cur)
        if cur == dst: break
        for nxt, w in g.get(cur, []):
            nd = dist[cur] + w
            if nd < dist.get(nxt, math.inf): dist[nxt] = nd
    return dist.get(dst, math.inf)

gw = str(topo.get('gateway', {}).get('switch') or '')
for n in topo.get('logical_nodes', []):
    name = str(n.get('name'))
    if allow and name not in allow: continue
    anchor = n.get('anchor_container') or f'netwaggle-node-{name}'
    d = dijkstra(str(n.get('switch')), gw)
    d_text = '' if math.isinf(d) else f'{d:.3f}'
    print(f'{name}\t{anchor}\t{d_text}')
PY
)

: > "$RUN_DIR/latency_probes.pids"
for row in "${NODE_ROWS[@]}"; do
  IFS=$'\t' read -r node anchor configured_ms <<< "$row"
  if ! container_running "$anchor"; then
    log "Skipping $node: anchor $anchor is not running"
    continue
  fi
  if ! wait_for_netwaggle_if "$anchor" 5; then
    log "Skipping $node: anchor $anchor has no netwaggle0"
    continue
  fi
  extra=()
  if [[ -n "$configured_ms" ]]; then
    extra+=(--configured-one-way-ms "$configured_ms")
  fi
  log "Probe $node via $anchor; configured one-way=${configured_ms:-unknown} ms"
  nsenter_net "$anchor" "$PYTHON_BIN" "$NETWAGGLE_DIR/scripts/node_latency_probe.py" \
    --node "$node" \
    --anchor-container "$anchor" \
    --profile "$PROFILE_NAME" \
    --mqtt-host "$MQTT_HOST" \
    --mqtt-port "$MQTT_PORT" \
    --interval "$INTERVAL" \
    --count "$COUNT" \
    --payload-bytes "$PAYLOAD_BYTES" \
    --log-every-n 10 \
    "${extra[@]}" \
    > "$RUN_DIR/latency_probe_${node}.log" 2>&1 &
  pid=$!
  echo "$pid" > "$RUN_DIR/latency_probe_${node}.pid"
  echo "$pid" >> "$RUN_DIR/latency_probes.pids"
done

log "Latency probes started. Web UI panel: http://localhost:8080"
