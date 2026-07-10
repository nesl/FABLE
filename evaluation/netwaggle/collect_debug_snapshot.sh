#!/usr/bin/env bash
# Collect a point-in-time debug bundle for NetWaggle + replay experiments.
# Safe to run while the three terminals are active, and also called automatically
# by 3_replay_containers.sh on exit.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_DIR=""
MAKE_TAR=1
TOPOLOGY="$DEFAULT_TOPOLOGY"

usage() {
  cat <<USAGE
usage: $0 [--run-dir PATH] [--topology PATH] [--no-tar]

Collects Docker logs, container inspect JSON, Compose rendered configs,
web-UI API buffers, NetWaggle metrics, routes for anchor containers, and a
small detector/replay timeline under runs/netwaggle/current/.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$(mkdir -p "$2" && cd "$2" && pwd)"; shift 2 ;;
    --topology) TOPOLOGY="$(readlink -f "$2")"; shift 2 ;;
    --no-tar) MAKE_TAR=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(ensure_run_dir debug_snapshot)"
fi
mkdir -p "$RUN_DIR" "$RUN_DIR/logs" "$RUN_DIR/compose" "$RUN_DIR/docker" "$RUN_DIR/inspect" "$RUN_DIR/api" "$RUN_DIR/metrics" "$RUN_DIR/netns" "$RUN_DIR/timelines"
write_run_event "$RUN_DIR" "debug_snapshot_start"

log "Collecting debug snapshot into $RUN_DIR"

# Compose configs and service logs.
if [[ -f "$REPLAY_DIR/compose.server.yaml" ]]; then
  save_compose_rendered "$RUN_DIR" compose.server.yaml server
  save_compose_logs "$RUN_DIR" compose.server.yaml server
  save_container_inspect_for_compose "$RUN_DIR" compose.server.yaml server
fi
if [[ -f "$REPLAY_DIR/compose.netwaggle.yaml" ]]; then
  save_compose_rendered "$RUN_DIR" compose.netwaggle.yaml netwaggle
  save_compose_logs "$RUN_DIR" compose.netwaggle.yaml replay
  save_container_inspect_for_compose "$RUN_DIR" compose.netwaggle.yaml replay
fi

save_docker_ps "$RUN_DIR" snapshot
save_webui_api_state "$RUN_DIR"

# Host process/network state useful for Mininet/OVS debugging.
(ps aux | grep -E 'netwaggle|mnexec|ovs|mosquitto|docker|node_latency_probe|mqtt_trace' | grep -v grep || true) > "$RUN_DIR/docker/host_processes_relevant.txt"
(ip addr || true) > "$RUN_DIR/netns/host_ip_addr.txt"
(ip route || true) > "$RUN_DIR/netns/host_ip_route.txt"
(sudo ovs-vsctl show || true) > "$RUN_DIR/netns/ovs_vsctl_show.txt" 2>&1

# Anchor namespace routes/interfaces.
if [[ -f "$TOPOLOGY" ]]; then
  while read -r anchor; do
    [[ -n "$anchor" ]] || continue
    if container_running "$anchor"; then
      {
        echo "# $anchor"
        docker inspect --format 'pid={{.State.Pid}} network={{.HostConfig.NetworkMode}} status={{.State.Status}} started={{.State.StartedAt}}' "$anchor" || true
        nsenter_net "$anchor" ip addr || true
        echo
        nsenter_net "$anchor" ip route || true
        echo
        nsenter_net "$anchor" ping -c 2 -W 1 10.255.0.1 || true
      } > "$RUN_DIR/netns/${anchor}.netns.txt" 2>&1
    else
      echo "anchor not running: $anchor" > "$RUN_DIR/netns/${anchor}.netns.txt"
    fi
  done < <(anchor_names "$TOPOLOGY")
fi

# Single metrics snapshot if NetWaggle is still alive.
if [[ -d "$NETWAGGLE_DIR" && -f "$TOPOLOGY" ]]; then
  (cd "$NETWAGGLE_DIR" && sudo PYTHONPATH=. "$PYTHON_BIN" -m netwaggle.metrics \
    --topology "$TOPOLOGY" \
    --out "$RUN_DIR/metrics/link_stats.snapshot.jsonl" \
    --once) > "$RUN_DIR/metrics/link_stats.snapshot.log" 2>&1 || true
fi

# Timeline hints for late detector/model readiness issues.
if [[ -f "$RUN_DIR/logs/terminal3_replay_containers.log" ]]; then
  grep -Ein \
    'readiness|/readiness/|ready|Starting model load|Loading YOLO|Finished loading model|model_load_failed|Running on .*zed data source|YOLO detections topic|Attempting to connect to local publisher|Local publisher .*not ready|Subscribed to local topic|Starting local subscriber|/replay/config|/replay/sync|replay_requested|event.*complete|\[publish:(net|local)\]|audio|yolo|zed|respeaker' \
    "$RUN_DIR/logs/terminal3_replay_containers.log" \
    > "$RUN_DIR/timelines/replay_detector_timeline.grep.txt" 2>/dev/null || true
fi
if [[ -f "$RUN_DIR/mqtt_trace.jsonl" ]]; then
  "$PYTHON_BIN" - "$RUN_DIR/mqtt_trace.jsonl" "$RUN_DIR/timelines/mqtt_topic_summary.txt" <<'PY' || true
import json, sys, collections
inp, out = sys.argv[1], sys.argv[2]
counts = collections.Counter()
first = {}
last = {}
with open(inp, 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        topic = obj.get('topic', '')
        counts[topic] += 1
        ts = obj.get('ts') or obj.get('time') or obj.get('received_at')
        first.setdefault(topic, ts)
        last[topic] = ts
with open(out, 'w', encoding='utf-8') as g:
    for topic, count in counts.most_common():
        g.write(f'{count}\t{first.get(topic)}\t{last.get(topic)}\t{topic}\n')
PY
fi

write_run_event "$RUN_DIR" "debug_snapshot_done"

if (( MAKE_TAR )); then
  tarball="$RUN_ROOT/$(basename "$RUN_DIR")_debug_snapshot.tar.gz"
  (cd "$RUN_ROOT" && tar -czf "$tarball" "$(basename "$RUN_DIR")")
  log "Wrote tarball: $tarball"
fi

log "Snapshot complete: $RUN_DIR"
