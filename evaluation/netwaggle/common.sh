#!/usr/bin/env bash
# Common helpers for FABLE/NetWaggle evaluation scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FABLE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPLAY_DIR="$FABLE_ROOT/iobt-minimal-ce-replay"
NETWAGGLE_DIR="$FABLE_ROOT/netwaggle"
RUN_ROOT="$FABLE_ROOT/runs/netwaggle"
CURRENT_LINK="$RUN_ROOT/current"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
DEFAULT_TOPOLOGY="$NETWAGGLE_DIR/configs/fable_single_host.json"
DEFAULT_PROFILE="$NETWAGGLE_DIR/configs/profiles/good_network.json"
DEFAULT_MQTT_HOST="10.255.0.1"
DEFAULT_MQTT_PORT="1883"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

fail() {
  log "ERROR: $*"
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

resolve_profile() {
  local profile="${1:-}"
  if [[ -z "$profile" ]]; then
    printf '%s\n' "$DEFAULT_PROFILE"
    return
  fi
  if [[ -f "$profile" ]]; then
    readlink -f "$profile"
    return
  fi
  if [[ -f "$NETWAGGLE_DIR/configs/profiles/${profile}.json" ]]; then
    readlink -f "$NETWAGGLE_DIR/configs/profiles/${profile}.json"
    return
  fi
  fail "Profile not found: $profile"
}

profile_name() {
  local path="$1"
  "$PYTHON_BIN" - "$path" <<'PY'
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = json.load(f)

name = data.get('name')
if not name:
    base = sys.argv[1].split('/')[-1]
    name = base[:-5] if base.endswith('.json') else base
print(name)
PY
}

anchor_names() {
  local topology="${1:-$DEFAULT_TOPOLOGY}"
  "$PYTHON_BIN" - "$topology" <<'PY'
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = json.load(f)
for node in data.get('logical_nodes', []):
    print(node.get('anchor_container') or f"netwaggle-node-{node['name']}")
PY
}

container_pid() {
  docker inspect --format '{{.State.Pid}}' "$1" 2>/dev/null || true
}

container_running() {
  local pid
  pid="$(container_pid "$1")"
  [[ -n "$pid" && "$pid" != "0" ]]
}

nsenter_net() {
  local container="$1"
  shift
  local pid
  pid="$(container_pid "$container")"
  [[ -n "$pid" && "$pid" != "0" ]] || fail "Container is not running: $container"
  sudo nsenter -t "$pid" -n "$@"
}

wait_for_netwaggle_if() {
  local container="$1"
  local timeout_sec="${2:-60}"
  local start now
  start="$(date +%s)"
  while true; do
    if nsenter_net "$container" ip addr show netwaggle0 >/dev/null 2>&1; then
      return 0
    fi
    now="$(date +%s)"
    if (( now - start >= timeout_sec )); then
      return 1
    fi
    sleep 1
  done
}

ensure_system_python_mininet() {
  "$PYTHON_BIN" - <<'PY'
from mininet.net import Mininet
print('Mininet import OK')
PY
}

ensure_python_package() {
  local import_name="$1"
  local package_hint="${2:-$1}"
  "$PYTHON_BIN" - "$import_name" "$package_hint" <<'PY'
import importlib, sys
name, hint = sys.argv[1], sys.argv[2]
try:
    importlib.import_module(name)
except Exception as exc:
    raise SystemExit(f"Missing Python package {name!r}. Install with: {sys.executable} -m pip install {hint}") from exc
PY
}

safe_kill_pid_file() {
  local file="$1"
  [[ -f "$file" ]] || return 0
  local pid
  pid="$(cat "$file" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    log "Stopping PID $pid from $file"
    sudo kill "$pid" >/dev/null 2>&1 || kill "$pid" >/dev/null 2>&1 || true
    sleep 1
    if kill -0 "$pid" >/dev/null 2>&1; then
      sudo kill -9 "$pid" >/dev/null 2>&1 || kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  fi
  rm -f "$file"
}
