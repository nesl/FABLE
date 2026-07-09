#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <anchor-container>" >&2
  exit 2
fi
pid="$(docker inspect --format '{{.State.Pid}}' "$1")"
sudo nsenter -t "$pid" -n ip addr show
sudo nsenter -t "$pid" -n ip route show
