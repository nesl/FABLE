#!/usr/bin/env bash
set -euo pipefail

echo "== Host DNS =="
getent hosts deb.debian.org || true
getent hosts registry-1.docker.io || true

echo
if command -v resolvectl >/dev/null 2>&1; then
  echo "== resolvectl DNS summary =="
  resolvectl status | sed -n '1,80p' || true
fi

echo
if command -v docker >/dev/null 2>&1; then
  echo "== Docker daemon info =="
  docker info --format 'Server={{.ServerVersion}} Driver={{.Driver}} DefaultRuntime={{.DefaultRuntime}}' || true
  echo
  echo "== Docker pull smoke test =="
  docker pull hello-world:latest || true
  echo
  echo "== Docker DNS smoke test =="
  docker run --rm busybox:1.36 nslookup deb.debian.org || true
  docker run --rm busybox:1.36 nslookup registry-1.docker.io || true
fi

echo
cat <<'MSG'
If host DNS works but Docker DNS/pull fails, configure Docker daemon DNS:

  sudo mkdir -p /etc/docker
  sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak.$(date +%s) 2>/dev/null || true
  printf '{\n  "dns": ["1.1.1.1", "8.8.8.8"]\n}\n' | sudo tee /etc/docker/daemon.json
  sudo systemctl restart docker

Then rerun this script.
MSG
