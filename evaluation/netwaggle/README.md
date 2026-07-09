# NetWaggle evaluation bring-up scripts

This folder contains convenience scripts for running the FABLE replay stack
through NetWaggle. The scripts assume this repository layout:

```text
FABLE/
├── iobt-minimal-ce-replay/
├── netwaggle/
└── evaluation/netwaggle/
```

The intended first architecture is wired and single-host: raw replay streams stay
local inside a logical node, while MQTT traffic is routed through Mininet/TC.
The host MQTT broker and web UI remain outside Mininet.

## 1. Fast smoke test, no full replay

This starts only the host MQTT/UI, NetWaggle anchor containers, and the Mininet
runner under a deterministic high-latency profile.

```bash
cd ~/Documents/FABLE
./evaluation/netwaggle/start_stack.sh --profile smoke_fixed_latency --anchors-only
```

Open the UI:

```text
http://localhost:8080
```

Then publish synthetic status/CE messages through the `orin11` logical node:

```bash
./evaluation/netwaggle/smoke_fixed_latency.sh
```

Expected behavior:

- The script first pings `10.255.0.1` from inside `netwaggle-node-orin11`.
- With `smoke_fixed_latency`, ping RTT should be roughly 200 ms plus overhead.
- The UI event log should show synthetic messages on:
  - `/debug/orin11/analytics/yolo/status`
  - `/debug/orin11/audio_detector/status`
  - `/complex_events/demo`
  - `/replay/status/zed/orin11`

This is the quickest way to verify that container namespace attachment, Mininet,
MQTT routing, and the web UI are all connected.

## 2. Host validation

After `start_stack.sh`, run:

```bash
./evaluation/netwaggle/host_validation.sh
```

This prints the `netwaggle0` interface, routes, ping results, peer reachability,
and publishes one validation message to the web UI.

## 3. Full replay stack

Start server, anchors, NetWaggle, and replay services:

```bash
./evaluation/netwaggle/start_stack.sh --profile good_network --regenerate-compose
```

Then list scenarios:

```bash
./evaluation/netwaggle/list_scenarios.sh
```

Run a short replay window:

```bash
./evaluation/netwaggle/run_replay_window.sh \
  --scenario 20260414_111951 \
  --start 0 \
  --end 30 \
  --playback-mode max
```

Replace the scenario ID with one that exists in your catalog.

## 4. Try other network profiles

```bash
./evaluation/netwaggle/stop_stack.sh
./evaluation/netwaggle/start_stack.sh --profile constrained_bandwidth --regenerate-compose
```

Available profiles live in `netwaggle/configs/profiles/`, including:

- `good_network`
- `smoke_fixed_latency`
- `constrained_bandwidth`
- `high_latency_cloud`
- `lossy_edge`
- `cloud_degraded`

Each run writes logs and traces to:

```text
FABLE/runs/netwaggle/<timestamp>_<profile>/
```

The symlink below points to the latest run:

```text
FABLE/runs/netwaggle/current
```

Useful files:

```text
netwaggle_runner.log
mqtt_trace.jsonl
compose.netwaggle.rendered.yaml
profile.json
topology.json
```

## 5. Stop everything

```bash
./evaluation/netwaggle/stop_stack.sh
```

To leave the host MQTT broker and web UI running:

```bash
./evaluation/netwaggle/stop_stack.sh --keep-server
```

## Notes

- The scripts intentionally call `/usr/bin/python3` for Mininet-related code so
  an activated project venv does not hide the system Mininet installation.
- Mininet/NetWaggle commands need `sudo` because they create OVS switches, veths,
  routes, namespaces, and TC qdiscs.
- Docker Compose commands do not use `sudo` by default.
- The smoke test uses `docker run --network container:<anchor>` with the
  `eclipse-mosquitto:2` image, so the publish path is actually through the
  NetWaggle-attached logical node namespace.
