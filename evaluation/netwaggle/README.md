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

## 6. Starting replay from the web UI

`run_replay_window.sh` is only a CLI wrapper around the same MQTT control path
used by the web UI. When NetWaggle is running, you can start replay from the UI
instead:

1. Start the stack with NetWaggle, for example:

   ```bash
   ./evaluation/netwaggle/start_stack.sh --profile good_network --regenerate-compose
   ```

2. Open the web UI:

   ```text
   http://localhost:8080
   ```

3. Select the scenario/window/playback settings in the UI and press the replay
   start button.

The browser and web UI publish `/replay/config` and `/replay/sync` to the
host-side MQTT broker. Replay containers subscribe to that broker through their
NetWaggle-attached namespace, so the experiment still uses Mininet for the
container network path. This keeps human/control interaction stable while
network-shaping replay/detection/event traffic.

## 7. Network settings in the UI

It is possible to expose network profiles in the UI, but the safest design is to
keep privileged Mininet control outside the web UI container. A good next step is
an evaluation-only NetWaggle control service on the host that can list profiles,
report the active profile, and request a profile change. The web UI can call that
service, while the service performs the privileged `tc`/Mininet work.

Avoid putting direct `sudo`, `tc`, or Mininet commands inside the main web UI
container. That would mix replay visualization with privileged host networking
and make the core replay stack less portable.

## Seeing Mininet/NetWaggle latency in the web UI

`start_stack.sh` now starts small timestamped latency probes by default. Each probe
runs in a logical node's network namespace and publishes to:

```text
/netwaggle/probe/<node>
```

The web UI subscribes to `/netwaggle/#` and displays an observed one-way estimate
for each node. The estimate is:

```text
web-ui receive timestamp - probe sent timestamp
```

Because both processes run on the same physical host, their clocks are effectively
shared. The observed value includes Mininet/TC path delay plus MQTT broker and web
UI receive overhead. The configured value shown next to it is computed from the
active NetWaggle profile and published on `/netwaggle/profile`.

For a fixed-latency smoke test:

```bash
cd ~/Documents/FABLE
./evaluation/netwaggle/start_stack.sh --profile smoke_fixed_latency --anchors-only
# Open http://localhost:8080 and look at the NetWaggle Network panel.
```

For a real replay, start the full stack and then use the web UI's normal replay
controls:

```bash
./evaluation/netwaggle/start_stack.sh --profile good_network --regenerate-compose
# Open http://localhost:8080, choose a scenario, and click Start replay.
```

To disable probes:

```bash
./evaluation/netwaggle/start_stack.sh --profile good_network --no-probes
```

## Three-terminal debug workflow

For normal debugging, use these three terminals instead of `start_stack.sh`.
This keeps each layer visible without requiring many panes.

### Terminal 1: MQTT + Web UI

```bash
cd ~/Documents/FABLE
./evaluation/netwaggle/1_server_ui_mqtt.sh
```

Open <http://localhost:8080>. This terminal shows web UI and broker logs.

### Terminal 2: NetWaggle / Mininet

```bash
cd ~/Documents/FABLE
./evaluation/netwaggle/2_netwaggle_network.sh --profile smoke_fixed_latency
```

This starts the anchor containers, attaches them to Mininet, starts latency
probes for the web UI NetWaggle panel, and tails NetWaggle/probe logs. Use
`--profile good_network` or another profile for replay experiments.

### Terminal 3: Replay + detector containers

```bash
cd ~/Documents/FABLE
./evaluation/netwaggle/3_replay_containers.sh
```

This starts the replay, detector, and CE containers in the foreground and enables
visible publish logging by default. Once replay is started from the web UI, you
should see lines such as:

```text
[publish:local] service=zed topic=data count=30 bytes=...
[publish:net] service=yolo topic=/debug/.../analytics/yolo/frame count=1 bytes=...
[publish:net] service=audio_detector topic=/debug/.../audio_detector/status count=1 bytes=...
[publish:net] service=zed topic=/replay/status/zed/... count=1 bytes=...
```

The local publish lines are raw replay-to-detector IPC within a logical node.
The net publish lines are MQTT traffic that should traverse NetWaggle.

To reduce log volume:

```bash
IOBT_LOG_LOCAL_PUBLISH_EVERY_N=120 ./evaluation/netwaggle/3_replay_containers.sh
```

or disable publish logs entirely:

```bash
./evaluation/netwaggle/3_replay_containers.sh --quiet-publish-logs
```

## 8. Three-terminal workflow with automatic debug capture

For day-to-day debugging, use the three-terminal scripts. They now save the same
information you would otherwise gather manually.

Terminal 1 starts MQTT and the web UI:

```bash
cd ~/Documents/FABLE
./evaluation/netwaggle/1_server_ui_mqtt.sh --new-run
```

Terminal 2 starts NetWaggle, Mininet, anchors, latency probes, and MQTT tracing:

```bash
cd ~/Documents/FABLE
./evaluation/netwaggle/2_netwaggle_network.sh --profile smoke_fixed_latency
```

Terminal 3 starts replay/detector/CE containers with publish logs enabled:

```bash
cd ~/Documents/FABLE
./evaluation/netwaggle/3_replay_containers.sh
```

All three terminals share the same run directory through:

```text
runs/netwaggle/current
```

Useful saved files include:

```text
runs/netwaggle/current/events.log
runs/netwaggle/current/logs/terminal1_server_ui_mqtt.log
runs/netwaggle/current/logs/terminal3_replay_containers.log
runs/netwaggle/current/logs/server.compose_logs.final.log
runs/netwaggle/current/logs/replay.compose_logs.final.log
runs/netwaggle/current/netwaggle_runner.log
runs/netwaggle/current/mqtt_trace.jsonl
runs/netwaggle/current/compose/server.rendered.yaml
runs/netwaggle/current/compose/netwaggle.rendered.yaml
runs/netwaggle/current/docker/docker_ps_snapshot.txt
runs/netwaggle/current/inspect/*.inspect.json
runs/netwaggle/current/api/state.json
runs/netwaggle/current/api/netwaggle.json
runs/netwaggle/current/api/messages_1000.json
runs/netwaggle/current/netns/*.netns.txt
runs/netwaggle/current/metrics/link_stats.snapshot.jsonl
runs/netwaggle/current/timelines/replay_detector_timeline.grep.txt
runs/netwaggle/current/timelines/mqtt_topic_summary.txt
```

The replay timeline is especially useful for late detector/model-load bugs. It
pulls out lines such as YOLO model load start/finish, local IPC subscription,
replay complete events, and `[publish:local]` / `[publish:net]` messages.

You can collect a snapshot at any time without stopping the run:

```bash
cd ~/Documents/FABLE
./evaluation/netwaggle/collect_debug_snapshot.sh
```

To create a tarball you can upload/share:

```bash
cd ~/Documents/FABLE
./evaluation/netwaggle/collect_debug_snapshot.sh
tar -czf netwaggle_debug_current.tar.gz -C runs/netwaggle current
```

`stop_stack.sh` also collects one final snapshot before shutting down containers
and Mininet.
