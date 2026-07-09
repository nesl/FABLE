# NetWaggle for FABLE replay

NetWaggle integrates the FABLE `iobt-minimal-ce-replay` stack with a Mininet/TC
network emulator. The initial integration is intentionally wired-only and
single-host focused: local high-bandwidth replay-to-detector streams remain on
local IPC, while MQTT traffic between logical node bundles is routed through an
emulated Mininet topology.

## Architecture

The replay stack already separates local and network communication:

- ZED replay -> YOLO detector uses `/tmp/zed.ipc` inside the logical node.
- ReSpeaker replay -> audio detector uses `/tmp/respeaker.ipc` inside the logical node.
- MQTT carries replay control, compact detections, status, GPS, and CE outputs.

NetWaggle preserves that split. It groups colocated services into one logical
node namespace, then attaches that namespace to Mininet:

```text
netwaggle-node-orin11
  ├── zed-replay-orin11
  ├── yolo-detector-orin11
  ├── respeaker-replay-orin11
  └── audio-detector-orin11

netwaggle-node-cloud1
  └── complex-event-detector
```

The host MQTT broker remains outside Mininet for the first version. NetWaggle
creates a host-side gateway interface, default `10.255.0.1/16`, and routes the
logical node containers to that broker through Mininet.

## Directory layout

```text
FABLE/
├── iobt-minimal-ce-replay/
└── netwaggle/
    ├── netwaggle/                 # Python package
    │   ├── runner.py              # Mininet topology runner + Docker namespace attach
    │   ├── docker_attach.py       # veth/namespace attach helpers
    │   ├── topology.py            # JSON config loader
    │   ├── metrics.py             # interface/qdisc JSONL metrics
    │   └── cleanup.py             # best-effort cleanup
    ├── configs/
    │   ├── fable_single_host.json
    │   └── profiles/
    └── scripts/
        ├── make_netwaggle_compose.py
        ├── netwaggle_up.sh
        ├── netwaggle_down.sh
        ├── inspect_routes.sh
        └── mqtt_trace.py
```

## Requirements

Run on a Linux host with:

- Docker Compose
- Mininet or Mininet-WiFi installed
- Open vSwitch
- `iproute2`, `tc`, `nsenter`, and root privileges
- Python packages: `pyyaml`, `paho-mqtt`

This cannot be fully tested in an unprivileged container because it needs root
network namespace and OVS/TC access.

## Quick start

From `FABLE/iobt-minimal-ce-replay`:

```bash
# 1. Generate or refresh the normal replay compose file.
python3 setup/generate_replay_compose.py --device orin11 --compose-out compose.replay.yaml

# 2. Convert the replay compose file into a NetWaggle-aware compose file.
python3 ../netwaggle/scripts/make_netwaggle_compose.py \
  --compose-in compose.replay.yaml \
  --compose-out compose.netwaggle.yaml \
  --node-map ../netwaggle/configs/fable_single_host.json

# 3. Start the normal host-side MQTT broker and web UI.
docker compose -f compose.server.yaml up --build

# 4. In another terminal, start the replay stack with shared node namespaces.
docker compose -f compose.netwaggle.yaml up --build

# 5. In another terminal, attach those namespaces to Mininet.
cd ../netwaggle
sudo PYTHONPATH=. python3 -m netwaggle.runner \
  --topology configs/fable_single_host.json \
  --profile configs/profiles/good_network.json \
  --no-cli --hold
```

Once NetWaggle is running, use the existing web UI or `tools/replay_control.py`
to start replay. Services should connect to MQTT at `10.255.0.1:1883`.

## Network profiles

Example profiles are in `configs/profiles/`:

- `good_network.json`
- `constrained_bandwidth.json`
- `high_latency_cloud.json`
- `lossy_edge.json`
- `cloud_degraded.json`

These replace the `links` section of `fable_single_host.json` at runtime.

## Metrics

Collect interface and qdisc stats:

```bash
sudo PYTHONPATH=. python3 -m netwaggle.metrics \
  --topology configs/fable_single_host.json \
  --out runs/netwaggle/link_stats.jsonl
```

Trace MQTT topic sizes from the host broker:

```bash
python3 scripts/mqtt_trace.py --host localhost --out runs/netwaggle/mqtt_trace.jsonl
```

The MQTT tracer classifies topics coarsely as `control`, `predicate_or_event`,
`artifact`, `continuation`, `debug`, or `other`. This is intentionally simple;
it is meant to bootstrap the network-cost metrics before the CE payload schema is
fully formalized.

## Cleanup

```bash
cd FABLE/netwaggle
sudo PYTHONPATH=. python3 -m netwaggle.cleanup --topology configs/fable_single_host.json
# or
./scripts/netwaggle_down.sh
```

## Design notes

This first integration does **not** route raw ZED or raw audio frames through
Mininet. That is deliberate. It keeps local sensor-to-detector paths local and
emulates only inter-node MQTT traffic. Raw-over-network should be added later as
an explicit baseline mode rather than as the default experiment path.

Mobility is also intentionally disabled for now. The topology is switch/TC based,
which is enough for the initial single-desktop experiments over bandwidth,
latency, loss, and cloud/edge degradation profiles.

## Easier evaluation bring-up

For day-to-day testing, prefer the wrappers in `FABLE/evaluation/netwaggle/`.
They start the host MQTT/UI services, NetWaggle anchor containers, Mininet
runner, optional MQTT tracing, and optionally the full replay stack.

Fast smoke test without full replay:

```bash
cd FABLE
./evaluation/netwaggle/start_stack.sh --profile smoke_fixed_latency --anchors-only
./evaluation/netwaggle/smoke_fixed_latency.sh
```

Full replay setup:

```bash
cd FABLE
./evaluation/netwaggle/start_stack.sh --profile good_network --regenerate-compose
./evaluation/netwaggle/list_scenarios.sh
./evaluation/netwaggle/run_replay_window.sh --scenario <SCENARIO_ID> --start 0 --end 30
```

Stop and clean up:

```bash
./evaluation/netwaggle/stop_stack.sh
```
