# Physical replay architecture

The Raspberry Pi is the logical sensing node even when its evidence comes from
a recording rather than attached capture hardware. FABLE source ownership is
therefore assigned to `physical_rpi`; the host PC is never recorded as the
camera or microphone source merely because it stores or transmits the bytes.

## Supported ingress modes

1. **Pi-local recording**: copy the bounded recording (or a scenario shard) to
   the Pi before the run. Replay reads it locally and publishes raw evidence on
   Pi-local IPC. This is the cleanest raw-data-locality experiment.
2. **Host-fed recording**: the PC streams a bounded recording to a Pi receiver
   over the physical Ethernet link. The receiver timestamps, sequences, and
   buffers it as `physical_rpi_camera_replay` or
   `physical_rpi_microphone_replay`. The source remains logically Pi-owned.

Host-fed replay must use a bounded receiver buffer and explicit backpressure;
mounting the host filesystem on the Pi would make storage convenient but would
not exercise the physical ingress link in a controlled way.

## Provider-output modes

Keep these modes separate in manifests and results:

- **REAL_PROVIDER**: the Pi emits raw replay locally and an actual YOLO provider
  executes on its selected compute node. If that node is the Jetson, raw video
  crosses the Pi-to-Jetson link and violates `raw_data_must_remain_local=True`
  unless the experiment explicitly relaxes that policy.
- **REPLAYED_PROVIDER_OUTPUT**: the Pi replays recorded YOLO detections. This is
  valid for network/control experiments, but it does not measure YOLO compute,
  accuracy, startup, or GPU behavior.

The output mode must be included in run metadata so replayed detections cannot
be mistaken for real inference.

## Physical and emulated network boundaries

The host-to-Pi recording feed uses the real Ethernet path. Detection/result
egress from the Pi must enter the Mininet-controlled path through an explicit
gateway or tunnel before reaching the broker/orchestrator. Merely publishing
from the Pi directly to the host broker uses the same physical Ethernet link
and bypasses Mininet.

A practical topology is:

```text
host recording sender --physical Ethernet--> Pi replay/receiver
                                             |
                                             +-- local raw IPC --> local provider
                                             |
                                             +-- detection MQTT --> tunnel/gateway
                                                                      |
                                                               Mininet namespace
                                                                      |
                                                               broker/orchestrator
```

Use separate addresses/ports for physical ingress and emulated egress. Packet
captures should be taken on both boundaries to demonstrate which path each
payload used.

## Current constraints

- The Pi can run CPU audio and geometry providers. YOLO on the Pi requires a
  separately measured CPU/TFLite/ONNX realization; catalog GPU profiles must
  not be overcommitted for a real-provider experiment.
- The Jetson can run GPU providers, but remote placement of YOLO changes the
  raw-data movement policy and network path being evaluated.
- The existing desktop replay compose stack uses shared local IPC directories
  and logical Docker/Mininet nodes. It cannot be treated as a physical-Pi run
  until its replay process or receiver is running on the Pi and egress is
  routed through the explicit Mininet gateway.

## Single-slot Pi staging

Only the one video assigned to the Pi is staged there. The PC retains the full
dataset and replays every remaining source. Use the host-side staging command:

```bash
.venv/bin/python scripts/stage_rpi_replay.py /absolute/path/scenario.svo2 \
  --experiment-id EXPERIMENT_ID --scenario-id SCENARIO_ID
```

The managed slot is `${FABLE_RPI_ROOT:-/opt/fable}/replay-cache`. The command checks
free space, transfers through a temporary file, verifies SHA-256, atomically
promotes the asset to `current.<extension>`, and writes `current.json`. It
removes the previous managed video and any managed extracted-frame directory
when replacing the slot. It never cleans files outside that directory.

## Three-device execution case

`scripts/run_physical_three_node_case.sh` prepares and runs the recommended
Route convoy recording `20241008-route-convoy-1-r012` with this placement:

- `orin1` recording: one left-camera MP4 staged and served by the physical Pi;
- `orin1` YOLO: real YOLOv8n CUDA inference on the physical Jetson;
- `orin4` and `orin7`: ordinary replay/provider services on the current PC;
- orchestration, semantic runtime, broker, and explicit subnet relay: current PC.

The suite stops the desktop `zed-orin1` and `yolo-orin1` services before the
cell, preventing duplicate evidence. The physical worker participates in the
normal `/replay/config` readiness barrier and `/replay/sync` start barrier, and
publishes the usual `/<node>/analytics/yolo/bbox` payloads. Run with:

```bash
FABLE_PHYSICAL_IDENTITY_FILE=/path/to/key \
  scripts/run_physical_three_node_case.sh
```
