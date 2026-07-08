# Minimal IoBT replay + local detector + complex-event starter

This is a small working subset carved out of the larger IoBT-MAX replay/container repository.
It keeps only the pieces needed for this architecture:

```text
real/replayed sensor data → per-node IPC (*.ipc) → local primitive detector → MQTT → complex-event detector
```

Raw/high-bandwidth data stays local to a node emulation directory such as `/tmp/iobt-orin11`.
Only compact detector outputs are published over MQTT.

## What is included

```text
compose.server.yaml                         # minimal Mosquitto broker
server/mosquitto.conf                       # anonymous local MQTT config
lib/iobt_max_service.py                     # shared IoBT-MAX service base class
setup/generate_compose.py                   # scenario compose generator
setup/zed_settings/                         # ZED calibration/settings files

tools/replay_control.py                     # publishes /replay/config and /replay/sync
tools/mqtt_tail.py                          # quick MQTT topic monitor

services/replay/zed/                        # x86 ZED SVO replay container
services/replay/respeaker/                  # x86 ReSpeaker FLAC replay container
services/replay/gps/                        # x86 GPS replay container

services/analytics/yolo_detector/           # local ZED IPC → MQTT YOLO detections
services/analytics/audio_detector/          # local ReSpeaker IPC → MQTT audio events
services/orchestration/complex_event_detector/ # starter MQTT complex-event detector
```

## Expected data layout

The compose generator expects the current date-folder layout used by the original `local-containers/services/setup/setup.py` script:

```text
<data-dir>/<YYYYMMDD>/
  orin11/
    <scenario>_dvpg_gq_orin_11_zed.svo2
    <scenario>_dvpg_gq_orin_11_zed.csv
    <scenario>_dvpg_gq_orin_11_respeaker.flac
    <scenario>_dvpg_gq_orin_11_respeaker.csv
  orin12/
    ...
  GPS/
    <object>/
      <scenario>_*_gps.csv
```

For example, scenario `20260414_134838` should live under:

```text
<data-dir>/20260414/
```

## Quick start

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the MQTT broker:

```bash
docker compose -f compose.server.yaml up -d
```

Generate a scenario compose file:

```bash
python3 setup/generate_compose.py \
  --scenario 20260414_134838 \
  --data-dir /absolute/path/to/data \
  --compose-out compose.generated.20260414_134838.yaml
```

Build the generated replay/detector containers:

```bash
docker compose -f compose.generated.20260414_134838.yaml build
```

Start the containers:

```bash
docker compose -f compose.generated.20260414_134838.yaml up --remove-orphans
```

In another terminal, trigger replay:

```bash
source .venv/bin/activate
python3 tools/replay_control.py \
  --scenario 20260414_134838 \
  --start 0 \
  --end -1 \
  --sync-delay 5
```

Watch MQTT messages:

```bash
source .venv/bin/activate
python3 tools/mqtt_tail.py --topic '#'
```

Useful focused topic filters:

```bash
python3 tools/mqtt_tail.py --topic '/+/analytics/yolo/bbox'
python3 tools/mqtt_tail.py --topic '/+/audio_detector/detections'
python3 tools/mqtt_tail.py --topic '/complex_events/#'
python3 tools/mqtt_tail.py --topic '/replay/#'
```

## CPU-only / audio-only smoke test

If you do not have a CUDA-capable host or do not want to build the ZED container yet, generate only ReSpeaker/audio/CE services:

```bash
python3 setup/generate_compose.py \
  --scenario 20260414_134838 \
  --data-dir /absolute/path/to/data \
  --no-zed \
  --no-yolo \
  --compose-out compose.audio_only.yaml

docker compose -f compose.audio_only.yaml build
docker compose -f compose.audio_only.yaml up --remove-orphans
```

Then send replay control:

```bash
python3 tools/replay_control.py --scenario 20260414_134838 --start 0 --end -1 --sync-delay 5
```

## ZED/GPU notes

The ZED replay image uses `stereolabs/zed:5.4-runtime-cuda12.8-ubuntu22.04` and requires the NVIDIA Container Toolkit.
The generated compose uses:

```yaml
gpus: all
privileged: true
```

By default, ZED replay does **not** publish RGB/depth over MQTT. It publishes raw frames over IPC for colocated analytics. To allow RGB/depth MQTT debug streams, pass:

```bash
python3 setup/generate_compose.py ... --debug-raw-mqtt
```

## Plumbing test without YOLO inference

To check ZED IPC → YOLO container → MQTT without loading YOLO, generate with:

```bash
python3 setup/generate_compose.py \
  --scenario 20260414_134838 \
  --data-dir /absolute/path/to/data \
  --load-yolo-model false
```

The patched YOLO app will publish synthetic `class=test` detections when `LOAD_MODEL=false`.

## Architecture details

Each emulated node gets a private host tmp directory:

```text
/tmp/iobt-orin11
/tmp/iobt-orin12
```

Inside the containers for that node, the directory is mounted as `/tmp`, so local IPC sockets line up:

```text
zed replay container:        /tmp/zed.ipc
local yolo detector:         /tmp/zed.ipc

respeaker replay container:  /tmp/respeaker.ipc
local audio detector:        /tmp/respeaker.ipc
```

On the host, they are isolated by node:

```text
/tmp/iobt-orin11/zed.ipc
/tmp/iobt-orin11/respeaker.ipc
/tmp/iobt-orin12/zed.ipc
/tmp/iobt-orin12/respeaker.ipc
```

Primitive detector MQTT topics:

```text
/<node>/analytics/yolo/bbox
/<node>/audio_detector/detections
/<object>_replay/gps
```

Starter complex-event output:

```text
/complex_events/demo
```

## Cleanup

```bash
docker compose -f compose.generated.20260414_134838.yaml down --remove-orphans
docker compose -f compose.server.yaml down
sudo rm -rf /tmp/iobt-orin* /tmp/iobt-gps
```
