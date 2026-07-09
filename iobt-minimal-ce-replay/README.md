# IoBT Minimal CE Replay Starter

This repo is a trimmed replay + detector + complex-event testbed carved out of the larger IoBT-MAX codebase.

The recommended workflow is **frontend mode**:

```text
Terminal 1: compose.server.yaml  -> MQTT broker + lightweight web UI
Terminal 2: compose.replay.yaml  -> persistent replay + detector + CE containers
Browser:    web UI              -> selects scenario and sends /replay/config + /replay/sync
```

The web UI does **not** read the sensor files directly. It only sends MQTT replay commands. The replay containers mount your SSD parent folders, resolve the selected scenario ID to matching files, and then replay those files.

## Data roots

By default, the replay stack searches these parent paths:

```text
/media/brianw/Extreme SSD/West Point Experimentation
/media/brianw/Extreme SSD/GQ Data
```

Expected layout:

```text
<root>/<YYYYMMDD>/<orinX>/<scenario_id>...zed...
<root>/<YYYYMMDD>/<orinX>/<scenario_id>...respeaker...
<root>/<YYYYMMDD>/GPS/<object>/<scenario_id>...gps...
```

Example:

```text
/media/brianw/Extreme SSD/West Point Experimentation/20260413/orin11/20260413_134838_..._zed.svo2
/media/brianw/Extreme SSD/West Point Experimentation/20260413/orin11/20260413_134838_..._respeaker.flac
/media/brianw/Extreme SSD/West Point Experimentation/20260413/GPS/car1/20260413_134838_...gps.csv
```

A scenario ID such as `20260413_134838` is resolved by extracting the date prefix `20260413`, then searching the mounted roots for the matching date folder.

To override the default roots:

```bash
python3 setup/generate_replay_compose.py \
  --data-dir "/media/brianw/Extreme SSD/West Point Experimentation" \
  --data-dir "/media/brianw/Extreme SSD/GQ Data" \
  --compose-out compose.replay.yaml
```

or:

```bash
export IOBT_HOST_DATA_ROOTS="/path/to/root1:/path/to/root2"
```

## Scenario catalog

To list valid experiments and their inferred start/end information:

```bash
python3 setup/scenario_catalog.py --output-dir generated
```

This writes:

```text
generated/scenario_catalog.json
generated/scenario_catalog.csv
```

When the server is running, you can also open:

```text
http://localhost:8080/scenarios
```

The catalog includes scenario IDs, source root, date folder, available modalities, device folders, file counts, and best-effort observed start/end datetimes when timestamp CSVs are available.

## Frontend mode: two-terminal workflow

### Terminal 1: start MQTT + web UI

```bash
docker compose -f compose.server.yaml up --build
```

Open the frontend:

```text
http://localhost:8080
```

Useful pages:

```text
http://localhost:8080            # replay controls and MQTT tail
http://localhost:8080/device     # per-device YOLO/audio display
http://localhost:8080/scenarios  # scenario catalog
```

### Terminal 2: generate and start replay containers

Generate `compose.replay.yaml` using one of the presets below, then run it in the foreground:

```bash
docker compose -f compose.replay.yaml up --build
```

### Browser: start replay

In the web UI, enter a full scenario ID, for example:

```text
20260413_134838
```

Then click **Start replay**.

The UI publishes:

```text
/replay/config
/replay/sync
```

The replay containers receive those messages, find matching files under the mounted roots, and start replay.

If a replay container prints `Waiting for sync on /replay/sync...` for a long time, click **Start replay** again. That usually means the child replay process subscribed after the first non-retained sync message was published.

## Default replay setting: full stack

By default, `generate_replay_compose.py` now includes:

```text
ZED replay:          enabled, GPU requested
YOLO detector:       enabled, model loaded, GPU requested
ReSpeaker replay:    enabled
Audio detector:      enabled
GPS replay:          enabled
CE detector:         enabled
```

Run all discovered devices with the full default stack:

```bash
python3 setup/generate_replay_compose.py \
  --compose-out compose.replay.yaml

docker compose -f compose.replay.yaml up --build
```

Run only one device with the full default stack:

```bash
python3 setup/generate_replay_compose.py \
  --device orin11 \
  --compose-out compose.replay.yaml

docker compose -f compose.replay.yaml up --build
```

The generated stack will still include one shared `gps-replay` and one `complex-event-detector` unless you disable them.

## Preset: ZED + YOLO only

Use this when debugging video and YOLO. It disables ReSpeaker, audio detector, GPS, and CE.

```bash
python3 setup/generate_replay_compose.py \
  --device orin11 \
  --yolo-debug \
  --compose-out compose.replay.yaml

docker compose -f compose.replay.yaml up --build
```

This runs:

```text
zed-replay-orin11
yolo-detector-orin11
```

and enables:

```text
YOLO model loading
YOLO annotated debug images
YOLO frame probe under /debug/<node>/analytics/yolo/frame
YOLO debug status under /debug/<node>/analytics/yolo/status
```

Open:

```text
http://localhost:8080/device
```

Select `orin11`. The device page should tell you whether YOLO is receiving decoded images, whether the model loaded, and whether the latest frames contain detections.

A less-instrumented ZED+YOLO-only run:

```bash
python3 setup/generate_replay_compose.py \
  --device orin11 \
  --no-respeaker \
  --no-audio-detector \
  --no-gps \
  --no-ce \
  --compose-out compose.replay.yaml

docker compose -f compose.replay.yaml up --build
```

## Preset: audio replay + audio detector only

Use this when debugging ReSpeaker and audio detections. It disables ZED, YOLO, GPS, and CE.

```bash
python3 setup/generate_replay_compose.py \
  --device orin11 \
  --no-zed \
  --no-yolo \
  --no-gps \
  --no-ce \
  --compose-out compose.replay.yaml

docker compose -f compose.replay.yaml up --build
```

This runs:

```text
respeaker-replay-orin11
audio-detector-orin11
```

Actual audio detections publish to:

```text
/<node>/audio_detector/detections
```

By default, synthetic status topics are disabled. If you need debug status, add:

```bash
--audio-debug-status
```

or:

```bash
--detector-debug-status
```

Debug status publishes under `/debug/...`, not under the normal event topics.

## Preset: ZED + YOLO + audio

This is the normal single-device multimodal run, without GPS or CE:

```bash
python3 setup/generate_replay_compose.py \
  --device orin11 \
  --no-gps \
  --no-ce \
  --compose-out compose.replay.yaml

docker compose -f compose.replay.yaml up --build
```

This runs:

```text
zed-replay-orin11
yolo-detector-orin11
respeaker-replay-orin11
audio-detector-orin11
```

Add debug visualization if needed:

```bash
python3 setup/generate_replay_compose.py \
  --device orin11 \
  --yolo-debug-images \
  --yolo-frame-debug \
  --no-gps \
  --no-ce \
  --compose-out compose.replay.yaml
```

## Preset: full multimodal + CE for one device

```bash
python3 setup/generate_replay_compose.py \
  --device orin11 \
  --compose-out compose.replay.yaml

docker compose -f compose.replay.yaml up --build
```

This runs:

```text
zed-replay-orin11
yolo-detector-orin11
respeaker-replay-orin11
audio-detector-orin11
gps-replay
complex-event-detector
```

## Preset: full multimodal + CE for all discovered devices

```bash
python3 setup/generate_replay_compose.py \
  --compose-out compose.replay.yaml

docker compose -f compose.replay.yaml up --build
```

This discovers all device folders such as `orin3`, `orin11`, `orin14`, etc. Use this only after the one-device case works, because YOLO containers can be expensive.

## GPU behavior

ZED replay imports `pyzed.sl`, which requires CUDA libraries inside the container. Therefore ZED replay normally needs GPU access.

Default behavior:

```text
ZED replay:    gpus: all
YOLO detector: gpus: all
```

To force CPU/no Compose GPU reservation for YOLO:

```bash
python3 setup/generate_replay_compose.py \
  --device orin11 \
  --no-yolo-gpu \
  --compose-out compose.replay.yaml
```

To disable ZED GPU reservation, mainly for non-ZED/audio-only testing:

```bash
python3 setup/generate_replay_compose.py \
  --no-zed-gpu \
  --compose-out compose.replay.yaml
```

If Docker says:

```text
could not select device driver "" with capabilities: [[gpu]]
```

then NVIDIA Container Toolkit is not configured correctly for Docker. Test with:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
```

If needed:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## MQTT topics

High-bandwidth data stays node-local over IPC:

```text
ZED replay       -> /tmp/iobt-orin11/zed.ipc       -> YOLO detector
ReSpeaker replay -> /tmp/iobt-orin11/respeaker.ipc -> audio detector
```

Compact outputs go over MQTT:

```text
/<node>/analytics/yolo/bbox
/<node>/audio_detector/detections
/<object>_replay/gps
/complex_events/demo
```

Optional debug topics are clearly separated:

```text
/debug/<node>/analytics/yolo/frame
/debug/<node>/analytics/yolo/status
/debug/<node>/audio_detector/status
```

## Useful MQTT monitors

After installing Python requirements locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Watch everything:

```bash
python3 tools/mqtt_tail.py --topic '#'
```

Focused monitors:

```bash
python3 tools/mqtt_tail.py --topic '/replay/#'
python3 tools/mqtt_tail.py --topic '/+/analytics/yolo/bbox'
python3 tools/mqtt_tail.py --topic '/+/audio_detector/detections'
python3 tools/mqtt_tail.py --topic '/complex_events/#'
python3 tools/mqtt_tail.py --topic '/debug/#'
```

## Common troubleshooting

### ZED says `ImportError: libcuda.so.1`

The ZED SDK requires CUDA libraries. Use the default GPU-enabled generator or explicitly add GPU support:

```bash
python3 setup/generate_replay_compose.py \
  --device orin11 \
  --zed-gpu \
  --compose-out compose.replay.yaml
```

Also confirm Docker GPU access works:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
```

### ZED says `Waiting for sync on /replay/sync...`

The replay child process is waiting for the web UI sync message. Click **Start replay** again after that line appears, or use:

```bash
python3 tools/replay_control.py \
  --scenario 20260413_134838 \
  --start 0 \
  --end -1 \
  --sync-delay 3
```

### YOLO shows no detections

Use the YOLO debug preset:

```bash
python3 setup/generate_replay_compose.py \
  --device orin11 \
  --yolo-debug \
  --compose-out compose.replay.yaml
```

Then open:

```text
http://localhost:8080/device
```

Check whether the frame probe shows decoded frames. If frames are arriving but `last_dets=0`, YOLO is running but not detecting objects in the current frames. If no frame probe appears, ZED replay is not publishing images to `/tmp/zed.ipc`.

### Audio detections appear but no audio status appears

That is expected by default. Synthetic detector status is disabled unless you explicitly enable:

```bash
--audio-debug-status
```

Normal event output is:

```text
/<node>/audio_detector/detections
```

## Legacy scenario-specific compose

`setup/generate_compose.py` may still exist from earlier versions. It creates scenario-specific compose files like:

```bash
python3 setup/generate_compose.py \
  --scenario 20260413_134838 \
  --compose-out compose.generated.20260413_134838.yaml
```

For normal use, prefer the persistent frontend workflow:

```text
compose.server.yaml + compose.replay.yaml + web UI scenario selection
```

## Optional NetWaggle network emulation

This repository can be run with the sibling `../netwaggle` package to route
network-scoped MQTT traffic through a Mininet/TC topology while leaving local
replay-to-detector IPC streams on the same logical node. See `NETWAGGLE.md` and
`../netwaggle/README.md` for the run sequence.
