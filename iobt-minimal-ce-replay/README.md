# IoBT Minimal CE Replay Starter

This is a trimmed replay + analytics + complex-event testbed carved out of the larger IoBT-MAX repo.

The current recommended workflow uses **persistent replay containers**:

```text
compose.server.yaml    -> MQTT broker + lightweight web UI
compose.replay.yaml    -> persistent replay supervisors + optional local detectors + CE detector
web UI / MQTT          -> selects scenario dynamically with /replay/config and /replay/sync
```

The web UI does **not** read files directly. It only sends MQTT commands. The replay containers mount your parent SSD folders and choose the matching `<YYYYMMDD>/<orinX>/...` files when a scenario is selected.

## Expected data layout

Your data roots are assumed to be:

```text
/media/brianw/Extreme SSD/West Point Experimentation/
/media/brianw/Extreme SSD/GQ Data/
```

Under those roots, the expected layout is:

```text
<root>/20260413/orin11/<scenario>_*_zed.svo2
<root>/20260413/orin11/<scenario>_*_zed.csv
<root>/20260413/orin11/<scenario>_*_respeaker.flac
<root>/20260413/orin11/<scenario>_*_respeaker.csv
<root>/20260413/GPS/<object>/<scenario>_*gps.csv
```

Examples:

```text
/media/brianw/Extreme SSD/West Point Experimentation/20260413/orin11/20260413_134838_..._zed.svo2
/media/brianw/Extreme SSD/GQ Data/20250812/orin11/20250812_165739_..._respeaker.flac
```

## 1. Start MQTT + web UI

```bash
docker compose -f compose.server.yaml up -d --build
```

Open:

```text
http://localhost:8080
```

This starts only the broker and web UI. It does not replay sensor data by itself.

## 2. Generate the persistent replay stack

Generate `compose.replay.yaml` once from your available device folders:

```bash
python3 setup/generate_replay_compose.py --compose-out compose.replay.yaml
```

This scans your two default SSD parent roots, discovers node folders such as `orin11`, `orin12`, etc., and writes a scenario-agnostic compose file.

To force specific nodes:

```bash
python3 setup/generate_replay_compose.py \
  --nodes orin11 orin12 \
  --compose-out compose.replay.yaml
```

To use custom parent data roots:

```bash
python3 setup/generate_replay_compose.py \
  --data-dir "/media/brianw/Extreme SSD/West Point Experimentation" \
  --data-dir "/media/brianw/Extreme SSD/GQ Data" \
  --compose-out compose.replay.yaml
```

Or set:

```bash
export IOBT_HOST_DATA_ROOTS="/path/to/root1:/path/to/root2"
```

## 3. Start persistent replay containers

Audio/GPS/CE-first smoke test:

```bash
python3 setup/generate_replay_compose.py \
  --no-zed \
  --compose-out compose.replay.yaml

docker compose -f compose.replay.yaml up -d --build
```

Full replay supervisors, but without YOLO detector containers:

```bash
python3 setup/generate_replay_compose.py \
  --compose-out compose.replay.yaml

docker compose -f compose.replay.yaml up -d --build
```

Include local YOLO detector containers as well:

```bash
python3 setup/generate_replay_compose.py \
  --include-yolo \
  --load-yolo-model true \
  --compose-out compose.replay.yaml

docker compose -f compose.replay.yaml up -d --build
```

ZED replay requires NVIDIA Container Toolkit and the Stereolabs ZED runtime image. If that is not installed, start with `--no-zed`.

## 4. Start replay from the web UI

In the web UI, enter a full scenario prefix such as:

```text
20260413_134838
```

Then click **Start replay**.

The UI publishes:

```text
/replay/config  {"scenario":"20260413_134838", "start_time":0, "end_time":-1}
/replay/sync    {"scenario":"20260413_134838", "start_at": <future wall time>}
```

The persistent replay supervisors then search the mounted data roots for:

```text
/data_roots/west_point/20260413/orin11/...
/data_roots/gq/20260413/orin11/...
```

and start/restart the underlying replay app for the selected files.

You do **not** need a new compose file for every scenario. Regenerate `compose.replay.yaml` only when you change parent roots, add/remove device folders, or change which detector services you want active.

## 5. Watch outputs

The web UI tails these by default:

```text
/replay/#
/+/analytics/yolo/bbox
/+/audio_detector/detections
/complex_events/#
/trackers/#
/geospatialdetections/#
```

From the terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 tools/mqtt_tail.py --topic '#'
```

Useful focused monitors:

```bash
python3 tools/mqtt_tail.py --topic '/replay/#'
python3 tools/mqtt_tail.py --topic '/+/audio_detector/detections'
python3 tools/mqtt_tail.py --topic '/+/analytics/yolo/bbox'
python3 tools/mqtt_tail.py --topic '/complex_events/#'
```

## Architecture

High-bandwidth data stays node-local:

```text
ZED replay       -> /tmp/iobt-orin11/zed.ipc       -> YOLO detector
ReSpeaker replay -> /tmp/iobt-orin11/respeaker.ipc -> audio detector
```

Compact detector/event streams go over MQTT:

```text
/<node>/analytics/yolo/bbox
/<node>/audio_detector/detections
/<object>_replay/gps
/complex_events/demo
```

The replay supervisors are the small piece that make persistent containers possible. They listen for `/replay/config`, resolve the scenario into the right SSD/date/node files, create in-container symlinks, and start/restart the original replay app.

## Legacy scenario-specific generator

`setup/generate_compose.py` is still included because it is useful for debugging one scenario with an explicit date-folder mount:

```bash
python3 setup/generate_compose.py \
  --scenario 20260413_134838 \
  --compose-out compose.generated.20260413_134838.yaml
```

But for normal use, prefer `setup/generate_replay_compose.py` + `compose.replay.yaml`.
