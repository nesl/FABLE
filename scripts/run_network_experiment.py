#!/usr/bin/env python3
"""Run one replay-relative FABLE network experiment."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any
import uuid
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from netwaggle.experiment import NetworkCondition, load_network_experiment
from netwaggle.topology import Link, NetWaggleTopology
from netwaggle.util import load_json

COMPOSE_BASE = (
    "docker", "compose", "--env-file", ".env", "-p", "fable-e2e",
    "-f", "deployment/compose.example.yaml",
)
CONTROL_SOCKET = Path("/run/netwaggle/fable-control.sock")
AGENT_CONTAINER = "fable-e2e-sensor1-agent"
EDGE_CONTAINER = "fable-e2e-edge1-agent"
CONTROLLER_CONTAINER = "fable-e2e-controller"
VISUALIZATION_CONTAINER = "fable-e2e-visualization"
SESSION_SOCKET = "/tmp/fable-session.sock"
REPLAY_BARRIER = "/tmp/fable-network-ready"
CONTROLLER_BARRIER = "/tmp/fable-controller-ready"
NETWAGGLE_PYTHON = "/usr/bin/python3"


@dataclass(frozen=True, slots=True)
class ScheduledChange:
    due_seconds: float
    affected_link_name: str
    values: dict[str, Any]
    phase: str


class _LinkStatePublisher:
    """Publish one condition to the controller and wait for its replan."""

    def __init__(self, host: str, port: int) -> None:
        import paho.mqtt.client as mqtt
        self._mqtt = mqtt
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"fable-experiment-{uuid.uuid4()}",
        )
        self._connected = threading.Event()
        self._pending: dict[str, tuple[threading.Event, dict[str, Any]]] = {}
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._host, self._port = host, int(port)

    def start(self) -> None:
        self._client.connect(self._host, self._port, 60)
        self._client.loop_start()
        if not self._connected.wait(10):
            raise TimeoutError("timed out connecting network experiment to MQTT")

    def close(self) -> None:
        self._client.disconnect()
        self._client.loop_stop()

    def publish(self, links: list[dict[str, Any]]) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        reply_to = f"fable/v1/runtime/link/ack/{request_id}"
        event, response = threading.Event(), {}
        self._pending[request_id] = (event, response)
        self._client.subscribe(reply_to, qos=1)
        payload = json.dumps({
            "request_id": request_id, "reply_to": reply_to, "links": links,
        }, sort_keys=True).encode()
        info = self._client.publish("fable/v1/runtime/link", payload, qos=1)
        info.wait_for_publish(timeout=10)
        if not event.wait(180):
            raise TimeoutError("controller did not acknowledge network-condition replan")
        self._pending.pop(request_id, None)
        if not response.get("ok"):
            raise RuntimeError(f"controller rejected network condition: {response.get('error')}")
        return response

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        if int(getattr(reason_code, "value", reason_code)) == 0:
            self._connected.set()

    def _on_message(self, client, userdata, message) -> None:
        row = json.loads(message.payload)
        pending = self._pending.get(str(row.get("request_id", "")))
        if pending is not None:
            pending[1].update(row)
            pending[0].set()


def schedule(conditions: tuple[NetworkCondition, ...], topology: NetWaggleTopology) -> tuple[ScheduledChange, ...]:
    baselines = {link.name: _baseline(link) for link in topology.links}
    changes: list[ScheduledChange] = []
    for condition in conditions:
        if condition.affected_link_name not in baselines:
            raise ValueError(f"unknown affected_link_name: {condition.affected_link_name}")
        changes.append(ScheduledChange(condition.starts_at_seconds, condition.affected_link_name, condition.values(), "apply"))
        if condition.remains_for_seconds is not None:
            changes.append(ScheduledChange(
                condition.starts_at_seconds + condition.remains_for_seconds,
                condition.affected_link_name,
                baselines[condition.affected_link_name],
                "restore",
            ))
    return tuple(sorted(changes, key=lambda item: (item.due_seconds, item.affected_link_name, item.phase)))


def _baseline(link: Link) -> dict[str, Any]:
    values: dict[str, Any] = {"connection_status": "connected"}
    if link.bw is not None: values["bandwidth_mbps"] = link.bw
    if link.delay is not None: values["latency_ms"] = float(link.delay.removesuffix("ms"))
    if link.jitter is not None: values["jitter_ms"] = float(link.jitter.removesuffix("ms"))
    if link.loss is not None: values["packet_loss_fraction"] = link.loss / 100.0
    return values


def _runtime_links(change: ScheduledChange, topology: NetWaggleTopology) -> list[dict[str, Any]]:
    link = next(row for row in topology.links if row.name == change.affected_link_name)
    by_switch: dict[str, list[str]] = {}
    for attachment in topology.attachments:
        by_switch.setdefault(attachment.switch, []).append(attachment.name)
    left, right = by_switch.get(link.src, []), by_switch.get(link.dst, [])
    if len(left) != 1 or len(right) != 1:
        return []
    available = change.values.get("connection_status", "connected") == "connected"
    common = {
        "latency_ms": float(change.values.get("latency_ms", 0)),
        "bandwidth_mbps": change.values.get("bandwidth_mbps"),
        "available": available,
    }
    return [
        {"source_node": left[0], "destination_node": right[0], **common},
        {"source_node": right[0], "destination_node": left[0], **common},
    ]


def _request(socket_path: Path, change: ScheduledChange, epoch: int) -> dict[str, Any]:
    document = {
        "schema_version": "netwaggle.dynamic_control.v1",
        "kind": "LINK_CONDITION",
        "target": change.affected_link_name,
        "conditions": change.values,
        "condition_epoch": epoch,
    }
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(60)
        client.connect(str(socket_path))
        client.sendall((json.dumps(document, sort_keys=True) + "\n").encode())
        payload = bytearray()
        while b"\n" not in payload:
            chunk = client.recv(65536)
            if not chunk: break
            payload.extend(chunk)
    response = json.loads(payload.split(b"\n", 1)[0])
    if response.get("validated") is not True:
        raise RuntimeError(response.get("reason", "NetWaggle rejected condition"))
    return response


def _run(command: tuple[str, ...] | list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, check=check, text=True, capture_output=capture)


def _interface_counters(container: str, interface: str = "netwaggle0") -> dict[str, int]:
    counters: dict[str, int] = {}
    for direction in ("rx", "tx"):
        result = _run([
            "docker", "exec", container, "cat",
            f"/sys/class/net/{interface}/statistics/{direction}_bytes",
        ], capture=True)
        counters[f"{direction}_bytes"] = int(result.stdout.strip())
    return counters


def _quantitative_summary(
    controller_log: Path, network_start: dict[str, int], network_end: dict[str, int],
) -> dict[str, Any]:
    rows = []
    for line in controller_log.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("type") == "execution_plan" and row.get("generated_at"):
            rows.append(row)
    active: set[tuple[str, str, tuple[str, ...]]] = set()
    seconds: dict[str, float] = {}
    previous: datetime | None = None
    for row in rows:
        when = datetime.fromisoformat(str(row["generated_at"]))
        if previous is not None:
            elapsed = max(0.0, (when - previous).total_seconds())
            for provider_id, _node_id, _sources in active:
                seconds[provider_id] = seconds.get(provider_id, 0.0) + elapsed
        actions = row.get("actions", {})
        for item in actions.get("stop", ()):
            active.discard((item["provider_id"], item["node_id"], tuple(item["source_ids"])))
        for action in ("start", "keep"):
            for item in actions.get(action, ()):
                active.add((item["provider_id"], item["node_id"], tuple(item["source_ids"])))
        previous = when
    if previous is not None:
        elapsed = max(0.0, (datetime.now(UTC) - previous).total_seconds())
        for provider_id, _node_id, _sources in active:
            seconds[provider_id] = seconds.get(provider_id, 0.0) + elapsed
    return {
        "network_interface": "sensor1:netwaggle0",
        "network_bytes": {
            key: network_end[key] - network_start[key] for key in sorted(network_start)
        },
        "provider_seconds": {key: round(value, 3) for key, value in sorted(seconds.items())},
        "total_provider_seconds": round(sum(seconds.values()), 3),
        "plan_updates": len(rows),
    }


def _session(
    container: str, action: str, request: Path | None = None, *, check: bool = True
) -> None:
    command = ["docker", "exec", container, "python", "scripts/session_supervisor.py", action,
               "--socket", SESSION_SOCKET]
    if request is not None:
        command.extend(("--request", "/workspace/FABLE/" + str(request.relative_to(ROOT))))
    _run(command, check=check)


def _plan_nodes(logs: str, reason: str) -> set[str]:
    """Return nodes in the latest controller plan for one reason."""
    found: set[str] | None = None
    for line in logs.splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            row = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if row.get("type") == "execution_plan" and row.get("reason") == reason:
            found = {str(step["node_id"]) for step in row.get("steps", ())}
    return found or set()


def _completion_seen(logs: str, event_name: str) -> bool:
    return any(
        f'"event": "{event_name}"' in line and '"schema_version": "fable.completion.v1"' in line
        for line in logs.splitlines()
    )


def _dotenv_value(path: Path, name: str) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == name:
            return value.strip()
    return None


def _wait_for(predicate, description: str, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate(): return
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for {description}")


def _set_visualization_session_status(path: Path, status: str) -> None:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        state["session"] = {**state.get("session", {}), "status": status}
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    except (OSError, json.JSONDecodeError):
        pass


class _DashboardRecorder:
    """Record one server-rendered dashboard without browser/SSH traffic."""

    def __init__(self, output: Path, name: str, snapshot_path: str, *,
                 expected_devices: int | None = None, fps: float = 4.0,
                 base_url: str = "http://127.0.0.1:5000",
                 browser_path: str | None = None) -> None:
        self.output = output
        self.expected_devices = (
            None if expected_devices is None else int(expected_devices)
        )
        self.fps = float(fps)
        self.base_url = base_url.rstrip("/")
        self.snapshot_path = snapshot_path
        self.browser_path = browser_path
        self.target = output / f"{name}.mp4"
        self.thumbnail = output / f"{name}.jpg"
        self.metadata = output / f"{name}.json"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._encoder: subprocess.Popen[bytes] | None = None
        self._recording_started: float | None = None
        self._session_started: float | None = None
        self._frames_written = 0

    def start(self) -> None:
        self._session_started = time.monotonic()
        self._thread = threading.Thread(target=self._record, daemon=True)
        self._thread.start()

    def _record(self) -> None:
        period = 1.0 / self.fps
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                metadata: object
                if self.expected_devices is not None:
                    with urlopen(self.base_url + "/api/devices", timeout=5) as response:
                        metadata = json.loads(response.read())
                    if self.expected_devices and len(metadata) < self.expected_devices:
                        self._stop.wait(min(period, 0.25))
                        continue
                else:
                    with urlopen(self.base_url + "/api/state", timeout=5) as response:
                        metadata = json.loads(response.read())
                if self.browser_path is not None:
                    screenshot = self.output / f".{self.target.stem}-browser.jpg"
                    completed = subprocess.run([
                        "google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
                        "--hide-scrollbars", "--window-size=1280,720",
                        "--run-all-compositor-stages-before-draw",
                        "--virtual-time-budget=1000", f"--screenshot={screenshot}",
                        self.base_url + self.browser_path,
                    ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=10)
                    image = screenshot.read_bytes() if completed.returncode == 0 and screenshot.is_file() else b""
                    screenshot.unlink(missing_ok=True)
                else:
                    with urlopen(self.base_url + self.snapshot_path, timeout=10) as response:
                        image = response.read()
                if not image:
                    continue
                self.thumbnail.write_bytes(image)
                self.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
                if self._encoder is None:
                    self._encoder = subprocess.Popen([
                        "ffmpeg", "-loglevel", "error", "-y",
                        "-f", "image2pipe", "-framerate", str(self.fps),
                        "-vcodec", "mjpeg", "-i", "-",
                        "-an", "-c:v", "libx264", "-preset", "veryfast",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.target),
                    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    self._recording_started = time.monotonic()
                assert self._encoder.stdin is not None
                elapsed = time.monotonic() - (self._recording_started or time.monotonic())
                target_frames = max(self._frames_written + 1, round(elapsed * self.fps) + 1)
                for _ in range(target_frames - self._frames_written):
                    self._encoder.stdin.write(image)
                self._frames_written = target_frames
                self._encoder.stdin.flush()
            except (OSError, TimeoutError, json.JSONDecodeError, BrokenPipeError):
                pass
            self._stop.wait(max(0.0, period - (time.monotonic() - started)))

    def stop(self) -> Path | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None
        if self._encoder is not None:
            if self._encoder.stdin is not None:
                try: self._encoder.stdin.close()
                except BrokenPipeError: pass
            try: self._encoder.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._encoder.terminate(); self._encoder.wait(timeout=5)
            self._encoder = None
        return self.target if self.target.is_file() and self.target.stat().st_size else None

    @property
    def recording_offset_seconds(self) -> float:
        if self._recording_started is None or self._session_started is None:
            return 0.0
        return max(0.0, self._recording_started - self._session_started)


def _browser_screenshot(url: str, target: Path) -> bool:
    completed = subprocess.run([
        "google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
        "--hide-scrollbars", "--window-size=1280,720",
        "--run-all-compositor-stages-before-draw", "--virtual-time-budget=1000",
        f"--screenshot={target}", url,
    ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
       timeout=10)
    return completed.returncode == 0 and target.is_file() and target.stat().st_size > 0


def _combine_dashboard_images(device: Path, ce: Path, target: Path) -> bool:
    completed = subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(device), "-i", str(ce),
        "-filter_complex",
        "[0:v]scale=960:540:force_original_aspect_ratio=decrease,"
        "pad=960:540:(ow-iw)/2:(oh-ih)/2:black[left];"
        "[1:v]scale=960:540:force_original_aspect_ratio=decrease,"
        "pad=960:540:(ow-iw)/2:(oh-ih)/2:black[right];"
        "[left][right]hstack=inputs=2[out]",
        "-map", "[out]", "-frames:v", "1", str(target),
    ], check=False)
    return completed.returncode == 0 and target.is_file() and target.stat().st_size > 0


class _ProgressCapture:
    """Save both dashboard views whenever the furthest CE progress advances."""

    def __init__(self, output: Path, base_url: str = "http://127.0.0.1:5000") -> None:
        self.output = output / "event-progress"
        self.base_url = base_url.rstrip("/")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_satisfied = -1
        self._sequence = 0

    def start(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def _watch(self) -> None:
        while not self._stop.is_set():
            try:
                with urlopen(self.base_url + "/api/state", timeout=5) as response:
                    state = json.loads(response.read())
                instances = state.get("instances", ())
                satisfied = max(
                    (int(row.get("progress", {}).get("satisfied", 0)) for row in instances),
                    default=0,
                )
                total = max(
                    (int(row.get("progress", {}).get("total", 0)) for row in instances),
                    default=int(state.get("counts", {}).get("nodes", 0)),
                )
                if satisfied > self._last_satisfied:
                    self._capture(state, satisfied, total)
                    self._last_satisfied = satisfied
            except (OSError, TimeoutError, json.JSONDecodeError, ValueError):
                pass
            self._stop.wait(0.2)

    def _capture(self, state: dict[str, Any], satisfied: int, total: int) -> None:
        stem = f"{self._sequence:02d}-satisfied-{satisfied}-of-{total}"
        device = self.output / f"{stem}-devices.jpg"
        ce = self.output / f"{stem}-ce.jpg"
        combined = self.output / f"{stem}-combined.jpg"
        with urlopen(self.base_url + "/api/devices-snapshot.jpg", timeout=10) as response:
            image = response.read()
        if image:
            device.write_bytes(image)
        _browser_screenshot(self.base_url + "/", ce)
        if device.is_file() and ce.is_file():
            _combine_dashboard_images(device, ce, combined)
        (self.output / f"{stem}.json").write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._sequence += 1

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None


def _stitch_dashboard_videos(device: Path, ce: Path, target: Path) -> bool:
    """Create a synchronized side-by-side presentation video with replay audio."""
    if not device.is_file() or not ce.is_file():
        return False
    completed = subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(device), "-i", str(ce),
        "-filter_complex",
        "[0:v]scale=960:540:force_original_aspect_ratio=decrease,"
        "pad=960:540:(ow-iw)/2:(oh-ih)/2:black,setpts=PTS-STARTPTS[left];"
        "[1:v]scale=960:540:force_original_aspect_ratio=decrease,"
        "pad=960:540:(ow-iw)/2:(oh-ih)/2:black,setpts=PTS-STARTPTS[right];"
        "[left][right]hstack=inputs=2:shortest=1[out]",
        "-map", "[out]", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(target),
    ], check=False)
    return completed.returncode == 0 and target.is_file() and target.stat().st_size > 0


def _mux_replay_audio(
    video: Path, manifest: dict[str, Any], manifest_path: Path,
    recording_offset_seconds: float,
) -> bool:
    """Add the best synchronized replay microphone track to a dashboard MP4."""
    audio_sources = [row for row in manifest.get("sources", ()) if row.get("modality") == "audio"]
    if not audio_sources or not video.is_file():
        return False
    source = next(
        (row for row in audio_sources if "mobile" in str(row.get("source_id", "")).lower()),
        audio_sources[0],
    )
    configured = Path(str(source["path"]))
    if configured.is_absolute():
        recording_root = Path(
            os.environ.get("RECORDING_ROOT")
            or _dotenv_value(ROOT / ".env", "RECORDING_ROOT") or ROOT / "replay"
        ).expanduser()
        gq_root = Path(
            os.environ.get("GQ_RECORDING_ROOT")
            or _dotenv_value(ROOT / ".env", "GQ_RECORDING_ROOT") or recording_root
        ).expanduser()
        try:
            audio = recording_root / configured.relative_to("/recordings")
        except ValueError:
            try:
                audio = gq_root / configured.relative_to("/gq-recordings")
            except ValueError:
                try:
                    audio = ROOT / ".fable/runs" / "replay-assets" / configured.relative_to("/replay-assets")
                except ValueError:
                    audio = configured
    else:
        audio = (manifest_path.parent / configured).resolve()
    if not audio.is_file():
        return False
    duration_result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        check=False, text=True, capture_output=True,
    )
    try:
        video_duration = float(duration_result.stdout.strip())
    except ValueError:
        return False
    origin = min(
        datetime.fromisoformat(str(row["event_time_start"]).replace("Z", "+00:00"))
        for row in manifest.get("sources", ())
    )
    audio_start = datetime.fromisoformat(
        str(source["event_time_start"]).replace("Z", "+00:00")
    )
    speed = float(manifest.get("speed", 1.0))
    source_delay = (audio_start - origin).total_seconds() / speed
    seek = max(0.0, (recording_offset_seconds - source_delay) * speed)
    delay = max(0.0, source_delay - recording_offset_seconds)
    temporary = video.with_name(f".{video.stem}-with-audio.mp4")
    command = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(video)]
    if seek:
        command.extend(("-ss", f"{seek:.6f}"))
    command.extend(("-i", str(audio)))
    audio_filters = []
    # Encode the pre-source gap as real silence rather than relying on a
    # positive MP4 audio-stream timestamp. Some players normalize independent
    # stream timestamps and otherwise play delayed replay audio too early.
    if delay:
        # Replay microphone assets are normalized to mono by the manifest
        # preparation path; this form also works with older FFmpeg releases.
        audio_filters.append(f"adelay={round(delay * 1000)}")
    if speed == 1.0:
        pass
    else:
        factors = []
        remaining = speed
        while remaining > 2:
            factors.append(2.0); remaining /= 2
        while remaining < .5:
            factors.append(.5); remaining /= .5
        factors.append(remaining)
        audio_filters.extend(f"atempo={value:g}" for value in factors)
    audio_filters.append("apad")
    command.extend((
        "-filter_complex", f"[1:a]{','.join(audio_filters)}[replay_audio]",
        "-map", "0:v:0", "-map", "[replay_audio]", "-c:v", "copy", "-c:a", "aac",
        "-t", f"{video_duration:.6f}", "-movflags", "+faststart", str(temporary),
    ))
    completed = subprocess.run(command, check=False)
    if completed.returncode or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        return False
    temporary.replace(video)
    return True


def _replay_duration_seconds(manifest: dict[str, Any]) -> float:
    """Return the available synchronized media window, including source delays."""
    starts: list[datetime] = []
    ends: list[datetime] = []
    for source in manifest.get("sources", ()):
        segments = source.get("segments") or (source,)
        for segment in segments:
            start = datetime.fromisoformat(
                str(segment["event_time_start"]).replace("Z", "+00:00")
            ) + timedelta(seconds=float(segment.get("offset_seconds", 0)))
            duration = segment.get("duration_seconds")
            if duration is not None:
                starts.append(start)
                ends.append(start + timedelta(seconds=float(duration)))
    if not starts or not ends:
        return 0.0
    return max(0.0, (max(ends) - min(starts)).total_seconds() / float(manifest.get("speed", 1)))


def _stop_netwaggle(process: subprocess.Popen[str]) -> None:
    """Stop the root-owned Mininet process from the unprivileged runner."""
    if process.poll() is not None:
        return
    _run(["sudo", "pkill", "-INT", "-P", str(process.pid)], check=False)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        _run(["sudo", "pkill", "-TERM", "-P", str(process.pid)], check=False)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _run(["sudo", "pkill", "-KILL", "-P", str(process.pid)], check=False)
            process.wait(timeout=5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-running", action="store_true")
    parser.add_argument(
        "--hold-after-completion",
        action="store_true",
        help="leave the visualization available until Ctrl-C, then clean up",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--policy", choices=("fable", "b0"), default="fable",
        help="Provider policy: dynamic FABLE frontiers or always-on whole-event B0.",
    )
    parser.add_argument(
        "--event-only", action="store_true",
        help="Pin inference to the sensor and disable dashboard frame/metadata traffic.",
    )
    parser.add_argument(
        "--event-evidence", action="store_true",
        help="Add deduplicated JPEG evidence to visual predicate events (implies --event-only).",
    )
    parser.add_argument("--control-socket", type=Path, default=CONTROL_SOCKET)
    parser.add_argument("--compose-override", type=Path, default=Path("deployment/compose.netwaggle.yaml"))
    parser.add_argument(
        "--direct", action="store_true",
        help="use ordinary Docker networking and do not start NetWaggle",
    )
    args = parser.parse_args(argv)
    if args.event_evidence:
        args.event_only = True
    compose = (*COMPOSE_BASE, "-f", str(args.compose_override))

    experiment = load_network_experiment(args.experiment)
    topology = NetWaggleTopology.from_dict(load_json(experiment.topology))
    changes = schedule(experiment.conditions, topology)
    if args.direct and changes:
        raise ValueError("direct network mode cannot apply NetWaggle conditions")
    resolved = {
        "topology": None if args.direct else str(experiment.topology),
        "event": None if experiment.event is None else str(experiment.event),
        "replay_manifest": None if experiment.replay_manifest is None else str(experiment.replay_manifest),
        "timeout_seconds": experiment.timeout_seconds,
        "expected_initial_node": experiment.expected_initial_node,
        "expected_disconnected_node": experiment.expected_disconnected_node,
        "require_completion": experiment.require_completion,
        "policy": args.policy,
        "event_only": args.event_only,
        "event_evidence": args.event_evidence,
        "network_mode": "direct" if args.direct else "netwaggle",
        "conditions": [
            {"due_seconds": item.due_seconds, "affected_link_name": item.affected_link_name, "phase": item.phase, **item.values}
            for item in changes
        ],
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return 0

    if experiment.event is not None:
        experiment.event.relative_to(ROOT)
        os.environ["FABLE_CE_FILE"] = str(experiment.event.relative_to(ROOT))
    manifest: dict[str, Any] = {}
    if experiment.replay_manifest is not None:
        os.environ["FABLE_REPLAY_MANIFEST"] = str(experiment.replay_manifest)
        manifest = load_json(experiment.replay_manifest)
        recording_root = Path(
            os.environ.get("RECORDING_ROOT")
            or _dotenv_value(ROOT / ".env", "RECORDING_ROOT")
            or ROOT / "replay"
        ).expanduser()
        gq_recording_root = Path(
            os.environ.get("GQ_RECORDING_ROOT")
            or _dotenv_value(ROOT / ".env", "GQ_RECORDING_ROOT")
            or recording_root
        ).expanduser()
        missing = []
        for row in manifest.get("sources", ()):
            media_rows = row.get("segments") or ({"path": row["path"]},)
            for media in media_rows:
                configured = Path(str(media["path"]))
                if not configured.is_absolute():
                    missing.append((experiment.replay_manifest.parent / configured).resolve())
                    continue
                try:
                    host_path = recording_root / configured.relative_to("/recordings")
                except ValueError:
                    try:
                        host_path = gq_recording_root / configured.relative_to("/gq-recordings")
                    except ValueError:
                        try:
                            host_path = ROOT / ".fable/runs" / "replay-assets" / configured.relative_to("/replay-assets")
                        except ValueError:
                            host_path = recording_root / configured.name
                missing.append(host_path)
        missing = [path for path in missing if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing replay recording(s): " + ", ".join(map(str, missing)))

    output = (
        args.output
        or ROOT / ".fable/runs" / "network"
        / f"{args.experiment.stem}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    ).resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except PermissionError as exc:
        raise PermissionError(
            f"experiment output is not writable: {output.parent}. "
            "Do not run this script with sudo; repair the directory ownership "
            "or pass --output to a writable location such as /tmp/fable-run."
        ) from exc
    (output / "resolved-experiment.json").write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    deployment_path = ROOT / "deployment" / "config" / "deployment.netwaggle.real.yaml"
    inference_sources = tuple(manifest.get("source_policy", {}).get("inference", ()))
    if inference_sources:
        deployment = load_json(deployment_path)
        if args.event_only and "edge1" in deployment.get("nodes", {}):
            deployment["nodes"]["edge1"]["available"] = False
        profile_path = Path(str(deployment.get("provider_profiles", "")))
        if profile_path and not profile_path.is_absolute():
            resolved_profile = (deployment_path.parent / profile_path).resolve()
            deployment["provider_profiles"] = str(
                Path("/workspace/FABLE") / resolved_profile.relative_to(ROOT)
            )
        manifest_sources = {str(row["source_id"]): row for row in manifest.get("sources", ())}
        configured_sources = deployment.setdefault("sources", {})
        template = next(iter(configured_sources.values()))
        deployment["sources"] = {
            source_id: {
                "node_id": "sensor1",
                "data_type": "audio_window" if manifest_sources[source_id].get("modality") == "audio" else "video_frame",
                "site_id": template.get("site_id", "convergence_site"),
                "sample_bytes": 32000 if manifest_sources[source_id].get("modality") == "audio" else 2000000,
                "available": True,
            }
            for source_id in inference_sources
        }
        deployment_path = output / "resolved-deployment.json"
        deployment_path.write_text(json.dumps(deployment, indent=2, sort_keys=True) + "\n")
    timeline = output / "timeline.jsonl"
    netwaggle: subprocess.Popen[str] | None = None
    link_publisher: _LinkStatePublisher | None = None
    dashboard_recorders: list[_DashboardRecorder] = []
    progress_capture: _ProgressCapture | None = None
    started_sessions: list[str] = []
    stopping = False
    validation_passed = False
    network_start: dict[str, int] | None = None

    def stop(_signal=None, _frame=None):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        # NetWaggle needs privileged Mininet setup. Direct mode deliberately
        # bypasses both sudo and the topology runner.
        if not args.direct:
            _run(["sudo", "-v"])
            _run(["sudo", "rm", "-f", str(args.control_socket)])
        # Keep the experiment broker isolated from an older testbed broker that
        # commonly occupies the default host port. Containers still use 1883.
        os.environ["FABLE_MQTT_HOST_PORT"] = "1884"
        required_containers = (CONTROLLER_CONTAINER, AGENT_CONTAINER, EDGE_CONTAINER, VISUALIZATION_CONTAINER)
        running = _run(
            ["docker", "inspect", "-f", "{{.State.Running}}", *required_containers],
            check=False, capture=True,
        )
        if running.returncode or running.stdout.splitlines() != ["true"] * len(required_containers):
            raise RuntimeError("FABLE stack is not running; start it in another terminal with: ./run stack")

        try:
            output.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError("persistent-stack outputs must be inside the FABLE directory") from exc
        for container in (CONTROLLER_CONTAINER, AGENT_CONTAINER):
            _run(["docker", "exec", container, "rm", "-f", CONTROLLER_BARRIER, REPLAY_BARRIER])
        controller_log = output / "controller.log"
        sensor_log = output / "sensor1-agent.log"
        edge_log = output / "edge1-agent.log"
        session_specs = {
            CONTROLLER_CONTAINER: {
                "argv": ["python", "scripts/run_fable.py", str(experiment.event.relative_to(ROOT)),
                         str(deployment_path.relative_to(ROOT)), "--visualization-state-file",
                         ".fable/runs/visualization-state.json", "--visualization-session-name",
                         args.experiment.stem, "--command-timeout-seconds", "120",
                         "--frontier-mode", "whole_event" if args.policy == "b0" else "fable"],
                "env": {
                    "FABLE_CONTROLLER_START_BARRIER_FILE": CONTROLLER_BARRIER,
                    "FABLE_REPLAY_START_BARRIER_FILE": REPLAY_BARRIER,
                },
                "log": "/workspace/FABLE/" + str(controller_log.relative_to(ROOT)),
            },
            AGENT_CONTAINER: {
                "argv": ["python", "scripts/run_node_agent.py",
                         ("deployment/config/node_agent.netwaggle.evidence.sensor.yaml"
                          if args.event_evidence else
                          "deployment/config/node_agent.netwaggle.evaluation.sensor.yaml"
                          if args.event_only else "deployment/config/node_agent.netwaggle.real.sensor.yaml"),
                         "--replay-manifest", str(experiment.replay_manifest.relative_to(ROOT))],
                "env": {
                    "FABLE_REPLAY_START_BARRIER_FILE": REPLAY_BARRIER,
                    "FABLE_EVIDENCE_METRICS_FILE": (
                        "/workspace/FABLE/" + str((output / "evidence-metrics.json").relative_to(ROOT))
                    ),
                    "FABLE_EVIDENCE_ALL_TRACKS": (
                        "1" if args.event_evidence and args.policy == "b0" else "0"
                    ),
                },
                "log": "/workspace/FABLE/" + str(sensor_log.relative_to(ROOT)),
            },
            EDGE_CONTAINER: {
                "argv": ["python", "scripts/run_node_agent.py", "deployment/config/node_agent.netwaggle.real.edge.yaml"],
                "log": "/workspace/FABLE/" + str(edge_log.relative_to(ROOT)),
            },
        }
        for index, (container, spec) in enumerate(session_specs.items()):
            request = output / f"session-{index}.json"
            request.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
            _session(container, "start", request)
            started_sessions.append(container)
        if not args.direct:
            netwaggle = subprocess.Popen(
                # Mininet is installed into the host system interpreter, not
                # the FABLE virtual environment used by this process.
                ["sudo", NETWAGGLE_PYTHON, "-m", "netwaggle.runner", "--topology", str(experiment.topology),
                 "--hold", "--no-cli", "--control-socket", str(args.control_socket)],
                cwd=ROOT,
                text=True,
            )
            _wait_for(lambda: args.control_socket.is_socket() or netwaggle.poll() is not None, "NetWaggle control socket")
            if netwaggle.poll() is not None:
                raise RuntimeError(f"NetWaggle exited with status {netwaggle.returncode}")
            route = _run(["docker", "exec", AGENT_CONTAINER, "ip", "route", "get", topology.gateway.ip_without_prefix], capture=True).stdout
            if "dev netwaggle0" not in route:
                raise RuntimeError(f"agent MQTT route bypasses NetWaggle: {route.strip()}")
        agent_client_ids = tuple(f"fable-agent-{attachment.name}" for attachment in topology.attachments)
        _wait_for(
            lambda: all(client_id in _run(
                [*compose, "logs", "--no-color", "mqtt"], check=False, capture=True
            ).stdout for client_id in agent_client_ids),
            "node-agent MQTT subscriptions",
            timeout=60,
        )
        if not args.event_only:
            _wait_for(
                lambda: edge_log.is_file() and "Provider pre-warm complete" in edge_log.read_text(encoding="utf-8"),
                "edge provider pre-warm",
                timeout=180,
            )
            _wait_for(
                lambda: sensor_log.is_file() and "Provider pre-warm complete" in sensor_log.read_text(encoding="utf-8"),
                "sensor provider pre-warm",
                timeout=180,
            )
        _run(["docker", "exec", CONTROLLER_CONTAINER, "touch", CONTROLLER_BARRIER])
        _wait_for(
            lambda: controller_log.is_file() and '"reason": "initial"' in controller_log.read_text(encoding="utf-8"),
            "controller initial execution plan",
            # First CUDA/YOLO initialization on the edge can legitimately
            # approach the controller's 120-second command timeout.
            timeout=150,
        )
        if not args.direct and any(_runtime_links(change, topology) for change in changes):
            link_publisher = _LinkStatePublisher("127.0.0.1", 1884)
            link_publisher.start()
        started = time.monotonic()

        def apply_change(epoch: int, change: ScheduledChange) -> None:
            while not stopping and time.monotonic() - started < change.due_seconds:
                time.sleep(min(0.05, change.due_seconds - (time.monotonic() - started)))
            if stopping:
                return
            requested = time.monotonic() - started
            runtime_links = _runtime_links(change, topology)
            orchestrator = None
            disconnecting = change.values.get("connection_status") == "disconnected"
            if runtime_links and disconnecting:
                orchestrator = link_publisher.publish(runtime_links)
            response = _request(args.control_socket, change, epoch)
            if runtime_links and not disconnecting:
                orchestrator = link_publisher.publish(runtime_links)
            applied = time.monotonic() - started
            row = {
                "affected_link_name": change.affected_link_name,
                "phase": change.phase,
                "requested_offset_seconds": change.due_seconds,
                "request_started_offset_seconds": round(requested, 6),
                "applied_offset_seconds": round(applied, 6),
                "values": change.values,
                "response": response,
                "orchestrator": orchestrator,
            }
            with timeline.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)

        # A zero-second condition defines the state at replay release. Applying
        # it first prevents early frames from racing the controller replan.
        indexed_changes = tuple(enumerate(changes, 1))
        for epoch, change in indexed_changes:
            if change.due_seconds == 0:
                apply_change(epoch, change)
        # The agent and controller have separate /tmp namespaces. Mark the
        # dashboard started first, then release frames from the agent.
        _run(["docker", "exec", CONTROLLER_CONTAINER, "touch", REPLAY_BARRIER])
        _set_visualization_session_status(
            ROOT / ".fable/runs" / "visualization-state.json", "replay_started"
        )
        dashboard_recorders = [] if args.event_only else [
            _DashboardRecorder(
                output, "devices-dashboard", "/api/devices-snapshot.jpg",
                # Begin with the first live camera. Sources with later event
                # origins join the grid when their replay time arrives.
                expected_devices=0,
            ),
            _DashboardRecorder(
                output, "ce-dashboard", "", fps=1.0, browser_path="/"
            ),
        ]
        for recorder in dashboard_recorders:
            recorder.start()
        if not args.event_only:
            progress_capture = _ProgressCapture(output)
            progress_capture.start()
        if args.event_only and not args.direct:
            network_start = _interface_counters(AGENT_CONTAINER)
        _run(["docker", "exec", AGENT_CONTAINER, "touch", REPLAY_BARRIER])
        _wait_for(
            lambda: sensor_log.is_file() and "ready with synchronized source barrier" in sensor_log.read_text(encoding="utf-8"),
            "synchronized replay start",
        )
        started = time.monotonic()
        replay_started_at = started
        for epoch, change in indexed_changes:
            if change.due_seconds > 0:
                apply_change(epoch, change)
        event_name = experiment.event.stem if experiment.event is not None else ""
        if experiment.require_completion:
            _wait_for(
                lambda: controller_log.is_file() and _completion_seen(
                    controller_log.read_text(encoding="utf-8"), event_name),
                f"{event_name} completion", timeout=experiment.timeout_seconds,
            )
        logs = controller_log.read_text(encoding="utf-8")
        initial_nodes = _plan_nodes(logs, "initial")
        expected_initial_node = "sensor1" if args.event_only else experiment.expected_initial_node
        if expected_initial_node:
            if initial_nodes != {expected_initial_node}:
                raise RuntimeError(f"initial plan nodes {sorted(initial_nodes)} != expected {[expected_initial_node]}")
        disconnected_nodes: set[str] = set()
        if experiment.expected_disconnected_node:
            disconnected_nodes = _plan_nodes(logs, "network_condition")
            if disconnected_nodes != {experiment.expected_disconnected_node}:
                raise RuntimeError(f"disconnected plan nodes {sorted(disconnected_nodes)} != expected {[experiment.expected_disconnected_node]}")
        completion_seen = _completion_seen(logs, event_name)
        validation = {
            "event": event_name,
            "completion_seen": completion_seen,
            "initial_plan_nodes": sorted(initial_nodes),
            "disconnected_plan_nodes": sorted(disconnected_nodes),
            "passed": not experiment.require_completion or completion_seen,
        }
        (output / "validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validation_passed = True
        replay_duration = _replay_duration_seconds(manifest)
        while not stopping and time.monotonic() - replay_started_at < replay_duration:
            time.sleep(min(0.25, replay_duration - (time.monotonic() - replay_started_at)))
        # Optional-completion experiments run for a fixed replay window, so
        # completion must be sampled again at the end rather than only just
        # after replay release.
        final_logs = controller_log.read_text(encoding="utf-8")
        validation["completion_seen"] = _completion_seen(final_logs, event_name)
        validation["passed"] = (
            not experiment.require_completion or bool(validation["completion_seen"])
        )
        (output / "validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if network_start is not None:
            network_end = _interface_counters(AGENT_CONTAINER)
            quantitative = _quantitative_summary(controller_log, network_start, network_end)
            evidence_metrics = output / "evidence-metrics.json"
            if evidence_metrics.is_file():
                evidence = json.loads(evidence_metrics.read_text(encoding="utf-8"))
                evidence["attempted_payload_mbps"] = round(
                    float(evidence.get("attempted_payload_bytes", 0)) * 8
                    / max(replay_duration, 0.001) / 1_000_000,
                    6,
                )
                quantitative["event_evidence"] = evidence
            (output / "quantitative-summary.json").write_text(
                json.dumps(quantitative, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(json.dumps({"quantitative_summary": quantitative}, sort_keys=True), flush=True)
        _set_visualization_session_status(
            ROOT / ".fable/runs" / "visualization-state.json",
            "replay_completed" if validation_passed else "replay_failed",
        )
        # Ensure the terminal lifecycle state is present in both recordings.
        time.sleep(0.5)
        if progress_capture is not None:
            progress_capture.stop()
            progress_capture = None
        device_recording_offset = (
            dashboard_recorders[0].recording_offset_seconds if dashboard_recorders else 0.0
        )
        recordings = [recorder.stop() for recorder in dashboard_recorders]
        if recordings and recordings[0] is not None and experiment.replay_manifest is not None:
            if _mux_replay_audio(
                recordings[0], manifest, experiment.replay_manifest,
                device_recording_offset,
            ):
                print(f"Added synchronized replay audio: {recordings[0]}", flush=True)
            else:
                print("Warning: replay audio could not be added to device dashboard", flush=True)
        if len(recordings) == 2 and recordings[0] is not None and recordings[1] is not None:
            combined = output / "combined-dashboard.mp4"
            if _stitch_dashboard_videos(recordings[0], recordings[1], combined):
                print(f"Saved combined dashboard video: {combined}", flush=True)
            else:
                print("Warning: dashboard videos could not be combined", flush=True)
        dashboard_recorders = []
        for label, recording in zip(("device", "CE progress"), recordings):
            if recording is not None:
                print(f"Saved {label} dashboard video: {recording}", flush=True)
            else:
                print(f"Warning: no {label} dashboard video was available", flush=True)
        print(json.dumps({"validation": validation}, sort_keys=True), flush=True)
        if args.hold_after_completion:
            print(
                "Experiment complete; visualization remains at "
                "http://127.0.0.1:5000 (press Ctrl-C to stop and clean up).",
                flush=True,
            )
            while not stopping:
                time.sleep(0.25)
        return 0
    finally:
        if progress_capture is not None:
            progress_capture.stop()
        for recorder in dashboard_recorders:
            recorder.stop()
        if link_publisher is not None:
            link_publisher.close()
        if netwaggle is not None:
            _stop_netwaggle(netwaggle)
        if not args.keep_running:
            for container in reversed(started_sessions):
                _session(container, "stop", check=False)
        state_path = ROOT / ".fable/runs" / "visualization-state.json"
        if state_path.is_file():
            _set_visualization_session_status(
                state_path, "replay_completed" if validation_passed else "replay_failed"
            )


if __name__ == "__main__":
    raise SystemExit(main())
