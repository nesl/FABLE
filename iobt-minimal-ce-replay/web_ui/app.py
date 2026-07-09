#!/usr/bin/env python3
"""Small web UI for the minimal IoBT replay/CE testbed.

The UI is intentionally lightweight:
  * publishes /replay/config and /replay/sync to start synchronized replay
  * publishes optional global control messages
  * tails important MQTT topics through a Server-Sent Events stream

Run locally:
  MQTT_HOST=localhost MQTT_PORT=1883 uvicorn web_ui.app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import deque
from pathlib import Path
from threading import Lock, Timer
from typing import Any, Deque, Dict, Iterable, List, Optional

from lib.scenario_catalog import build_and_write_catalog, parse_roots, scan_scenarios

import paho.mqtt.client as mqtt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

MQTT_HOST = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_CLIENT_ID = os.environ.get("MQTT_CLIENT_ID", "iobt-minimal-web-ui")

DEFAULT_DATA_ROOTS = [
    "/media/brianw/Extreme SSD/West Point Experimentation",
    "/media/brianw/Extreme SSD/GQ Data",
]

DATA_ROOTS = [p for p in os.environ.get("IOBT_DATA_ROOTS", "").split(os.pathsep) if p] or DEFAULT_DATA_ROOTS
SCENARIO_CATALOG_DIR = Path(os.environ.get("SCENARIO_CATALOG_DIR", "/generated"))
_scenario_catalog_cache: Dict[str, Any] | None = None

def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}

CLEAR_REPLAY_RETAINED_ON_STARTUP = env_bool("CLEAR_REPLAY_RETAINED_ON_STARTUP", True)

SUBSCRIPTIONS = [
    "/replay/#",
    "/+/analytics/yolo/bbox",
    "/+/analytics/yolo/status",              # legacy status topic
    "/debug/+/analytics/yolo/status",        # explicit synthetic debug topic
    "/+/analytics/yolo/annotated/compressed",
    "/debug/+/analytics/yolo/annotated/compressed",
    "/debug/+/analytics/yolo/frame",           # explicit synthetic frame-input probe
    "/+/audio_detector/detections",
    "/+/audio_detector/status",             # legacy status topic
    "/debug/+/audio_detector/status",       # explicit synthetic debug topic
    "/complex_events/#",
    "/trackers/#",
    "/geospatialdetections/#",
]

app = FastAPI(title="IoBT Minimal Replay UI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

_messages: Deque[Dict[str, Any]] = deque(maxlen=50)
_lock = Lock()
_next_id = 1
_mqtt_client: mqtt.Client | None = None
_connected = False
_last_replay_config: Dict[str, Any] | None = None
_last_replay_sync: Dict[str, Any] | None = None
_startup_retained_clear_done = False


def _remember_replay_message(topic: str, payload: Any) -> None:
    global _last_replay_config, _last_replay_sync
    if topic == "/replay/config":
        _last_replay_config = payload if isinstance(payload, dict) else {"raw": payload}
    elif topic == "/replay/sync":
        _last_replay_sync = payload if isinstance(payload, dict) else {"raw": payload}


class ReplayStartRequest(BaseModel):
    scenario: str = Field(..., min_length=1)
    start: float = 0.0
    end: float = -1.0
    sync_delay: float = Field(1.0, ge=0.0, le=120.0)
    playback_mode: str = Field("max", description="max, realtime, or scaled")
    speed: Optional[float] = Field(None, description="Speed multiplier used only when playback_mode=scaled")
    send_control: bool = False


class ControlRequest(BaseModel):
    action: str = Field(..., min_length=1)


class PublishRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    payload: Any = ""
    qos: int = Field(0, ge=0, le=2)
    retain: bool = False


def _normalize_payload(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _record_message(topic: str, payload: Any, direction: str = "in") -> Dict[str, Any]:
    global _next_id
    with _lock:
        item = {
            "id": _next_id,
            "ts": time.time(),
            "direction": direction,
            "topic": topic,
            "payload": payload,
        }
        _next_id += 1
        _messages.append(item)
        _remember_replay_message(topic, payload)
        return item


def _recent_messages(after_id: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
    with _lock:
        items = [m for m in _messages if int(m["id"]) > after_id]
    return items[-limit:]


def _publish(topic: str, payload: Any, qos: int = 1, retain: bool = False) -> None:
    if _mqtt_client is None:
        raise HTTPException(status_code=503, detail="MQTT client is not initialized")
    if isinstance(payload, (dict, list)):
        wire_payload = json.dumps(payload)
    else:
        wire_payload = str(payload)
    result = _mqtt_client.publish(topic, wire_payload, qos=qos, retain=retain)
    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        raise HTTPException(status_code=502, detail=f"MQTT publish failed with rc={result.rc}")
    _record_message(topic, payload, direction="out")


def _clear_retained(topic: str) -> None:
    """Clear one retained MQTT topic on the broker."""
    if _mqtt_client is None:
        raise HTTPException(status_code=503, detail="MQTT client is not initialized")
    result = _mqtt_client.publish(topic, payload=None, qos=1, retain=True)
    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        raise HTTPException(status_code=502, detail=f"MQTT retain-clear failed with rc={result.rc}")
    _record_message(topic, {"cleared_retained": True}, direction="out")


def _clear_retained_with_client(client: mqtt.Client, topic: str, reason: str) -> None:
    result = client.publish(topic, payload=None, qos=1, retain=True)
    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        _record_message(topic, {"cleared_retained": True, "reason": reason}, direction="out")
    else:
        _record_message(topic, {"cleared_retained": False, "reason": reason, "rc": result.rc}, direction="status")




def _publish_sync_once(sync: Dict[str, Any], *, delay_sec: float = 0.0, burst_index: int = 0, note: str = "") -> None:
    """Publish one /replay/sync message, optionally after a small delay.

    This is intentionally MQTT-only. The goal is to avoid the common race where
    /replay/config starts/restarts a replay child, but the child subscribes to
    /replay/sync just after the first sync message was published. All messages
    in a burst carry the same replay_id and start_at so replay apps can ignore
    duplicates if they already started.
    """
    def _send() -> None:
        payload = dict(sync)
        payload["burst_index"] = int(burst_index)
        if note:
            payload["note"] = note
        try:
            _publish("/replay/sync", payload, qos=1, retain=False)
        except Exception as exc:  # avoid Timer thread crashing silently
            _record_message("$web_ui/replay_sync_burst_error", {
                "error": str(exc),
                "burst_index": burst_index,
                "sync": payload,
            }, direction="status")

    if delay_sec <= 0:
        _send()
    else:
        timer = Timer(delay_sec, _send)
        timer.daemon = True
        timer.start()


def _publish_sync_burst(sync: Dict[str, Any], *, initial_delay: float, reason: str) -> Dict[str, Any]:
    """Publish a short sync burst to handle late replay-child startup.

    The first message is sent immediately, then again near/after the requested
    start time. If a child catches the first sync, later duplicates should be
    ignored by replay_id/start_at. If it starts late, one of the later MQTT
    sync messages should reach it without requiring the user to click Resend.
    """
    base = max(float(initial_delay), 0.0)
    delays = [0.0]
    for d in (max(base * 0.5, 0.25), base + 0.25, base + 1.0, base + 2.0):
        if d not in delays:
            delays.append(d)
    for i, d in enumerate(delays):
        _publish_sync_once(sync, delay_sec=d, burst_index=i, note=reason)
    return {"count": len(delays), "delays_sec": delays}


def _on_connect(client: mqtt.Client, _userdata: Any, _flags: Any, reason_code: Any, _properties: Any = None) -> None:
    global _connected, _startup_retained_clear_done, _last_replay_config, _last_replay_sync
    _connected = True
    for topic in SUBSCRIPTIONS:
        client.subscribe(topic, qos=0)
    _record_message("$web_ui/status", {"connected": True, "host": MQTT_HOST, "port": MQTT_PORT}, direction="status")
    if CLEAR_REPLAY_RETAINED_ON_STARTUP and not _startup_retained_clear_done:
        _clear_retained_with_client(client, "/replay/config", "web_ui_startup")
        _clear_retained_with_client(client, "/replay/sync", "web_ui_startup")
        _last_replay_config = None
        _last_replay_sync = None
        _startup_retained_clear_done = True
        _record_message("$web_ui/status", {"cleared_replay_retained_on_startup": True}, direction="status")


def _on_disconnect(_client: mqtt.Client, _userdata: Any, *args: Any) -> None:
    global _connected
    _connected = False
    reason = args[-2] if len(args) >= 2 else (args[0] if args else "unknown")
    _record_message("$web_ui/status", {"connected": False, "reason_code": str(reason)}, direction="status")


def _on_message(_client: mqtt.Client, _userdata: Any, msg: mqtt.MQTTMessage) -> None:
    _record_message(msg.topic, _normalize_payload(msg.payload), direction="in")


def _connect_mqtt() -> mqtt.Client:
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=MQTT_CLIENT_ID)
    except AttributeError:
        # Compatibility with older paho-mqtt versions.
        client = mqtt.Client(client_id=MQTT_CLIENT_ID)
    client.on_connect = _on_connect
    client.on_disconnect = _on_disconnect
    client.on_message = _on_message
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_start()
    return client


@app.on_event("startup")
def startup() -> None:
    global _mqtt_client, _scenario_catalog_cache
    _mqtt_client = _connect_mqtt()
    try:
        roots = [Path(p) for p in DATA_ROOTS]
        _scenario_catalog_cache = build_and_write_catalog(roots, SCENARIO_CATALOG_DIR)
        _record_message("$web_ui/scenario_catalog", {
            "count": _scenario_catalog_cache["metadata"]["count"],
            "json": _scenario_catalog_cache["paths"]["json"],
            "csv": _scenario_catalog_cache["paths"]["csv"],
        }, direction="status")
    except Exception as exc:
        _scenario_catalog_cache = {"metadata": {"roots": DATA_ROOTS, "count": 0, "error": str(exc)}, "scenarios": [], "paths": {}}
        _record_message("$web_ui/scenario_catalog", {"error": str(exc)}, direction="status")


@app.on_event("shutdown")
def shutdown() -> None:
    global _mqtt_client
    if _mqtt_client is not None:
        _mqtt_client.loop_stop()
        _mqtt_client.disconnect()
        _mqtt_client = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/device")
def device_view() -> FileResponse:
    return FileResponse(STATIC / "device_view.html")


@app.get("/scenarios")
def scenarios_view() -> FileResponse:
    return FileResponse(STATIC / "scenarios.html")


@app.get("/api/state")
def state() -> Dict[str, Any]:
    return {
        "mqtt": {"host": MQTT_HOST, "port": MQTT_PORT, "connected": _connected},
        "subscriptions": SUBSCRIPTIONS,
        "data_roots": DATA_ROOTS,
        "recent": _recent_messages(limit=50),
        "replay": {"config": _last_replay_config, "sync": _last_replay_sync},
    }


@app.get("/api/data-roots")
def data_roots() -> Dict[str, Any]:
    return {
        "data_roots": DATA_ROOTS,
        "note": (
            "The web UI does not mount replay data. "
            "compose.replay.yaml mounts parent data roots, and persistent replay supervisors "
            "resolve the requested scenario/date against those mounted roots at replay time."
        ),
    }


@app.get("/api/messages")
def messages(after_id: int = 0, limit: int = 50) -> Dict[str, Any]:
    limit = min(max(int(limit), 1), 50)
    return {"messages": _recent_messages(after_id=after_id, limit=limit)}


@app.get("/api/scenarios")
def scenarios() -> Dict[str, Any]:
    global _scenario_catalog_cache
    if _scenario_catalog_cache is None:
        _scenario_catalog_cache = build_and_write_catalog([Path(p) for p in DATA_ROOTS], SCENARIO_CATALOG_DIR)
    return {
        "metadata": _scenario_catalog_cache.get("metadata", {}),
        "paths": _scenario_catalog_cache.get("paths", {}),
        "scenarios": _scenario_catalog_cache.get("scenarios", []),
    }


@app.post("/api/scenarios/refresh")
def refresh_scenarios() -> Dict[str, Any]:
    global _scenario_catalog_cache
    _scenario_catalog_cache = build_and_write_catalog([Path(p) for p in DATA_ROOTS], SCENARIO_CATALOG_DIR)
    _record_message("$web_ui/scenario_catalog", {
        "count": _scenario_catalog_cache["metadata"]["count"],
        "json": _scenario_catalog_cache["paths"].get("json"),
        "csv": _scenario_catalog_cache["paths"].get("csv"),
    }, direction="status")
    return {
        "ok": True,
        "metadata": _scenario_catalog_cache.get("metadata", {}),
        "paths": _scenario_catalog_cache.get("paths", {}),
        "scenarios": _scenario_catalog_cache.get("scenarios", []),
    }


@app.get("/api/scenarios/catalog.json")
def scenario_catalog_json() -> FileResponse:
    path = SCENARIO_CATALOG_DIR / "scenario_catalog.json"
    if not path.exists():
        build_and_write_catalog([Path(p) for p in DATA_ROOTS], SCENARIO_CATALOG_DIR)
    return FileResponse(path, media_type="application/json", filename="scenario_catalog.json")


@app.get("/api/scenarios/catalog.csv")
def scenario_catalog_csv() -> FileResponse:
    path = SCENARIO_CATALOG_DIR / "scenario_catalog.csv"
    if not path.exists():
        build_and_write_catalog([Path(p) for p in DATA_ROOTS], SCENARIO_CATALOG_DIR)
    return FileResponse(path, media_type="text/csv", filename="scenario_catalog.csv")


def _normalize_playback_timing(mode: str, speed: Optional[float]) -> tuple[str, float]:
    normalized = str(mode or "max").lower().strip()
    if normalized in {"fast", "asap", "unlimited"}:
        normalized = "max"
    if normalized not in {"max", "realtime", "scaled"}:
        raise HTTPException(status_code=400, detail="playback_mode must be max, realtime, or scaled")

    # Max-speed and realtime modes ignore the custom multiplier.
    # This lets the web UI omit speed entirely unless Custom speed is selected.
    if normalized == "max":
        return normalized, 1.0
    if normalized == "realtime":
        return normalized, 1.0

    scaled_speed = 1.0 if speed is None else float(speed)
    if scaled_speed <= 0:
        raise HTTPException(status_code=400, detail="speed must be > 0 for scaled mode")
    return normalized, scaled_speed


@app.post("/api/replay/start")
def start_replay(req: ReplayStartRequest) -> Dict[str, Any]:
    playback_mode, speed = _normalize_playback_timing(req.playback_mode, req.speed)
    replay_id = f"{req.scenario}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    config = {
        "scenario": req.scenario,
        "start_time": req.start,
        "end_time": req.end,
        "playback_mode": playback_mode,
        "speed": speed,
        "replay_id": replay_id,
    }
    start_at = time.time() + req.sync_delay
    sync = {
        "scenario": req.scenario,
        "start_at": start_at,
        "playback_mode": playback_mode,
        "speed": speed,
        "replay_id": replay_id,
    }

    # Keep /replay/config retained so supervisors that start later know the
    # selected scenario. /replay/sync is deliberately non-retained; supervisors
    # that already received it seed/rebroadcast it to their child replay process.
    _publish("/replay/config", config, qos=1, retain=True)
    if req.send_control:
        _publish("control", "collection-start", qos=1)

    burst = _publish_sync_burst(sync, initial_delay=req.sync_delay, reason="start_replay")

    return {
        "ok": True,
        "config": config,
        "sync": sync,
        "sync_burst": burst,
        "message": (
            f"Replay sync burst published in {playback_mode} mode "
            f"({burst['count']} MQTT sync messages over ~{max(burst['delays_sec']):.1f}s). "
            "The persistent replay supervisors resolve the scenario against the mounted SSD data roots."
        ),
    }


@app.post("/api/replay/resend-sync")
def resend_sync(delay: float = 0.5) -> Dict[str, Any]:
    if not _last_replay_config or not isinstance(_last_replay_config, dict):
        raise HTTPException(status_code=400, detail="No replay config has been published yet")
    scenario = str(_last_replay_config.get("scenario") or "").strip()
    if not scenario:
        raise HTTPException(status_code=400, detail="Last replay config has no scenario")
    playback_mode, speed = _normalize_playback_timing(
        str(_last_replay_config.get("playback_mode", "max")),
        float(_last_replay_config.get("speed", 1.0)),
    )
    delay = max(float(delay), 0.0)
    replay_id = str(_last_replay_config.get("replay_id") or f"{scenario}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}")
    # Keep config and sync on the same replay_id. Resend sync is meant to start
    # an already configured replay child, not create an unrelated replay session.
    _last_replay_config["replay_id"] = replay_id
    _last_replay_config["playback_mode"] = playback_mode
    _last_replay_config["speed"] = speed
    sync = {
        "scenario": scenario,
        "start_at": time.time() + delay,
        "playback_mode": playback_mode,
        "speed": speed,
        "replay_id": replay_id,
        "resent": True,
    }
    _publish("/replay/config", _last_replay_config, qos=1, retain=True)
    burst = _publish_sync_burst(sync, initial_delay=delay, reason="resend_sync")
    return {"ok": True, "sync": sync, "sync_burst": burst, "message": f"Resent /replay/sync burst for {scenario}"}


@app.post("/api/replay/clear")
def clear_replay_command() -> Dict[str, Any]:
    global _last_replay_config, _last_replay_sync
    _clear_retained("/replay/config")
    _clear_retained("/replay/sync")
    _last_replay_config = None
    _last_replay_sync = None
    return {"ok": True, "message": "Cleared retained replay config/sync from MQTT broker"}


@app.post("/api/control")
def send_control(req: ControlRequest) -> Dict[str, Any]:
    allowed = {"collection-start", "collection-stop", "collection-shutdown"}
    if req.action not in allowed:
        raise HTTPException(status_code=400, detail=f"action must be one of {sorted(allowed)}")
    _publish("control", req.action, qos=1)
    return {
        "ok": True,
        "topic": "control",
        "payload": req.action,
        "note": (
            "Replay config/sync were not cleared. Use Clear replay command for that. "
            "collection-shutdown may cause replay/detector services to quit until Docker Compose recreates them."
        ),
    }


@app.post("/api/publish")
def publish(req: PublishRequest) -> Dict[str, Any]:
    # Convenience endpoint for debugging. Keep this local-only in real deployments.
    _publish(req.topic, req.payload, qos=req.qos, retain=req.retain)
    return {"ok": True, "topic": req.topic, "payload": req.payload}


def _sse_event_stream() -> Iterable[str]:
    last_id = 0
    while True:
        new_items = _recent_messages(after_id=last_id, limit=50)
        if new_items:
            for item in new_items:
                last_id = max(last_id, int(item["id"]))
                yield f"id: {item['id']}\n"
                yield "event: mqtt\n"
                yield f"data: {json.dumps(item, separators=(',', ':'))}\n\n"
        else:
            yield f"event: heartbeat\ndata: {json.dumps({'ts': time.time(), 'connected': _connected})}\n\n"
        time.sleep(0.5)


@app.get("/api/events")
def events() -> StreamingResponse:
    return StreamingResponse(_sse_event_stream(), media_type="text/event-stream")
