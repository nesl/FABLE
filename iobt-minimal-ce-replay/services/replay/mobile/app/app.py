"""Synchronized MP4 replay adapter exposing the standard local ZED frame ABI."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
import os
import subprocess
import sys
import threading
import time

import cv2
import msgpack
import numpy as np
import pynng

sys.path.append("/lib/iobtmax")
from iobt_max_service import iobt_max_service, state
from config_protocol import evaluate_config


class MobileReplay(iobt_max_service):
    def __init__(self, args):
        self.name = "zed"
        super().__init__(self.name)
        self.args = args
        self.segment_manifest = (
            json.loads(open(args.segment_manifest, encoding="utf-8").read())
            if args.segment_manifest
            else None
        )
        self.sync = threading.Event()
        self.sync_payload = {}
        self.node_id = os.environ.get("MCP_NODE_NAME", "")
        self.active_replay_id = None
        self.last_config_signature = None
        self.event_start_at = None
        self.timeline_start_epoch = (
            float(self.segment_manifest["timeline_start_epoch"])
            if self.segment_manifest
            else float(args.timeline_start_epoch)
        )
        self._retrospective_requests = deque()
        self._retrospective_condition = threading.Condition()
        self.subscribe("net", "/replay/sync", self._on_sync)
        self.subscribe("net", "/replay/config", self._on_config)
        self.subscribe(
            "net",
            f"/fable/v1/retrospective/{self.node_id}/raw-video/request",
            self._on_retrospective_request,
        )
        self._retrospective_thread = threading.Thread(
            target=self._retrospective_worker,
            name="mobile-raw-retrospective",
            daemon=True,
        )
        self._retrospective_thread.start()
        self.audio_sample_rate = 16_000
        self.audio_frame_samples = self.audio_sample_rate // 10
        self.audio_pub = pynng.Pub0()
        self.audio_pub.listen("ipc:///tmp/respeaker.ipc")
        self.pieces = self._pieces()
        self.audio_by_video = {
            piece["video"]: self._decode_audio(piece) for piece in self.pieces
        }

    def _pieces(self):
        return (
            self.segment_manifest["segments"]
            if self.segment_manifest
            else [
                {
                    "video": self.args.video,
                    "recording_start_epoch": self.args.recording_start_epoch,
                    "trim_start_seconds": self.args.start,
                    "trim_end_seconds": self.args.end,
                }
            ]
        )

    def _decode_audio(self, piece):
        duration = max(
            0.0,
            float(piece["trim_end_seconds"])
            - float(piece["trim_start_seconds"]),
        )
        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            str(piece["trim_start_seconds"]),
            "-t",
            str(duration),
            "-i",
            piece["video"],
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.audio_sample_rate),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-",
        ]
        result = subprocess.run(command, check=True, capture_output=True)
        return np.frombuffer(result.stdout, dtype="<i2").copy()

    def _publish_audio(self, payload):
        waveform = np.asarray(payload.pop("waveform"), dtype=np.int16)
        wire_payload = {
            **payload,
            "numpy_waveform": {
                "shape": list(waveform.shape),
                "data": waveform.tobytes(),
                "dtype": waveform.dtype.name,
            },
        }
        self.audio_pub.send(
            msgpack.packb({"topic": "rawaudio", "payload": wire_payload})
        )

    def _wait_until(self, target):
        while time.time() < target:
            if self.sync.is_set() or self.state == state.quit:
                return False
            time.sleep(min(0.01, max(0.0, target - time.time())))
        return True

    def _play_audio_piece(
        self,
        piece,
        *,
        replay_id,
        mode,
        speed,
        wall_start,
        event_start,
        timeline_start,
    ):
        samples = self.audio_by_video[piece["video"]]
        trim_start = float(piece["trim_start_seconds"])
        recording_start = float(piece["recording_start_epoch"])
        piece_offset = recording_start + trim_start - timeline_start
        for offset in range(0, len(samples), self.audio_frame_samples):
            if self.sync.is_set() or self.state == state.quit:
                break
            audio_elapsed = offset / self.audio_sample_rate
            relative_time = piece_offset + audio_elapsed
            event_time = event_start + relative_time
            wall_target = wall_start + relative_time / max(speed, 1e-6)
            if mode in {"realtime", "scaled"} and not self._wait_until(wall_target):
                break
            chunk = samples[offset : offset + self.audio_frame_samples, None]
            self._publish_audio(
                {
                    "t": event_time,
                    "recording_t": recording_start + trim_start + audio_elapsed,
                    "replay_id": replay_id,
                    "waveform": chunk,
                }
            )

    def _on_sync(self, _topic, payload):
        try:
            message = json.loads(payload)
        except Exception:
            return
        if str(message.get("action") or "START").upper() == "STOP":
            targets = set(message.get("target_nodes") or ())
            aliases = {os.environ.get("MCP_NODE_NAME", "")}
            if targets and aliases.isdisjoint(targets):
                return
            self.sync_payload = message
            self.sync.set()
            return
        scenario = str(message.get("scenario") or "")
        if scenario and scenario != self.args.scenario:
            return
        self.sync_payload = message
        replay_id = message.get("replay_id")
        self.active_replay_id = str(replay_id) if replay_id is not None else None
        self.event_start_at = float(message.get("event_start_at") or time.time())
        self.sync.set()

    def _on_config(self, _topic, payload):
        """Acknowledge configuration freshness without starting playback.

        Mobile archives are mounted when the container starts, unlike the
        scenario-switching ZED supervisor.  They still participate in the
        same configuration barrier: PROBE and START refresh readiness, while
        only a subsequent /replay/sync START may set ``self.sync``.
        """

        try:
            message = json.loads(payload)
        except Exception:
            return
        if not isinstance(message, dict):
            return
        decision = evaluate_config(
            message,
            node_id=self.node_id,
            loaded_scenario=self.args.scenario,
            prior_signature=self.last_config_signature,
        )
        if not decision.accepted:
            return
        self.last_config_signature = decision.signature
        replay_id = message.get("replay_id")
        reason = {
            "PROBE": "probe_ready",
            "START": (
                "duplicate_configuration_ready"
                if decision.reason == "duplicate_config"
                else "configuration_ready"
            ),
            "STOP": "configuration_stopped",
        }[decision.action]
        self.publish_readiness(
            "mobile",
            ready=True,
            reason=reason,
            scenario=self.args.scenario,
            replay_id=replay_id,
            config_action=decision.action,
        )

    @staticmethod
    def _parse_event_epoch(value):
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()

    def _publish_retrospective_status(self, request, status, *, reason="", frames=0):
        self.publish(
            "net",
            f"/fable/v1/retrospective/{self.node_id}/raw-video/status",
            json.dumps(
                {
                    "schema_version": "fable.raw_retrospective_replay_status.v1",
                    "request_id": request.get("request_id"),
                    "demand_id": request.get("demand_id"),
                    "replay_id": request.get("replay_id"),
                    "node_id": self.node_id,
                    "status": status,
                    "reason": reason,
                    "frames_published": int(frames),
                }
            ),
        )

    def _on_retrospective_request(self, _topic, payload):
        request = {}
        try:
            request = json.loads(payload)
            if request.get("schema_version") != "fable.raw_retrospective_replay_request.v1":
                raise ValueError("unsupported retrospective request schema")
            if request.get("replay_id") != self.active_replay_id:
                raise ValueError("retrospective request replay_id is not active")
            interval = request.get("event_time_interval") or {}
            if not interval.get("start") or not interval.get("end"):
                raise ValueError("retrospective request has no event-time interval")
            with self._retrospective_condition:
                self._retrospective_requests.append(request)
                self._retrospective_condition.notify()
        except Exception as exc:
            self._publish_retrospective_status(
                request, "REJECTED", reason=str(exc), frames=0
            )

    def _retrospective_worker(self):
        while self.state != state.quit:
            with self._retrospective_condition:
                while not self._retrospective_requests and self.state != state.quit:
                    self._retrospective_condition.wait(timeout=0.5)
                if self.state == state.quit:
                    return
                request = self._retrospective_requests.popleft()
            try:
                self._execute_retrospective_request(request)
            except Exception as exc:
                print(f"[MOBILE] Retrospective replay failed: {exc}", flush=True)
                self._publish_retrospective_status(
                    request, "FAILED", reason=str(exc), frames=0
                )

    def _execute_retrospective_request(self, request):
        if self.event_start_at is None:
            raise ValueError("replay has not been synchronized")
        interval = request["event_time_interval"]
        event_start = self._parse_event_epoch(interval["start"])
        event_end = self._parse_event_epoch(interval["end"])
        if event_end < event_start:
            raise ValueError("retrospective interval ends before it starts")
        source_start = self.timeline_start_epoch + event_start - self.event_start_at
        source_end = self.timeline_start_epoch + event_end - self.event_start_at
        requested_fps = max(0.1, float(request.get("requested_fps", 5.0)))
        maximum_frames = max(1, min(5000, int(request.get("maximum_frames", 900))))
        frames = 0
        self._publish_retrospective_status(request, "STARTED", frames=0)
        for piece in self.pieces:
            if frames >= maximum_frames or self.state == state.quit:
                break
            recording_start = float(piece["recording_start_epoch"])
            trim_start = float(piece["trim_start_seconds"])
            trim_end = float(piece["trim_end_seconds"])
            overlap_start = max(trim_start, source_start - recording_start)
            overlap_end = min(trim_end, source_end - recording_start)
            if overlap_end < overlap_start:
                continue
            capture = cv2.VideoCapture(piece["video"])
            native_fps = max(1.0, float(capture.get(cv2.CAP_PROP_FPS) or 30.0))
            stride = max(1, int(round(native_fps / requested_fps)))
            capture.set(cv2.CAP_PROP_POS_MSEC, overlap_start * 1000.0)
            source_index = 0
            try:
                while frames < maximum_frames and self.state != state.quit:
                    position = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if position > overlap_end:
                        break
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if source_index % stride:
                        source_index += 1
                        continue
                    source_index += 1
                    ok, encoded = cv2.imencode(".jpg", frame)
                    if not ok:
                        continue
                    source_epoch = recording_start + position
                    event_time = self.event_start_at + source_epoch - self.timeline_start_epoch
                    self.publish(
                        "local",
                        "data",
                        {
                            "t": event_time,
                            "i": encoded.tobytes(),
                            "d": None,
                            "replay_id": request.get("replay_id"),
                        },
                    )
                    frames += 1
                    time.sleep(1.0 / requested_fps)
            finally:
                capture.release()
        if frames == 0:
            raise ValueError("retrospective interval is outside loaded mobile recordings")
        self._publish_retrospective_status(request, "COMPLETED", frames=frames)

    def service_initialize(self):
        self.publish_readiness(
            "mobile",
            ready=True,
            reason="data_loaded_waiting_for_sync",
            scenario=self.args.scenario,
            video_file=self.args.video,
            local_ipc="/tmp/zed.ipc",
            audio_local_ipc="/tmp/respeaker.ipc",
            audio_sample_rate_hz=self.audio_sample_rate,
            audio_channels=1,
        )

    def service_step(self):
        while self.state != state.quit:
            if not self.sync.wait(0.2):
                continue
            self.sync.clear()
            if str(self.sync_payload.get("action") or "START").upper() == "STOP":
                self.publish_readiness(
                    "mobile",
                    ready=True,
                    reason="explicit_replay_stop",
                    scenario=self.args.scenario,
                    replay_id=self.sync_payload.get("replay_id"),
                )
                continue
            self._play()

    def _play(self):
        replay_id = self.sync_payload.get("replay_id")
        mode = str(self.sync_payload.get("playback_mode") or "max")
        speed = float(self.sync_payload.get("speed") or 1.0)
        wall_start = float(self.sync_payload.get("start_at") or time.time())
        event_start = float(self.sync_payload.get("event_start_at") or wall_start)
        while time.time() < wall_start and not self.sync.is_set():
            time.sleep(0.01)
        pieces = self.pieces
        timeline_start = self.timeline_start_epoch
        frame_index = 0
        audio_frames = 0
        for piece in pieces:
            if self.sync.is_set() or self.state == state.quit:
                break
            audio_thread = threading.Thread(
                target=self._play_audio_piece,
                kwargs={
                    "piece": piece,
                    "replay_id": replay_id,
                    "mode": mode,
                    "speed": speed,
                    "wall_start": wall_start,
                    "event_start": event_start,
                    "timeline_start": timeline_start,
                },
            )
            audio_thread.start()
            capture = cv2.VideoCapture(piece["video"])
            trim_start = float(piece["trim_start_seconds"])
            trim_end = float(piece["trim_end_seconds"])
            capture.set(cv2.CAP_PROP_POS_MSEC, trim_start * 1000.0)
            last_position = trim_start
            piece_offset = (
                float(piece["recording_start_epoch"])
                + trim_start
                - timeline_start
            )
            while not self.sync.is_set() and self.state != state.quit:
                position = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if position < last_position:
                    position = last_position
                if position >= trim_end:
                    break
                ok, frame = capture.read()
                if not ok:
                    break
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    continue
                relative_time = piece_offset + max(0.0, position - trim_start)
                # Processing speed changes wall scheduling, never source event
                # time.  Evidence therefore stays aligned across modalities.
                event_time = event_start + relative_time
                self.publish(
                    "local",
                    "data",
                    {
                        "t": event_time,
                        "i": encoded.tobytes(),
                        "d": None,
                        "replay_id": replay_id,
                    },
                )
                frame_index += 1
                last_position = position
                if mode in {"realtime", "scaled"}:
                    target = wall_start + relative_time / max(speed, 1e-6)
                    time.sleep(max(0.0, target - time.time()))
            capture.release()
            audio_thread.join()
            audio_frames += (
                len(self.audio_by_video[piece["video"]])
                + self.audio_frame_samples
                - 1
            ) // self.audio_frame_samples
        self.publish_readiness(
            "mobile",
            ready=True,
            reason="replay_complete",
            scenario=self.args.scenario,
            replay_id=replay_id,
            frames=frame_index,
            audio_frames=audio_frames,
        )
        # Match the ZED replay ABI.  Readiness answers whether a service can
        # participate in a replay; this retained status event marks the hard
        # end of ordinary evidence generation for the specific replay ID.
        # Node agents use it to prevent queued live results from leaking across
        # an outage restoration while retaining raw media for explicit
        # recovery requests.
        complete = {
            "service": "mobile_replay",
            "node": self.node_id or self.hostname,
            "event": "complete",
            "event_time": event_start + max(
                (
                    float(piece["recording_start_epoch"])
                    + float(piece["trim_end_seconds"])
                    - timeline_start
                    for piece in pieces
                ),
                default=0.0,
            ),
            "replay_id": replay_id,
            "frames": frame_index,
            "audio_frames": audio_frames,
            "t": time.time(),
        }
        info = self.mqtt_client.publish(
            f"/replay/status/mobile/{self.node_id or self.hostname}",
            json.dumps(complete),
            qos=1,
            retain=True,
        )
        info.wait_for_publish(timeout=2.0)

    def service_initialize_collect(self):
        pass

    def service_stop_collect(self):
        pass

    def service_stop(self):
        with self._retrospective_condition:
            self._retrospective_condition.notify_all()
        self.audio_pub.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video")
    parser.add_argument("--segment-manifest")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--recording-start-epoch", type=float)
    parser.add_argument("--timeline-start-epoch", type=float)
    args = parser.parse_args()
    if bool(args.video) == bool(args.segment_manifest):
        parser.error("provide exactly one of --video or --segment-manifest")
    if args.video and (args.start is None or args.end is None):
        parser.error("--video requires --start and --end")
    if args.video and (
        args.recording_start_epoch is None or args.timeline_start_epoch is None
    ):
        parser.error(
            "--video requires --recording-start-epoch and --timeline-start-epoch"
        )
    MobileReplay(args).start()


if __name__ == "__main__":
    main()
