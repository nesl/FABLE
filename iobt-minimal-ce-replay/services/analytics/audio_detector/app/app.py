#!/usr/bin/python

import os, sys
sys.path.append("/lib/iobtmax")

from iobt_max_service import iobt_max_service, state

import time
import numpy as np
import queue
import json
from threading import Lock


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


class audio_detector(iobt_max_service):

    def __init__(self):
        self.name = "audio_detector"
        iobt_max_service.__init__(self, self.name)

        self.q = queue.Queue()
        self.detection_threshold = float(os.environ["DETECTION_THRESHOLD"])

        self.net_topic = self.get_topic_name("detections")
        self.status_topic = os.environ.get("AUDIO_STATUS_TOPIC", f"/debug/{self.hostname}/audio_detector/status")
        self.publish_status = env_bool("AUDIO_PUBLISH_STATUS", False)
        self.status_fps = float(os.environ.get("AUDIO_STATUS_FPS", "1.0"))
        # By default, do not publish idle /audio_detector/status heartbeats before replay
        # actually delivers audio frames. Actual detections are still published immediately.
        self.publish_idle_status = env_bool("AUDIO_PUBLISH_IDLE_STATUS", False)
        self.print_idle_status = env_bool("AUDIO_PRINT_IDLE_STATUS", False)
        self.expected_local_ipc = "/tmp/respeaker.ipc"
        self.start_wall_ts = time.time()
        self.frames_total = 0
        self.detections_total = 0
        self.last_frame_wall_ts = None
        self.last_detection_wall_ts = None
        self.last_db = None
        self.last_error = None
        self.last_status_pub_time = 0.0
        print(f"Issuing audio detections on {self.net_topic}", flush=True)
        print(
            f"Issuing optional audio debug status on {self.status_topic} enabled={self.publish_status} "
            f"fps={self.status_fps} idle_status={self.publish_idle_status}",
            flush=True,
        )

        self.subscribe("local", "respeaker", self.get_respeaker_data)

        self.lock = Lock()
        self.data_file = None

    def get_respeaker_data(self, data):
        self.q.put(data)
        time.sleep(0)

    def service_initialize(self):
        print("Calling audio-detector service initialize", flush=True)

    def service_stop(self):
        print("Calling audio-detector service stop", flush=True)

    def service_initialize_collect(self):
        print("Calling audio-detector initialize collect", flush=True)
        self.data_file_name = self.get_file_name("csv")
        with self.lock:
            self.data_file = open(self.data_file_name, "w")
            self.data_file.write("Timestamp,Loudness\n")

    def service_stop_collect(self):
        print("Calling audio-detector stop collect", flush=True)
        with self.lock:
            if self.data_file is not None:
                self.data_file.close()
            self.data_file = None

    def diagnosis(self):
        now = time.time()
        if not os.path.exists(self.expected_local_ipc):
            return "waiting_for_respeaker_ipc_socket"
        if self.frames_total == 0:
            return "waiting_for_audio_frames"
        if self.last_frame_wall_ts and now - self.last_frame_wall_ts > 5.0:
            return "audio_frames_stale"
        if self.last_detection_wall_ts and now - self.last_detection_wall_ts < 5.0:
            return "active_audio_detection"
        return "audio_active_below_threshold"

    def publish_status_msg(self, *, frame_count, num_det, frame_rate, data_rate, avg_loudness, last_det):
        if not self.publish_status or self.status_fps <= 0:
            return
        if self.frames_total == 0 and not self.publish_idle_status:
            return
        now = time.time()
        if now - self.last_status_pub_time < 1.0 / self.status_fps:
            return
        self.last_status_pub_time = now
        msg = {
            "kind": "debug_status",
            "synthetic_debug": True,
            "t": now,
            "node": self.hostname,
            "expected_local_ipc": self.expected_local_ipc,
            "local_ipc_exists": os.path.exists(self.expected_local_ipc),
            "threshold_db": self.detection_threshold,
            "avg_db": round(float(avg_loudness), 3) if frame_count > 0 else None,
            "last_db": round(float(self.last_db), 3) if self.last_db is not None else None,
            "last_detection": bool(last_det),
            "detections": int(num_det),
            "frames": int(frame_count),
            "frames_total": int(self.frames_total),
            "detections_total": int(self.detections_total),
            "frame_rate": round(float(frame_rate), 3),
            "data_rate_bps": round(float(data_rate), 3),
            "last_frame_age_sec": round(now - self.last_frame_wall_ts, 3) if self.last_frame_wall_ts else None,
            "last_detection_age_sec": round(now - self.last_detection_wall_ts, 3) if self.last_detection_wall_ts else None,
            "queue_size": int(self.q.qsize()),
            "diagnosis": self.diagnosis(),
            "last_error": self.last_error,
            "uptime_sec": round(now - self.start_wall_ts, 3),
        }
        self.publish("net", self.status_topic, json.dumps(msg))

    def service_step(self):
        frame_count = 0
        data_len = 0
        loudness = 0.0
        num_det = 0
        last_report_time = time.time()
        last_det = False

        while not self.state == state.quit:
            now = time.time()
            report_due = now - last_report_time > 1.0
            status_due = self.publish_status and self.status_fps > 0 and now - self.last_status_pub_time >= 1.0 / self.status_fps

            if report_due or status_due:
                delta = max(now - last_report_time, 1e-9)
                frame_rate = frame_count / delta
                data_rate = data_len / delta
                avg_loudness = loudness / (frame_count + 1e-10)
                avg_frame_size = data_len / (frame_count + 1e-10)
                ts = self.ts_to_string(now)

                if report_due and (frame_count > 0 or self.frames_total > 0 or self.print_idle_status):
                    print(
                        f"[{ts}] Audio status: detections={num_det}/{frame_count} avg_loudness={avg_loudness:0.3f}db "
                        f"frame_rate={frame_rate:0.3f}/s data_rate={data_rate:0.3f}b/s avg_frame_size={avg_frame_size:0.1f}b "
                        f"diagnosis={self.diagnosis()}",
                        flush=True,
                    )

                if status_due:
                    self.publish_status_msg(
                        frame_count=frame_count,
                        num_det=num_det,
                        frame_rate=frame_rate,
                        data_rate=data_rate,
                        avg_loudness=avg_loudness,
                        last_det=last_det,
                    )

                if report_due:
                    frame_count = 0
                    data_len = 0
                    loudness = 0.0
                    num_det = 0
                    last_report_time = now

            try:
                data = self.q.get(timeout=0.25)
            except queue.Empty:
                continue

            try:
                payload = data["payload"]
                waveform = payload["waveform"].astype(float)
                timestamp = payload["t"]

                frame_count += 1
                self.frames_total += 1
                self.last_frame_wall_ts = time.time()
                rms = np.sqrt(np.mean(waveform[:, [1, 2, 3, 4]] ** 2))
                dbs = 20 * np.log10(1e-16 + rms / 32767)
                det = bool(dbs > self.detection_threshold)
                self.last_db = dbs
                last_det = det
                loudness += dbs
                data_len += 2 * np.prod(waveform.shape)
                num_det += int(det)

                if self.state == state.collecting and self.data_file is not None:
                    ts_string = self.ts_to_string(timestamp)
                    line = f"{ts_string},{dbs:0.3f}\n"
                    with self.lock:
                        self.data_file.write(line)

                if det:
                    self.detections_total += 1
                    self.last_detection_wall_ts = time.time()
                    msg = json.dumps({
                        "kind": "detection",
                        "t": timestamp,
                        "node": self.hostname,
                        "event": "loud_audio",
                        "db": float(dbs),
                        "threshold_db": self.detection_threshold,
                    })
                    self.publish("net", self.net_topic, msg)
            except Exception as exc:
                self.last_error = f"audio_process_failed: {exc}"
                print(self.last_error, flush=True)

            time.sleep(0)

        return True


def main():
    node = audio_detector()
    node.start()


if __name__ == '__main__':
    main()
