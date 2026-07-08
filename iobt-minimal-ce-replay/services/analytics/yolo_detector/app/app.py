import sys, os
sys.path.append("/lib/iobtmax")

from iobt_max_service import iobt_max_service, state

import time
import cv2
import numpy as np
import json
from datetime import datetime
from ultralytics import YOLO
import torch
try:
    import spatial_transform_utils as st
    SPATIAL_IMPORT_ERROR = None
except Exception as exc:
    st = None
    SPATIAL_IMPORT_ERROR = exc
import queue
from threading import Lock
import base64
import traceback


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


class yolo_detector(iobt_max_service):
    def __init__(self):

        if "SERVICE_NAME" in os.environ:
            name = os.environ["SERVICE_NAME"]
            if name == "service":
                name = "yolo"
        else:
            name = "yolo"

        iobt_max_service.__init__(self, name)

        if "SOURCE" in os.environ:
            self.source_host = os.environ["SOURCE"]
            if self.source_host == "local":
                self.source_host = self.hostname
        else:
            self.source_host = self.hostname

        self.source_mode = "local" if self.source_host == self.hostname else "mqtt"
        self.node_short_name = self.get_node_short_name(self.source_host)
        self.lock = Lock()

        env_path = os.environ.get("ENV_INFO", "environments/gq/env_info.json")
        try:
            with open(env_path) as env_file:
                self.info = json.load(env_file)
        except Exception as exc:
            print(f"Could not load environment info from {env_path}: {exc}", flush=True)
            self.info = {"nodes": {}}

        if st is None:
            self.this_node_info = None
            self.point_projector = None
            print(f"spatial_transform_utils unavailable; world coordinates disabled: {SPATIAL_IMPORT_ERROR}", flush=True)
        elif self.node_short_name in self.info.get("nodes", {}):
            self.this_node_info = self.info["nodes"][self.node_short_name]
            self.point_projector = st.point_projector(self.this_node_info)
        else:
            self.this_node_info = None
            self.point_projector = None
            print(f"No environment info found for {self.node_short_name}", flush=True)

        self.load_model = env_bool("LOAD_MODEL", True)
        self.model_loaded = False
        self.model = None
        self.yolo_device = "cpu"

        # Optional derived/debug stream for the web UI. This is intentionally low-rate.
        self.publish_annotated = env_bool("YOLO_PUBLISH_ANNOTATED", False)
        self.annotated_fps = float(os.environ.get("YOLO_ANNOTATED_FPS", "1.0"))
        self.annotated_width = int(os.environ.get("YOLO_ANNOTATED_WIDTH", "960"))
        self.annotated_jpeg_quality = int(os.environ.get("YOLO_ANNOTATED_JPEG_QUALITY", "70"))
        self.last_annotated_pub_time = 0.0

        # Optional low-rate diagnostic stream for the web UI. Disabled by default
        # so synthetic health messages are not mixed into the replay/event stream.
        self.publish_status_enabled = env_bool("YOLO_PUBLISH_STATUS", False)
        self.status_fps = float(os.environ.get("YOLO_STATUS_FPS", "1.0"))
        # Keep startup clean by default: no synthetic debug/status MQTT messages
        # until replay has actually been requested or a frame has arrived.
        self.publish_pre_replay_status = env_bool("YOLO_PUBLISH_PRE_REPLAY_DEBUG_STATUS", False)
        self.print_idle_status = env_bool("YOLO_PRINT_IDLE_STATUS", False)
        self.last_status_pub_time = 0.0
        self.start_wall_ts = time.time()
        self.replay_requested = False
        self.last_replay_config = None
        self.last_replay_sync = None
        self.replay_request_wall_ts = None
        self.input_frames_total = 0
        self.detections_total = 0
        self.last_frame_wall_ts = None
        self.last_frame_data_ts = None
        self.last_detection_wall_ts = None
        self.last_detections = 0
        self.last_error = None
        self.last_publish_error = None
        self.expected_local_ipc = "/tmp/zed.ipc" if self.source_mode == "local" else None
        self.last_frame_shape = None
        self.last_payload_keys = None
        self.last_rgb_bytes = None
        self.decode_failures = 0
        self.process_failures = 0
        self.model_loading = False

        self.yolo_topic = f"/{self.source_host}/analytics/yolo/bbox"
        self.annotated_topic = os.environ.get("YOLO_ANNOTATED_TOPIC", f"/debug/{self.source_host}/analytics/yolo/annotated/compressed")
        self.status_topic = os.environ.get("YOLO_STATUS_TOPIC", f"/debug/{self.source_host}/analytics/yolo/status")
        self.publish_frame_status_enabled = env_bool("YOLO_PUBLISH_FRAME_STATUS", self.publish_status_enabled)
        self.frame_status_fps = float(os.environ.get("YOLO_FRAME_STATUS_FPS", str(self.status_fps)))
        self.frame_status_topic = os.environ.get("YOLO_FRAME_STATUS_TOPIC", f"/debug/{self.source_host}/analytics/yolo/frame")
        self.last_frame_status_pub_time = 0.0
        self.q = queue.Queue()

        self.softfail = False
        self.remote_work_buffer = {}

        if self.load_model:
            try:
                self.model_loading = True
                self.publish_status(force=True)
                print("Starting model load (this may take 30s)...", flush=True)
                if torch.cuda.is_available() and os.environ.get("YOLO_DEVICE", "auto") != "cpu":
                    torch.cuda.set_device(0)
                    self.yolo_device = 0
                else:
                    self.yolo_device = "cpu"
                model_path = os.environ.get("YOLO_MODEL", "/app/yolov8n.pt")
                print(f"Loading YOLO model {model_path} on device={self.yolo_device}", flush=True)
                self.model = YOLO(model_path)
                print(self.model.info(), flush=True)
                # Optional warmup. This should never publish detections.
                cv_rgb = cv2.imread("bus.jpg")
                if cv_rgb is not None:
                    self.detect(cv_rgb)
                self.model_loaded = True
                print("Finished loading model", flush=True)
            except Exception as exc:
                self.last_error = f"model_load_failed: {exc}"
                print(self.last_error, flush=True)
                traceback.print_exc()
            finally:
                self.model_loading = False
                self.publish_status(force=True)
        else:
            print("Warning: Not loading model. YOLO detector will emit synthetic test detections when frames arrive.", flush=True)


        # Replay-control topics are not event data; they are only used to decide
        # when optional debug diagnostics are allowed to start.
        self.subscribe("net", "/replay/config", self.get_replay_config)
        self.subscribe("net", "/replay/sync", self.get_replay_sync)

        if self.source_host == self.hostname:
            self.subscribe("local", "zed", self.get_local_zed_data)
        else:
            self.subscribe("net", f"/{self.source_host}/zed/rgb_left/compressed", self.get_remote_zed_data)
            self.subscribe("net", f"/{self.source_host}/zed/depth/compressed", self.get_remote_zed_data)

        print(f"Running on {self.hostname} with zed data source {self.source_host} and service name {self.servicename}...", flush=True)
        print(f"YOLO detections topic: {self.yolo_topic}", flush=True)
        print(f"YOLO debug status topic: {self.status_topic} enabled={self.publish_status_enabled} fps={self.status_fps}", flush=True)
        print(f"YOLO debug annotated-image topic: {self.annotated_topic} enabled={self.publish_annotated} fps={self.annotated_fps}", flush=True)
        print(f"YOLO debug frame-probe topic: {self.frame_status_topic} enabled={self.publish_frame_status_enabled} fps={self.frame_status_fps}", flush=True)

        self.step_frame_count = 0
        self.frame_count = 0
        self.max_depth = 40
        self.publish_status(force=True)

    def get_replay_config(self, topic, msg) -> None:
        self.last_replay_config = msg

    def get_replay_sync(self, topic, msg) -> None:
        self.replay_requested = True
        self.last_replay_sync = msg
        self.replay_request_wall_ts = time.time()
        # Now that replay has been requested, debug status can report whether
        # frames are arriving. This remains under /debug/... only.
        self.publish_status(force=True)

    def get_local_zed_data(self, data):
        if self.softfail:
            return
        self.q.put(data)
        time.sleep(0)

    def get_remote_zed_data(self, topic, msg) -> None:
        if self.softfail:
            return

        node = topic.split("/")[1]
        if node not in self.remote_work_buffer:
            self.remote_work_buffer[node] = {"rgb": None, "depth": None, "rgb_time": 0, "depth_time": 0}
        if "rgb" in topic:
            self.remote_work_buffer[node]["rgb"] = base64.b64decode(msg)
            self.remote_work_buffer[node]["rgb_time"] = time.time()
        if "depth" in topic:
            self.remote_work_buffer[node]["depth"] = base64.b64decode(msg)
            self.remote_work_buffer[node]["depth_time"] = time.time()

        if self.remote_work_buffer[node]["depth"] is not None and self.remote_work_buffer[node]["rgb"] is not None:
            time_delta = np.abs(self.remote_work_buffer[node]["rgb_time"] - self.remote_work_buffer[node]["depth_time"])
            if time_delta < 0.5:
                payload = {"t": time.time() * 1e6, "i": self.remote_work_buffer[node]["rgb"], "d": self.remote_work_buffer[node]["depth"]}
                msg = {"topic": "data", "node": node, "payload": payload}
                self.q.put(msg)
                self.remote_work_buffer[node]["depth"] = None
                self.remote_work_buffer[node]["rgb"] = None

        time.sleep(0)

    def service_initialize(self):
        print("Calling yolo-detector service initialize", flush=True)

    def service_stop(self):
        print("Calling yolo-detector service stop", flush=True)

    def service_initialize_collect(self):
        print("Calling yolo-detector init collect", flush=True)
        self.frame_count = 0
        self.dets_file_name = self.get_file_name("json")
        with self.lock:
            self.dets_file = open(self.dets_file_name, "w")
            self.dets_file.write("[\n")
        self.collection_initialized = True

    def service_stop_collect(self):
        print("Calling yolo-detector stop collect", flush=True)
        self.collection_initialized = False
        with self.lock:
            self.dets_file.write("\n]")
            self.dets_file.close()

    def service_control_callback(self, topic, msg) -> None:
        print(f"Got node service control callback msg: {msg}", flush=True)
        if " " in msg:
            parts = msg.split(" ")
            cmd = parts[0]
        else:
            cmd = msg

        if cmd == "start_softfail":
            self.softfail = True
            self.publish_status(force=True)
        elif cmd == "stop_softfail":
            self.softfail = False
            self.publish_status(force=True)
        else:
            print(f"Got unknown control message: {msg}", flush=True)

    def service_step(self):
        self.step_frame_count = 0
        self.detection_count = 1e-16
        self.start_time = time.time()
        self.obj_count = {}
        self.total_latency = 0

        while not self.state == state.quit and time.time() - self.start_time < 5:
            self.publish_status()
            try:
                data = self.q.get(timeout=0.25)
            except queue.Empty:
                continue

            try:
                self.process_data(data)
            except Exception as exc:
                self.process_failures += 1
                self.last_error = f"process_data_failed: {exc}"
                print(self.last_error, flush=True)
                traceback.print_exc()
                self.publish_status(force=True)
            time.sleep(0)

        if self.step_frame_count > 0 or self.print_idle_status:
            self.report_stats()
        self.publish_status(force=True)

    def process_data(self, data):
        payload = data["payload"]
        self.last_payload_keys = sorted(list(payload.keys()))

        if "i" not in payload:
            self.last_error = f"received_zed_payload_without_rgb_i_key keys={self.last_payload_keys}"
            self.publish_status(force=True)
            return

        buf = np.frombuffer(payload["i"], dtype=np.uint8)
        self.last_rgb_bytes = int(buf.size)
        cv_rgb = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if cv_rgb is None:
            self.decode_failures += 1
            self.last_error = f"received_zed_frame_but_rgb_decode_failed bytes={self.last_rgb_bytes}"
            self.publish_status(force=True)
            return
        self.last_frame_shape = f"{cv_rgb.shape[1]}x{cv_rgb.shape[0]}"

        cv_depth = None
        if "d" in payload and payload["d"] is not None:
            buf = np.frombuffer(payload["d"], dtype=np.uint8)
            cv_depth = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if cv_depth is None:
                self.last_error = "received_zed_frame_but_depth_decode_failed"

        # Historical local path uses seconds; remote path above uses microseconds.
        raw_t = float(payload.get("t", time.time()))
        t_seconds = raw_t / 1e6 if raw_t > 1e12 else raw_t
        t = self.ts_to_string(t_seconds)

        self.input_frames_total += 1
        self.last_frame_wall_ts = time.time()
        self.last_frame_data_ts = t_seconds

        if self.load_model and self.model_loaded and self.model is not None:
            dets = self.detect(cv_rgb, cv_depth, t)
        elif self.load_model and not self.model_loaded:
            dets = []
        else:
            dets = [{"node": self.node_short_name, "source_host": self.source_host, "model": "no_model", "class": "test", "conf": 1.0, "box": [100, 80, 120, 100], "depth": -1.0, "world": [], "t": t}]

        self.last_detections = len(dets)
        self.publish_frame_status(t)
        if dets:
            self.last_detection_wall_ts = time.time()
            self.detections_total += len(dets)
            dets_payload = json.dumps(dets)
            try:
                self.publish("net", self.yolo_topic, dets_payload)
            except Exception as exc:
                self.last_publish_error = f"bbox_publish_failed: {exc}"
                print(self.last_publish_error, flush=True)
            if self.state == state.collecting:
                with self.lock:
                    if self.frame_count >= 1:
                        self.dets_file.write(",\n")
                    self.dets_file.write(dets_payload)

        self.publish_annotated_frame(cv_rgb, dets, t)
        self.publish_status()

        self.frame_count += 1
        self.step_frame_count += 1
        self.detection_count += len(dets)
        self.total_latency += max(0.0, time.time() - t_seconds)
        for det in dets:
            self.obj_count[det["class"]] = self.obj_count.get(det["class"], 0) + 1

    def detect(self, cv_rgb, cv_depth=None, t=0):
        if self.model is None:
            return []
        results = self.model(cv_rgb, verbose=False, device=getattr(self, "yolo_device", None))
        dets = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                c = r.names[int(box.cls.item())]
                b = [int(x) for x in box.xywh[0].cpu().numpy()]
                p = float(box.conf.cpu().item())

                if cv_depth is None:
                    d = -1.0
                    w = []
                else:
                    cx = min(max(b[0], 0), cv_depth.shape[1] - 1)
                    cy = min(max(b[1], 0), cv_depth.shape[0] - 1)
                    dp = cv_depth[cy, cx]
                    if dp <= 6:
                        d = -1.0
                        w = []
                    else:
                        d = round(self.max_depth * (float(dp) / 65535.0), 3)
                        s = 1080 / cv_rgb.shape[0]
                        points_img = torch.tensor([[s * b[0], s * b[1], d]]).float()
                        if self.point_projector is not None:
                            with torch.no_grad():
                                points_world = self.point_projector.image_to_world(points_img)
                            w = [round(float(x), 3) for x in points_world[0, :]]
                        else:
                            w = [0.0, 0.0, 0.0]

                dets.append({"node": self.node_short_name, "source_host": self.source_host, "model": "yolov8", "class": c, "conf": p, "box": b, "depth": d, "world": w, "t": t})

        return dets

    def publish_frame_status(self, t):
        """Publish a low-rate debug heartbeat only after a real ZED frame was decoded.

        This is not replay data and not a detector event. It lives under /debug/... so
        the device page can answer: is YOLO receiving images at all?
        """
        if not self.publish_frame_status_enabled or self.frame_status_fps <= 0:
            return
        now = time.time()
        if now - self.last_frame_status_pub_time < 1.0 / self.frame_status_fps:
            return
        self.last_frame_status_pub_time = now
        payload = {
            "kind": "debug_frame_probe",
            "synthetic_debug": True,
            "t": now,
            "data_t": t,
            "node": self.node_short_name,
            "hostname": self.hostname,
            "source_host": self.source_host,
            "source_mode": self.source_mode,
            "input_frames_total": int(self.input_frames_total),
            "last_frame_shape": self.last_frame_shape,
            "last_rgb_bytes": self.last_rgb_bytes,
            "last_payload_keys": self.last_payload_keys,
            "decode_failures": int(self.decode_failures),
            "process_failures": int(self.process_failures),
            "model_loaded": bool(self.model_loaded),
            "model_loading": bool(self.model_loading),
            "yolo_device": str(self.yolo_device),
            "last_detections": int(self.last_detections),
            "detections_total": int(self.detections_total),
            "diagnosis": self.diagnosis(),
            "last_error": self.last_error,
        }
        try:
            self.publish("net", self.frame_status_topic, json.dumps(payload))
        except Exception as exc:
            self.last_publish_error = f"frame_status_publish_failed: {exc}"
            print(self.last_publish_error, flush=True)

    def publish_annotated_frame(self, cv_rgb, dets, t):
        if not self.publish_annotated:
            return
        if self.annotated_fps <= 0:
            return
        now = time.time()
        if now - self.last_annotated_pub_time < 1.0 / self.annotated_fps:
            return
        self.last_annotated_pub_time = now

        annotated = cv_rgb.copy()
        h, w = annotated.shape[:2]
        for det in dets:
            cx, cy, bw, bh = [int(v) for v in det.get("box", [0, 0, 0, 0])]
            x1 = max(0, cx - bw // 2)
            y1 = max(0, cy - bh // 2)
            x2 = min(w - 1, cx + bw // 2)
            y2 = min(h - 1, cy + bh // 2)
            label = f"{det.get('class', '?')} {float(det.get('conf', 0.0)):.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        if self.annotated_width > 0 and annotated.shape[1] > self.annotated_width:
            scale = self.annotated_width / float(annotated.shape[1])
            annotated = cv2.resize(annotated, (self.annotated_width, int(annotated.shape[0] * scale)))

        ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.annotated_jpeg_quality])
        if not ok:
            self.last_error = "annotated_jpeg_encode_failed"
            return
        payload = {
            "kind": "annotated_frame",
            "node": self.node_short_name,
            "source_host": self.source_host,
            "t": t,
            "format": "jpg",
            "encoding": "base64",
            "width": int(annotated.shape[1]),
            "height": int(annotated.shape[0]),
            "detections": len(dets),
            "frame_index": int(self.input_frames_total),
            "image": base64.b64encode(encoded).decode("ascii"),
        }
        try:
            self.publish("net", self.annotated_topic, json.dumps(payload))
        except Exception as exc:
            self.last_publish_error = f"annotated_publish_failed: {exc}"
            print(self.last_publish_error, flush=True)

    def diagnosis(self):
        now = time.time()
        if self.softfail:
            return "softfail_enabled"
        if self.source_mode == "local" and self.expected_local_ipc and not os.path.exists(self.expected_local_ipc):
            return "waiting_for_zed_ipc_socket"
        if self.load_model and self.model_loading:
            return "model_loading"
        if self.load_model and not self.model_loaded:
            return "model_not_loaded"
        if self.input_frames_total == 0:
            return "waiting_for_video_frames"
        frame_age = now - self.last_frame_wall_ts if self.last_frame_wall_ts else None
        if frame_age is not None and frame_age > 5.0:
            return "video_frames_stale"
        if self.last_detections > 0:
            return "active_detections"
        return "video_active_no_yolo_detections"

    def state_string(self):
        try:
            return self.state.name
        except Exception:
            return str(self.state)

    def publish_status(self, force=False):
        if not self.publish_status_enabled or self.status_fps <= 0:
            return
        # Do not emit synthetic debug/status messages at container startup.
        # They are allowed only after replay is requested, after frames arrive,
        # or when explicitly overridden for low-level debugging.
        if self.input_frames_total == 0 and not self.replay_requested and not self.publish_pre_replay_status:
            return
        now = time.time()
        if not force and now - self.last_status_pub_time < 1.0 / self.status_fps:
            return
        self.last_status_pub_time = now
        local_ipc_exists = os.path.exists(self.expected_local_ipc) if self.expected_local_ipc else None
        payload = {
            "kind": "debug_status",
            "synthetic_debug": True,
            "t": now,
            "node": self.node_short_name,
            "hostname": self.hostname,
            "source_host": self.source_host,
            "source_mode": self.source_mode,
            "replay_requested": bool(self.replay_requested),
            "last_replay_config": self.last_replay_config,
            "last_replay_sync": self.last_replay_sync,
            "replay_request_age_sec": round(now - self.replay_request_wall_ts, 3) if self.replay_request_wall_ts else None,
            "expected_local_ipc": self.expected_local_ipc,
            "local_ipc_exists": local_ipc_exists,
            "load_model": bool(self.load_model),
            "model_loaded": bool(self.model_loaded),
            "model_loading": bool(self.model_loading),
            "yolo_device": str(self.yolo_device),
            "publish_annotated": bool(self.publish_annotated),
            "input_frames_total": int(self.input_frames_total),
            "detections_total": int(self.detections_total),
            "last_detections": int(self.last_detections),
            "queue_size": int(self.q.qsize()),
            "last_frame_age_sec": round(now - self.last_frame_wall_ts, 3) if self.last_frame_wall_ts else None,
            "last_detection_age_sec": round(now - self.last_detection_wall_ts, 3) if self.last_detection_wall_ts else None,
            "last_frame_shape": self.last_frame_shape,
            "last_payload_keys": self.last_payload_keys,
            "last_rgb_bytes": self.last_rgb_bytes,
            "decode_failures": int(self.decode_failures),
            "process_failures": int(self.process_failures),
            "state": self.state_string(),
            "diagnosis": self.diagnosis(),
            "last_error": self.last_error,
            "last_publish_error": self.last_publish_error,
            "uptime_sec": round(now - self.start_wall_ts, 3),
        }
        try:
            self.publish("net", self.status_topic, json.dumps(payload))
        except Exception as exc:
            self.last_publish_error = f"status_publish_failed: {exc}"
            print(self.last_publish_error, flush=True)

    def report_stats(self):
        t = datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")
        frame_rate = self.step_frame_count / max(time.time() - self.start_time, 1e-9)
        det_rate = self.detection_count / max(time.time() - self.start_time, 1e-9)
        class_info = " ".join([f"{c}: {self.obj_count[c] / self.detection_count:.3f}" for c in self.obj_count])
        avg_latency = self.total_latency / (1e-4 + self.step_frame_count)
        print(f"[{t}] Input Rate: {frame_rate:.2f}/fps Det Rate: {det_rate:.2f}/s Latency: {avg_latency:.4f}s [{class_info}]", flush=True)


def main():
    node = yolo_detector()
    node.start()


if __name__ == '__main__':
    main()
