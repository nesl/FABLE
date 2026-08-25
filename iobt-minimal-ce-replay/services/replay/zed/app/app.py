#!/usr/bin/python
#Add MCP python library dir to path

import os, sys
sys.path.append("/lib/iobtmax")
from iobt_max_service import iobt_max_service, state

import time
import numpy as np
from datetime import datetime
import queue
import cv2
import pyzed.sl as sl
import base64
import glob
#import bagpy
import pandas as pd
import bisect
import argparse
import json
import threading
from pathlib import Path

APP_VERSION = "zed-replay-readiness-sync-v20260710-1"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


class zed_custom(iobt_max_service):

    def __init__(self,args):
        #Name the app and init the base class 
        self.name = "zed"
        iobt_max_service.__init__(self,self.name)
        print(f"[ZED] app version {APP_VERSION}", flush=True)

        self.data           = []
        self.fps            = 1
        self.frame_count    = 0
        self.playback_start_time     = time.time()
        self.frames         = []
        self.samples        = []
        self.q              = queue.LifoQueue()
        self.rate           = 1./self.fps
        self.data_thread    = None
        self.loop_wall_start = time.time()  # overwritten by sync message
        self.sync_start_at   = time.time()  # absolute wall time to begin frame 0
        self.sync_received   = False
        self._sync_event     = threading.Event()
        self._config_event   = threading.Event()
        self.sync_topic      = "/replay/sync"
        self.config_topic    = "/replay/config"
        self._loaded_video_file  = None   # track what is currently loaded
        self._pending_video_file = None   # track what we've been asked to load
        self._loaded_start_time  = None
        self._loaded_end_time    = None
        self._pending_start_time = None
        self._pending_end_time   = None
        self._replay_control_subscribed = False
        self.replay_child_config_enabled = _env_bool("REPLAY_CHILD_CONFIG_ENABLED", True)
        self._initial_sync_json = os.environ.get("REPLAY_INITIAL_SYNC_JSON", "").strip()
        self._last_sync_signature = None
        self._sync_lock = threading.RLock()

        self.cam            = sl.Camera()
        self.framescale     = 0.25
        self.bitrate        = 30000

        self.lo_res         = sl.Resolution()
        self.lo_res.width   = int(round(1920*self.framescale))
        self.lo_res.height  = int(round(1080*self.framescale))

        #self.video_files = glob.glob(f"/output/20240807_*_{self.hostname}_zed.svo")
        self.video_file = f"/output/{args.scenario}_{self.hostname}_zed.svo2"
        self.timestamp_file = f"/output/{args.scenario}_{self.hostname}_zed.csv"

        self.playback_start_time = args.start
        self.playback_end_time = args.end
        self.playback_mode = getattr(args, "playback_mode", "max")
        self.playback_speed = float(getattr(args, "speed", 1.0) or 1.0)
        self._normalize_playback_mode()
        self.status_interval = float(os.environ.get("REPLAY_STATUS_INTERVAL", "1.0"))

        self.send_rgb_mqtt   = not args.no_rgb_mqtt
        self.send_depth_mqtt = not args.no_depth_mqtt
        self.downsample_for_mqtt = args.downsample_for_mqtt

    def service_initialize(self):

        #Initialize any variables that persist until shutdown
        #and call code that runs once only

        print("Initializing zed replay service") 

        self.init                   = sl.InitParameters()
        #self.init.depth_mode        = sl.DEPTH_MODE.QUALITY
        self.init.depth_mode        = sl.DEPTH_MODE.PERFORMANCE
        self.init.camera_resolution = sl.RESOLUTION.HD1080
        self.init.coordinate_units  = sl.UNIT.METER
        self.init.camera_fps        = 15
        self.init.depth_maximum_distance = 40.0
        # We do replay timing ourselves so we can support max-speed, real-time,
        # and scaled slower/faster playback with the same synchronization message.
        self.init.svo_real_time_mode = False

        self.runtime_parameters              = sl.RuntimeParameters()
        #self.runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD

        self.last_mqtt_msg_time=0

        self.slow_pub_fps           = 1
        self.fast_pub_fps           = 15

        self.df_timestamps = pd.read_csv(self.timestamp_file)
        self.df_timestamps["Timestamp"] = pd.to_datetime(self.df_timestamps["Timestamp"])
        self.df_timestamps["Timestamp"] = self.df_timestamps["Timestamp"] - self.df_timestamps["Timestamp"].iloc[0]
        self.df_timestamps["Timestamp"] = self.df_timestamps["Timestamp"].dt.total_seconds()
        self.df_timestamps =self.df_timestamps.set_index("Timestamp")
        event_start = os.environ.get("FABLE_REPLAY_EVENT_START", "").strip()
        self.replay_event_start_epoch = (
            datetime.fromisoformat(event_start.replace("Z", "+00:00")).timestamp()
            if event_start
            else None
        )

        self.net_img_topic   = self.get_topic_name("rgb_left/compressed")
        self.net_depth_topic = self.get_topic_name("depth/compressed")
        self.local_data_topic = self.get_topic_name("data")
        self.sync_topic      = "/replay/sync"
        self.config_topic    = "/replay/config"
        if not self._replay_control_subscribed:
            self.subscribe("net", self.sync_topic, self._on_sync)
            if self.replay_child_config_enabled:
                self.subscribe("net", self.config_topic, self._on_config)
            else:
                print("[ZED] Child direct /replay/config subscription disabled; supervisor owns config.", flush=True)
            self._replay_control_subscribed = True
        
        self.img   = sl.Mat(self.lo_res.width, self.lo_res.height, sl.MAT_TYPE.U8_C4, sl.MEM.CPU)
        self.depth = sl.Mat(self.lo_res.width, self.lo_res.height, sl.MAT_TYPE.F32_C1, sl.MEM.CPU)

        self.init.set_from_svo_file(self.video_file)
        status = self.cam.open(self.init) 
        if status != sl.ERROR_CODE.SUCCESS:
            msg = f"ZED SVO file not found or failed to open: {self.video_file} (status={status})"
            print(msg)
            self._publish_error(msg)
            raise FileNotFoundError(msg)


        if(self.playback_end_time==-1):
            self.playback_end_time = self.df_timestamps.index[-1]
        self.total_playback_duration =  self.playback_end_time - self.playback_start_time

        self.playback_start_index = self.get_playback_index(self.playback_start_time)
        self.playback_end_index = self.get_playback_index(self.playback_end_time)
        self.total_playback_steps = self.playback_end_index - self.playback_start_index

        print("Done initiazing zed replay service")  
        self._loaded_video_file  = self.video_file
        self._loaded_start_time  = self.playback_start_time
        self._loaded_end_time    = self.playback_end_time
        self._pending_video_file = self.video_file
        self._pending_start_time = self.playback_start_time
        self._pending_end_time   = self.playback_end_time
        self.publish_replay_ready("data_loaded_waiting_for_sync")

    def _make_sync_signature(self, msg, start_at, mode, speed):
        replay_id = msg.get("replay_id") or msg.get("command_id") or msg.get("session_id")
        if replay_id:
            return ("id", str(replay_id))
        scenario = str(msg.get("scenario", ""))
        return ("sync", scenario, f"{float(start_at):.3f}", str(mode), f"{float(speed):.6f}")

    def _sync_matches_current_scenario(self, msg):
        scenario = str(msg.get("scenario") or "").strip() if isinstance(msg, dict) else ""
        if not scenario:
            return True
        loaded = ""
        try:
            loaded = Path(self.video_file).name.split(f"_{self.hostname}_zed")[0]
        except Exception:
            loaded = ""
        return not loaded or scenario == loaded

    def publish_replay_ready(self, reason, ready=True, **extra):
        try:
            scenario = Path(self.video_file).name.split(f"_{self.hostname}_zed")[0]
        except Exception:
            scenario = ""
        self.publish_readiness(
            "zed",
            ready=ready,
            reason=reason,
            scenario=scenario,
            video_file=self.video_file,
            timestamp_file=self.timestamp_file,
            start_time=self.playback_start_time,
            end_time=self.playback_end_time,
            local_ipc="/tmp/zed.ipc",
            total_playback_duration=getattr(self, "total_playback_duration", None),
            replay_id=os.environ.get("REPLAY_ACTIVE_ID") or None,
            **extra,
        )

    def _normalize_playback_mode(self):
        mode = str(getattr(self, "playback_mode", "max") or "max").lower().strip()
        if mode in {"fast", "asap", "unlimited"}:
            mode = "max"
        if mode not in {"max", "realtime", "scaled"}:
            mode = "max"
        speed = float(getattr(self, "playback_speed", 1.0) or 1.0)
        if mode == "realtime":
            speed = 1.0
        elif mode == "scaled":
            speed = max(speed, 1e-6)
        self.playback_mode = mode
        self.playback_speed = speed

    def _set_playback_timing(self, mode=None, speed=None):
        if mode is not None:
            self.playback_mode = mode
        if speed is not None:
            try:
                self.playback_speed = float(speed)
            except Exception:
                pass
        self._normalize_playback_mode()

    def _timing_speed(self):
        """Return None for max/as-fast-as-possible mode, else a positive speed multiplier."""
        self._normalize_playback_mode()
        if self.playback_mode == "max":
            return None
        return self.playback_speed

    def _sleep_until_playback_time(self, playback_time):
        """Sleep until the shared virtual replay clock reaches playback_time.

        Returns False if interrupted by config/sync/quit; True otherwise.
        In max mode there is intentionally no sleep.
        """
        speed = self._timing_speed()
        if speed is None:
            return True
        target_elapsed = max(0.0, float(playback_time) - float(self.playback_start_time))
        target_wall = self.sync_start_at + target_elapsed / speed
        while True:
            if self.state == state.quit or self._config_event.is_set() or self._sync_event.is_set():
                return False
            remaining = target_wall - time.time()
            if remaining <= 0:
                return True
            time.sleep(min(0.02, remaining))

    def service_initialize_collect(self):
        print("Calling zed replay service collection init")

    def service_stop_collect(self):
        print("Calling zed replay service stop collection")

    def service_stop(self):
        print("Stopping zed replay service")

    def _wait_for_config(self):
        """Block until a new /replay/config message arrives or service quits."""
        print("[ZED] Waiting for new /replay/config message...")
        self._config_event.clear()
        while not self._config_event.is_set():
            if self.state == state.quit:
                return
            time.sleep(0.5)
        self._config_event.clear()

    def _publish_error(self, message):
        """Publish a file-not-found error to the playback controller."""
        try:
            payload = json.dumps({
                "service": "zed_replay",
                "node":    self.hostname,
                "error":   message,
                "t":       self.ts_to_string(time.time())
            })
            self.publish("net", f"/replay/error/zed/{self.hostname}", payload)
        except Exception as e:
            print(f"[ZED] Failed to publish error: {e}")

    def _on_config(self, topic, payload):
        """MQTT callback for scenario config messages.
        Payload: {"scenario": "20250815_083306", "start_time": "0", "end_time": "-1"}
        """
        try:
            msg = json.loads(payload)
            scenario   = msg.get("scenario", None)
            start_time = msg.get("start_time", None)
            end_time   = msg.get("end_time",   None)
            playback_mode = msg.get("playback_mode", msg.get("mode", None))
            playback_speed = msg.get("speed", msg.get("playback_speed", None))
            self._set_playback_timing(playback_mode, playback_speed)

            new_start = float(start_time) if start_time is not None else 0.0
            new_end   = float(end_time)   if end_time   is not None else -1.0

            # If the scenario is blank or None, keep the currently loaded scenario files
            if not scenario:
                print("[ZED] Config message has blank scenario — keeping current files.")
                new_video_file = self._loaded_video_file if self._loaded_video_file else self.video_file
                new_csv        = self.timestamp_file
            else:
                new_video_file = f"/output/{scenario}_{self.hostname}_zed.svo2"
                new_csv        = f"/output/{scenario}_{self.hostname}_zed.csv"

            self.video_file          = new_video_file
            self.timestamp_file      = new_csv
            self.playback_start_time = new_start
            self.playback_end_time   = new_end

            print(f"[ZED] Config updated — scenario='{scenario}' start={new_start} end={new_end} mode={self.playback_mode} speed={self.playback_speed:g}")

            loaded_match  = (new_video_file == self._loaded_video_file 
                             and new_start == self._loaded_start_time 
                             and new_end   == self._loaded_end_time)
            pending_match = (new_video_file == self._pending_video_file 
                             and new_start == self._pending_start_time 
                             and new_end   == self._pending_end_time)

            if not loaded_match and not pending_match:
                self._pending_video_file = new_video_file
                self._pending_start_time = new_start
                self._pending_end_time   = new_end
                self._config_event.set()
            else:
                print("[ZED] Config is same as loaded/pending — ignoring duplicate.")

        except Exception as e:
            print(f"[ZED] Config parse error: {e}")


    def _on_sync(self, topic, payload):
        """MQTT callback for playback controller sync messages.
        The sync message carries an absolute 'start_at' wall time at which all
        nodes should begin playing frame 0.
        """
        try:
            msg = json.loads(payload)
            if not self._sync_matches_current_scenario(msg):
                print(f"[ZED] Ignoring sync for scenario={msg.get('scenario')} while loaded file={self.video_file}", flush=True)
                return
            self._set_playback_timing(msg.get('playback_mode', msg.get('mode', None)), msg.get('speed', msg.get('playback_speed', None)))
            start_at = float(msg.get('start_at', time.time()))
            signature = self._make_sync_signature(msg, start_at, self.playback_mode, self.playback_speed)
            with self._sync_lock:
                if signature == self._last_sync_signature:
                    print(f"[ZED] Duplicate sync ignored signature={signature}", flush=True)
                    return
                self._last_sync_signature = signature
                self.sync_start_at   = start_at
                self.loop_wall_start = self.sync_start_at
                self.sync_received   = True
                self._sync_event.set()
            print(f"[ZED] Sync received — start_at={self.sync_start_at:.3f} "
                  f"(in {self.sync_start_at - time.time():.2f}s) mode={self.playback_mode} speed={self.playback_speed:g} signature={signature}", flush=True)
        except Exception as e:
            print(f"[ZED] Sync parse error: {e}")

    def get_playback_time(self, ts=None):
        if ts is None:
            ts = time.time()
        speed = self._timing_speed()
        if speed is None:
            elapsed = 0.0
        else:
            elapsed = (max(0.0, ts - self.sync_start_at) * speed) % max(self.total_playback_duration, 1e-6)
        return self.playback_start_time + elapsed

    def _consume_pending_sync(self):
        """Consume the generation used to start playback without losing a newer one."""
        with self._sync_lock:
            self._sync_event.clear()
    
    def get_playback_index(self,playback_time):
        return self.df_timestamps.index.searchsorted(playback_time)

    def _maybe_load_initial_sync(self):
        raw = getattr(self, "_initial_sync_json", "")
        if not raw:
            return False
        # Consume once.  This seed was copied from the MQTT sync already seen by
        # the supervisor before the child subscribed.
        self._initial_sync_json = ""
        os.environ["REPLAY_INITIAL_SYNC_JSON"] = ""
        try:
            self._on_sync(self.sync_topic, raw)
            return True
        except Exception as exc:
            print(f"[ZED] Failed to apply initial sync from env: {exc}", flush=True)
            return False

    def _wait_for_sync_or_config(self):
        """Block until sync or config arrives. Returns 'sync', 'config', or 'quit'."""
        while True:
            if self.state == state.quit:
                return 'quit'
            if self._sync_event.is_set():
                self._sync_event.clear()
                return 'sync'
            if self._maybe_load_initial_sync():
                self._sync_event.clear()
                return 'sync'
            if self._config_event.is_set():
                return 'config'
            time.sleep(0.1)

    def service_step(self):
        """
        State machine identical to respeaker:
          LOAD DATA → WAIT SYNC → PLAY → (end: wait sync, config: reload, sync: restart)
        """
        frame_count            = 0
        service_step_start_time = time.time()
        data_loaded            = False
        sync_ready             = False  # True when we have a valid loop_wall_start

        while True:  # top-level state machine
            if self.state == state.quit:
                self.cam.close()
                return

            # ── LOAD DATA ────────────────────────────────────────────────────
            if self._config_event.is_set() or not data_loaded:
                had_sync = bool(self.sync_received or self._sync_event.is_set())
                self._config_event.clear()
                print(f"[ZED] Loading data... had_sync={had_sync}", flush=True)
                try:
                    self.cam.close()
                except Exception:
                    pass
                try:
                    self.service_initialize()
                    data_loaded = True
                except FileNotFoundError as e:
                    print(f"[ZED] File not found: {e} — waiting for new config...")
                    data_loaded = False
                    self._config_event.clear()
                    self._sync_event.clear()
                    while not self._config_event.is_set():
                        if self.state == state.quit:
                            return
                        time.sleep(0.1)
                    continue
                except Exception as e:
                    print(f"[ZED] Init failed: {e}")
                    time.sleep(2)
                    continue

                # Drain duplicate config messages that piled up during load
                if self._config_event.is_set():
                    print("[ZED] Draining config events that arrived during load.")
                    # _on_config suppresses duplicates via _pending_video_file,
                    # so if _config_event is still set it's a genuinely new config
                    continue
                sync_ready = bool(had_sync or self.sync_received or self._sync_event.is_set())
                if sync_ready:
                    print("[ZED] Preserved sync across data load — starting without waiting.", flush=True)

            # ── WAIT FOR SYNC ─────────────────────────────────────────────────
            # Skip if we already have a valid sync (e.g. interrupted mid-playback)
            if not sync_ready:
                if self._sync_event.is_set():
                    print(f"[ZED] Sync already received — starting playback immediately.")
                    self._sync_event.clear()
                else:
                    print(f"[ZED] Waiting for sync on {self.sync_topic}...")
                    result = self._wait_for_sync_or_config()
                    if result == 'quit':
                        self.cam.close()
                        return
                    if result == 'config':
                        continue
            sync_ready = False  # consume the sync; next iteration must wait again
            self._consume_pending_sync()

            # ── PLAY ──────────────────────────────────────────────────────────
            # Wait until start_at if it's in the future
            now = time.time()
            if self.sync_start_at > now:
                wait = self.sync_start_at - now
                print(f"[ZED] Waiting {wait:.2f}s until synchronized start_at={self.sync_start_at:.3f}")
                deadline = time.time() + wait
                while time.time() < deadline:
                    if self.state == state.quit:
                        self.cam.close()
                        return
                    if self._sync_event.is_set(): break
                    if self._config_event.is_set(): break
                    time.sleep(min(0.05, deadline - time.time()))
                if self._config_event.is_set():
                    sync_ready = False
                    continue
                if self._sync_event.is_set():
                    sync_ready = True
                    self._sync_event.clear()
                    continue

            print(f"[ZED] Starting playback sync_start_at={self.sync_start_at:.3f} mode={self.playback_mode} speed={self.playback_speed:g}")

            current_playback_time  = self.get_playback_time()
            current_playback_index = self.get_playback_index(current_playback_time)

            print(f"Start time: {self.playback_start_time:.2f}s Start Index: {self.playback_start_index}")
            print(f"End time: {self.playback_end_time:.2f}s End Index: {self.playback_end_index}")
            print(f"Total Time: {self.total_playback_duration}. Current Time: {current_playback_time:.2f}. Current Step: {current_playback_index}")

            self.cam.set_svo_position(current_playback_index)
            num_playback_steps = self.playback_end_index - current_playback_index + 1

            interrupted_by_sync   = False
            interrupted_by_config = False

            for i in range(num_playback_steps):
                if self.state == state.quit:
                    self.cam.close()
                    return
                if self._config_event.is_set():
                    interrupted_by_config = True
                    break
                if self._sync_event.is_set():
                    self._sync_event.clear()
                    interrupted_by_sync = True
                    print("[ZED] New sync received during playback — restarting.")
                    break

                target_playback_time = self.df_timestamps.index[current_playback_index] if current_playback_index < len(self.df_timestamps) else self.playback_end_time
                if not self._sleep_until_playback_time(target_playback_time):
                    if self._config_event.is_set():
                        interrupted_by_config = True
                    elif self._sync_event.is_set():
                        self._sync_event.clear()
                        interrupted_by_sync = True
                    break

                grab_status = self.cam.grab()
                if grab_status == sl.ERROR_CODE.SUCCESS:
                    timestamp = (
                        self.replay_event_start_epoch + float(target_playback_time)
                        if self.replay_event_start_epoch is not None
                        else time.time()
                    )
                    self.cam.retrieve_image(self.img, sl.VIEW.LEFT, resolution=self.lo_res)
                    self.cam.retrieve_measure(self.depth, sl.MEASURE.DEPTH, resolution=self.lo_res)
                    self.process_data(timestamp, self.img, self.depth)
                    self.frame_count += 1
                    frame_count += 1
                elif grab_status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    print(f"[ZED] End of SVO file reached.")
                    break
                else:
                    print(f"[ZED] Frame grab failed: {grab_status}")

                if time.time() - service_step_start_time >= self.status_interval:
                    ts    = time.time()
                    t     = datetime.fromtimestamp(ts).strftime("%Y/%m/%d %H:%M:%S.%f")
                    delta = ts - service_step_start_time
                    print(f"[{t}] Replay State: {self.state} Frame rate: {frame_count/delta:.3f}/s")
                    if current_playback_index < len(self.df_timestamps):
                        elapsed = max(0.0, float(self.df_timestamps.index[current_playback_index]) - float(self.playback_start_time))
                    else:
                        elapsed = self.total_playback_duration
                    pct     = 100.0 * elapsed / max(self.total_playback_duration, 1e-6)
                    status_msg = json.dumps({
                        "service":    "zed_replay",
                        "node":       self.hostname,
                        "start_time": self.playback_start_time,
                        "end_time":   self.playback_end_time,
                        "current":    round(elapsed, 2),
                        "duration":   round(self.total_playback_duration, 2),
                        "pct":        round(pct, 1),
                        "playback_mode": self.playback_mode,
                        "requested_speed_x": None if self.playback_mode == "max" else self.playback_speed,
                        "t":          self.ts_to_string(ts)
                    })
                    self.publish("net", f"/replay/status/zed/{self.hostname}", status_msg)
                    service_step_start_time = time.time()
                    frame_count = 0

                current_playback_index += 1
                if current_playback_index < len(self.df_timestamps):
                    current_playback_time = self.df_timestamps.index[current_playback_index]

            # ── END OF PLAYBACK WINDOW ────────────────────────────────────────
            if interrupted_by_config:
                sync_ready = False
                continue  # reload data at top of loop
            elif interrupted_by_sync:
                sync_ready = True  # sync already captured in loop_wall_start
                continue  # skip WAIT SYNC, go straight to PLAY
            else:
                # Natural end — publish a final progress update even if max-speed replay
                # completed before the periodic status interval elapsed.
                try:
                    ts = time.time()
                    status_msg = json.dumps({
                        "service":    "zed_replay",
                        "node":       self.hostname,
                        "event":      "complete",
                        "start_time": self.playback_start_time,
                        "end_time":   self.playback_end_time,
                        "current":    round(self.total_playback_duration, 2),
                        "duration":   round(self.total_playback_duration, 2),
                        "pct":        100.0,
                        "playback_mode": self.playback_mode,
                        "requested_speed_x": None if self.playback_mode == "max" else self.playback_speed,
                        "speed_x": round(self.total_playback_duration / max(ts - self.sync_start_at, 1e-6), 3),
                        "t":          self.ts_to_string(ts)
                    })
                    self.publish("net", f"/replay/status/zed/{self.hostname}", status_msg)
                except Exception as e:
                    print(f"[ZED] Failed to publish final replay status: {e}")
                # Natural end — wait for next sync from playback controller
                print("[ZED] End of playback window — waiting for sync to loop...")
                self.sync_received = False
                result = self._wait_for_sync_or_config()
                if result == 'quit':
                    self.cam.close()
                    return
                sync_ready = (result == 'sync')
                continue


    def process_data(self,timestamp, img, depth):

        cv2_img   = cv2.cvtColor(img.get_data(), cv2.COLOR_BGRA2BGR)

        img_depth = depth.get_data()
        scale = 65535/(self.init.depth_maximum_distance)
        img_depth = scale*img_depth
        img_depth = np.nan_to_num(img_depth, nan=0, posinf=65535, neginf=0,copy=False)
        np.clip(img_depth, a_min=0, a_max=65535, out=img_depth)
        img_depth = img_depth.astype(np.uint16)

        _, encoded_image = cv2.imencode('.png', cv2_img)
        _, encoded_depth = cv2.imencode('.png', img_depth)

        replay_id = (
            self._last_sync_signature[1]
            if self._last_sync_signature
            and self._last_sync_signature[0] == "id"
            else None
        )
        msg = {
            "t": timestamp,
            "i": encoded_image,
            "d": encoded_depth,
            "replay_id": replay_id,
        }
        self.publish("local","data",msg)

        if(time.time()-self.last_mqtt_msg_time>1/self.slow_pub_fps):
            if self.send_rgb_mqtt:
                cv2_img = cv2.resize(cv2_img, (0, 0), fx=self.downsample_for_mqtt, fy= self.downsample_for_mqtt)
                _, encoded_image = cv2.imencode('.png', cv2_img)
                self.publish("net",self.net_img_topic, base64.b64encode(encoded_image))

            if self.send_depth_mqtt:
                img_depth = cv2.resize(img_depth, (0, 0), fx= self.downsample_for_mqtt, fy= self.downsample_for_mqtt)
                _, encoded_depth = cv2.imencode('.png', img_depth)
                self.publish("net",self.net_depth_topic, base64.b64encode(encoded_depth))

            self.last_mqtt_msg_time = time.time()

def main():     

    parser = argparse.ArgumentParser(description="zed sensor app")
    parser.add_argument('--scenario', type=str, default='20241009_120237', help='Scenario file pattern')
    parser.add_argument('--start', type=int, default=0, help='Start time in seconds')
    parser.add_argument('--end', type=int, default=-1, help='End time in seconds')
    parser.add_argument('--no_rgb_mqtt', action='store_true', help='Send RGB mqtt messages?')
    parser.add_argument('--no_depth_mqtt', action='store_true', help='Send depth mqtt messages?')
    parser.add_argument('--downsample_for_mqtt', type=float, default=1, help='MQTT image downsampling factor')
    parser.add_argument('--playback-mode', choices=['max', 'realtime', 'scaled'], default='max',
                        help='Replay timing mode: max/as-fast-as-possible, realtime, or scaled speed.')
    parser.add_argument('--speed', type=float, default=1.0, help='Speed multiplier for scaled mode; realtime forces 1.0.')

    args = parser.parse_args()

    service = zed_custom(args)
    service.start()

if __name__ == '__main__':
    main()
