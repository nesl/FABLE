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

class zed_custom(iobt_max_service):

    def __init__(self,args):
        #Name the app and init the base class 
        self.name = "zed"
        iobt_max_service.__init__(self,self.name)

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
        self.init.svo_real_time_mode = True

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

        self.net_img_topic   = self.get_topic_name("rgb_left/compressed")
        self.net_depth_topic = self.get_topic_name("depth/compressed")
        self.local_data_topic = self.get_topic_name("data")
        self.sync_topic      = "/replay/sync"
        self.config_topic    = "/replay/config"
        self.subscribe("net", self.sync_topic, self._on_sync)
        self.subscribe("net", self.config_topic, self._on_config)
        
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

            print(f"[ZED] Config updated — scenario='{scenario}' start={new_start} end={new_end}")

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
            self.sync_start_at   = float(msg.get('start_at', time.time()))
            self.loop_wall_start = self.sync_start_at
            self.sync_received   = True
            self._sync_event.set()
            print(f"[ZED] Sync received — start_at={self.sync_start_at:.3f} "
                  f"(in {self.sync_start_at - time.time():.2f}s)")
        except Exception as e:
            print(f"[ZED] Sync parse error: {e}")

    def get_playback_time(self, ts=None):
        if ts is None:
            ts = time.time()
        elapsed = max(0.0, ts - self.sync_start_at) % max(self.total_playback_duration, 1e-6)
        return self.playback_start_time + elapsed
    
    def get_playback_index(self,playback_time):
        return self.df_timestamps.index.searchsorted(playback_time)

    def _wait_for_sync_or_config(self):
        """Block until sync or config arrives. Returns 'sync', 'config', or 'quit'."""
        while True:
            if self.state == state.quit:
                return 'quit'
            if self._sync_event.is_set():
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
                self._config_event.clear()
                self._sync_event.clear()
                self.sync_received = False
                print("[ZED] Loading data...")
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
                sync_ready = False  # new data loaded, need a fresh sync

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

            print(f"[ZED] Starting playback sync_start_at={self.sync_start_at:.3f}")

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
                    print("[ZED] Sync received during playback — restarting.")
                    break

                grab_status = self.cam.grab()
                if grab_status == sl.ERROR_CODE.SUCCESS:
                    timestamp = time.time()
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

                if time.time() - service_step_start_time >= 5:
                    ts    = time.time()
                    t     = datetime.fromtimestamp(ts).strftime("%Y/%m/%d %H:%M:%S.%f")
                    delta = ts - service_step_start_time
                    print(f"[{t}] Replay State: {self.state} Frame rate: {frame_count/delta:.3f}/s")
                    elapsed = max(0.0, time.time() - self.sync_start_at) % max(self.total_playback_duration, 1e-6)
                    pct     = 100.0 * elapsed / max(self.total_playback_duration, 1e-6)
                    status_msg = json.dumps({
                        "service":    "zed_replay",
                        "node":       self.hostname,
                        "start_time": self.playback_start_time,
                        "end_time":   self.playback_end_time,
                        "current":    round(elapsed, 2),
                        "duration":   round(self.total_playback_duration, 2),
                        "pct":        round(pct, 1),
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

        msg   =  {"t":timestamp,"i":encoded_image, "d":encoded_depth}
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

    args = parser.parse_args()

    service = zed_custom(args)
    service.start()

if __name__ == '__main__':
    main()
