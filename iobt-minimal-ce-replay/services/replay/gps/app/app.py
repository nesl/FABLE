#!/usr/bin/python

#Add MCP python library dir to path
import os, sys
sys.path.append("/lib/iobtmax")
from iobt_max_service import iobt_max_service, state
import time
import numpy as np
from datetime import datetime
import threading
import pandas as pd
import argparse
import json
import glob

class gps_replay(iobt_max_service):
    def __init__(self,args):
        self.name = "gps_replay"
        # Bypasses the framework race-condition bug
        self.service_control_topic = f"/{self.name}/control"
        iobt_max_service.__init__(self,self.name)
        
        self.data            = []
        self.loop_wall_start = time.time()  
        self.sync_start_at   = time.time()  
        self.sync_received   = False
        self._sync_event     = threading.Event()
        self._config_event   = threading.Event()
        
        self.sync_topic      = "/replay/sync"
        self.config_topic    = "/replay/config"
        
        self.scenario        = args.scenario
        self.input_file      = None
        
        self._loaded_start_time  = None
        self._loaded_end_time    = None
        self.playback_start_time  = args.start
        self.playback_end_time    = args.end

    def service_initialize(self):
        print("Initializing multi-vehicle gps replay service") 
        
        # 1. Discover ALL GPS files dynamically for this scenario
        search_pattern = f"/data/{self.scenario}/GPS/*/{self.scenario}_*_gps.csv"
        candidates = glob.glob(search_pattern)
        dfs = []
        
        if candidates:
            print(f"[GPS] Found {len(candidates)} active vehicles to replay.")
            for f in candidates:
                print(f"  -> Loading vehicle file: {f}")
                df = pd.read_csv(f)
                # Check if it is the new format
                if "Latitude" in df.columns:
                    # Extract the folder name (e.g., 'gps1', 'truck1') as the ObjectID
                    object_id = os.path.basename(os.path.dirname(f))
                    df["ObjectID"] = object_id + "_replay"
                    # Calculate ElapsedTime for this specific stream
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                    t0 = df["Timestamp"].iloc[0]
                    df["ElapsedTime"] = (df["Timestamp"] - t0).dt.total_seconds()
                    df = df.set_index("ElapsedTime")
                    # Rename columns to match output dictionary schema
                    df = df.rename(columns={
                        "Latitude": "Lat",
                        "Longitude": "Lon",
                        "Altitude": "Alt",
                        "Accuracy": "Precision",
                        "FixType": "Fix",
                        "RTKSolution": "Solution",
                        "BatteryVoltage": "Voltage",
                        "BatteryCharge": "Charge",
                        "Temperature": "Temp"
                    })
                    df['Fix'] = df['Fix'].astype(str)
                    df['Solution'] = df['Solution'].astype(str)
                else:
                    # Old format fallback
                    df["ObjectID"] = df["ObjectID"].apply(lambda x: str(x)+"_replay")
                    df = df.set_index("ElapsedTime")
                
                # FIXED: Moved outside of the 'else' block to run for both formats
                dfs.append(df) 
                
            # Combine all vehicle streams and sort chronologically by their elapsed offsets
            self.data = pd.concat(dfs).sort_index()
            self.input_file = f"Merged dataset ({len(candidates)} vehicles)"
        else:
            # Fallback legacy (singular gt_tracks.csv file)
            self.input_file = f"/data/{self.scenario}/{self.hostname}/gt_tracks.csv"
            if not os.path.exists(self.input_file):
                msg = f"GPS file not found: {self.input_file}"
                print(msg)
                self._publish_error(msg)
                raise FileNotFoundError(msg)
            print(f"[GPS] Loading legacy track file: {self.input_file}")
            df = pd.read_csv(self.input_file)
            df["ObjectID"] = df["ObjectID"].apply(lambda x: str(x)+"_replay")
            self.data = df.set_index("ElapsedTime")
            
        # 3. Create discrete MQTT publication topics for each uniquely identified vehicle
        self.objects = self.data["ObjectID"].unique()
        self.gps_topics = {}
        for objectid in self.objects:
            self.gps_topics[objectid] = f"/{objectid}/gps"
            print(f"  -> Registered topic: {self.gps_topics[objectid]}")

        if self.playback_end_time == -1:
            self.playback_end_time = self.data.index[-1]
            
        self.total_playback_duration = self.playback_end_time - self.playback_start_time
        
        self.playback_start_index = self.get_playback_index(self.playback_start_time)
        self.playback_end_index = self.get_playback_index(self.playback_end_time)
        self.total_playback_steps = self.playback_end_index - self.playback_start_index
        
        self.subscribe("net", self.sync_topic,   self._on_sync)
        self.subscribe("net", self.config_topic, self._on_config)
        print("Done initializing gps replay service")  
        
        self._loaded_start_time  = self.playback_start_time
        self._loaded_end_time    = self.playback_end_time

    def service_initialize_collect(self):
        pass

    def service_stop_collect(self):
        pass

    def service_stop(self):
        print("Stopping gps replay service")

    def _wait_for_config(self):
        print("[GPS] Waiting for new /replay/config message...")
        self._config_event.clear()
        while not self._config_event.is_set():
            if self.state == state.quit: return
            time.sleep(0.5)
        self._config_event.clear()

    def _publish_error(self, message):
        try:
            payload = json.dumps({
                "service": "gps_replay",
                "node":    self.hostname,
                "error":   message,
                "t":       self.ts_to_string(time.time())
            })
            self.publish("net", "/replay/error/gps", payload)
        except Exception as e:
            print(f"[GPS] Failed to publish error: {e}")

    def _on_sync(self, topic, payload):
        try:
            msg = json.loads(payload)
            self.sync_start_at   = float(msg.get('start_at', time.time()))
            self.loop_wall_start = self.sync_start_at
            self.sync_received   = True
            self._sync_event.set()
            print(f"[GPS] Sync received — start_at={self.sync_start_at:.3f} (in {self.sync_start_at - time.time():.2f}s)")
        except Exception as e:
            print(f"[GPS] Sync parse error: {e}")

    def _on_config(self, topic, payload):
        try:
            msg = json.loads(payload)
            scenario   = msg.get("scenario",   None)
            start_time = msg.get("start_time", None)
            end_time   = msg.get("end_time",   None)
            
            # FIXED: Fallback to currently loaded times if not specified in the message
            new_start = float(start_time) if start_time is not None else (self._loaded_start_time if self._loaded_start_time is not None else self.playback_start_time)
            new_end   = float(end_time)   if end_time   is not None else (self._loaded_end_time if self._loaded_end_time is not None else self.playback_end_time)

            if not scenario:
                print("[GPS] Config message has blank scenario — keeping current files.")
                new_scenario = self.scenario
            else:
                new_scenario = scenario

            print(f"[GPS] Config updated — scenario='{new_scenario}' start={new_start} end={new_end}")

            # Trigger reload only if scenario/times changed
            if not (new_scenario == self.scenario and new_start == self._loaded_start_time and new_end == self._loaded_end_time):
                self.scenario = new_scenario
                self.playback_start_time = new_start
                self.playback_end_time   = new_end
                self._config_event.set()
            else:
                print("[GPS] Config is same as loaded — ignoring duplicate.")

        except Exception as e:
            print(f"[GPS] Config parse error: {e}")

    def get_playback_time(self, ts=None):
        if ts is None: ts = time.time()
        elapsed = max(0.0, ts - self.sync_start_at) % max(self.total_playback_duration, 1e-6)
        return self.playback_start_time + elapsed

    def get_playback_index(self,playback_time):
        return self.data.index.searchsorted(playback_time)

    def service_step(self):
        while True:  
            if self.state == state.quit: break

            if not hasattr(self, 'total_playback_duration'):
                print("[GPS] No data loaded — waiting for config...")
                self._wait_for_config()
                
            try:
                self.service_initialize()
                # Clear config event so it doesn't immediately break the loop later
                if self._config_event.is_set():
                    self._config_event.clear()
            except FileNotFoundError:
                self._wait_for_config()
                continue
            except Exception as e:
                print(f"[GPS] Init failed: {e}")
                time.sleep(2)
                continue

            msg_count = 0
            print(f"Starting replay with {self.input_file} file...")
            print(f"Looping GPS from {self.playback_start_time} to {self.playback_end_time:.2f}. Duration of {self.total_playback_duration:.2f}s")
            
            if not self.sync_received:
                print(f"[GPS] Waiting for sync on {self.sync_topic}...")
                while not self.sync_received:
                    if self.state == state.quit: return
                    if self._config_event.is_set(): break
                    time.sleep(0.1)
                
                if self._config_event.is_set():
                    continue

            self._sync_event.clear()
            now = time.time()
            if self.sync_start_at > now:
                wait = self.sync_start_at - now
                print(f"[GPS] Waiting {wait:.2f}s until synchronized start_at={self.sync_start_at:.3f}")
                deadline = time.time() + wait
                while time.time() < deadline:
                    if self.state == state.quit: return
                    if self._sync_event.is_set(): break
                    if self._config_event.is_set(): break
                    time.sleep(min(0.05, deadline - time.time()))
                
                if self._config_event.is_set(): continue
                if self._sync_event.is_set():
                    self._sync_event.clear()
                    continue

            print(f"[GPS] Starting playback at sync_start_at={self.sync_start_at:.3f}")
            
            inner_running = True
            while inner_running:
                if self.state == state.quit: return
                if self._config_event.is_set(): break
                
                loop_start_time = time.time()
                current_playback_time  = self.get_playback_time()
                current_playback_index = self.get_playback_index(current_playback_time)
                
                for current_playback_index in np.arange(current_playback_index, self.playback_end_index+1):
                    if self.state == state.quit: break
                    if self._config_event.is_set(): break
                    if self._sync_event.is_set(): break
                    
                    current_playback_time = self.get_playback_time()
                    next_time = self.data.index[current_playback_index]
                    total_wait_duration = next_time - current_playback_time
                    
                    if total_wait_duration > 1:
                        print(f"Next wait duration: {total_wait_duration}")
                        
                    while True:
                        new_playback_time = self.get_playback_time()
                        if next_time - new_playback_time <= 0: break
                        if new_playback_time < current_playback_time: break
                        if self.state == state.quit: break
                        if self._config_event.is_set(): break
                        if self._sync_event.is_set(): break
                        
                        wait_duration = min(0.1, next_time - self.get_playback_time())
                        time.sleep(wait_duration)

                    row = self.data.iloc[current_playback_index]
                    objectid = row['ObjectID']
                    msg = {
                        "t":self.ts_to_string(time.time()),
                        "lt":row['Lat'],
                        "ln":row['Lon'],
                        "al":row['Alt'],
                        "ac":row['Precision'],
                        "fx":row['Fix'].strip() if isinstance(row['Fix'], str) else row['Fix'],
                        "sol":row['Solution'].strip() if isinstance(row['Solution'], str) else row['Solution'],
                        "v":row['Voltage'],
                        "c":row['Charge'],
                        "tmp":row['Temp'],
                        "r":int(row.get("RSSI", 0)),
                        "st":int(0)
                    }
                    self.publish("net", self.gps_topics[objectid], json.dumps(msg))
                    msg_count += 1
                    
                    if time.time() - loop_start_time >= 5:
                        ts    = time.time()
                        t     = datetime.fromtimestamp(ts).strftime("%Y/%m/%d %H:%M:%S.%f")
                        elapsed = max(0.0, time.time() - self.sync_start_at) % max(self.total_playback_duration, 1e-6)
                        pct     = 100.0 * elapsed / self.total_playback_duration
                        status  = json.dumps({
                            "service":    "gps_replay",
                            "node":       self.hostname,
                            "start_time": self.playback_start_time,
                            "end_time":   self.playback_end_time,
                            "current":    round(elapsed, 2),
                            "duration":   round(self.total_playback_duration, 2),
                            "pct":        round(pct, 1),
                            "t":          self.ts_to_string(ts)
                        })
                        self.publish("net", "/replay/status/gps", status)
                        loop_start_time = time.time()
                        msg_count = 0

                if self.state == state.quit: return
                if self._config_event.is_set(): break  
                if self._sync_event.is_set():
                    self._sync_event.clear()
                    continue  

                self.sync_received = False
                print("[GPS] End of playback window — waiting for sync to loop...")
                
                while not self.sync_received:
                    if self.state == state.quit: return
                    if self._config_event.is_set(): break
                    time.sleep(0.1)

                if self._config_event.is_set(): break
                self._sync_event.clear()
                
                # FIXED: Forces the "while inner_running" loop to restart playback seamlessly
                continue

def main():     
    parser = argparse.ArgumentParser(description="gps replay sensor app")
    parser.add_argument('--scenario', type=str, default='20241009_120237', help='Scenario file pattern')
    parser.add_argument('--start', type=int, default=0, help='Start time in seconds')
    parser.add_argument('--end', type=int, default=-1, help='End time in seconds')
    args = parser.parse_args()

    service = gps_replay(args)
    service.start()

if __name__ == '__main__':
    main()
