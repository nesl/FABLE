#!/usr/bin/python
# Add MCP python library dir to path
import os, sys
sys.path.append("/lib/iobtmax")

from iobt_max_service import iobt_max_service, state

import json
import threading
import time
import numpy as np
from datetime import datetime
from scipy.fftpack import fft
import soundfile as sf
import pandas as pd
import argparse


class respeaker_replay(iobt_max_service):

    def __init__(self, args):
        # Name the app and init the base class
        self.name = "respeaker"
        iobt_max_service.__init__(self, self.name)

        # fps, rate, and sample_per_frame are derived from the actual recording
        # in service_initialize — set placeholders here so attributes exist.
        self.fps              = None
        self.rate             = None
        self.sample_per_frame = None
        self.frame_count      = 0
        self.samples          = []
        self.power_samples    = []
        self.max_db           = 0
        self.min_db           = np.inf

        # ReSpeaker hardware constants — kept identical to the live driver so
        # that the published data has the same shape/meaning.
        self.RESPEAKER_RATE     = 16000
        self.RESPEAKER_CHANNELS = 6
        self.RESPEAKER_WIDTH    = 2

        # FFT visualization bins (same as live driver)
        self.fft_bins = [0, 60, 250, 2000, 4000, 6000]

        # Resolve file paths from the scenario argument.
        # Expected naming convention (produced by the live driver via get_file_name):
        #   <scenario>_<hostname>_respeaker.flac
        #   <scenario>_<hostname>_respeaker.csv
        self.flac_file      = f"/output/{args.scenario}_{self.hostname}_respeaker.flac"
        self.timestamp_file = f"/output/{args.scenario}_{self.hostname}_respeaker.csv"

        self.playback_start_time = args.start   # seconds offset into recording
        self.playback_end_time   = args.end     # seconds offset, -1 = full file
        self.loop_wall_start     = time.time()  # overwritten by sync message
        self.sync_start_at       = time.time()  # absolute wall time to begin frame 0
        self.sync_received       = False
        self._sync_event         = threading.Event()
        self._config_event       = threading.Event()
        self.sync_topic          = "/replay/sync"
        self.config_topic        = "/replay/config"
        self._loaded_flac_file   = None   # track what is currently loaded
        self._pending_flac_file  = None   # track what we've been asked to load
        self._loaded_start_time  = None
        self._loaded_end_time    = None
        self._pending_start_time = None
        self._pending_end_time   = None

    # ------------------------------------------------------------------
    # iobt_max_service abstract method implementations
    # ------------------------------------------------------------------

    def service_initialize(self):
        print("Initializing respeaker replay service")

        # MQTT topic names — identical to the live driver so subscribers see
        # exactly the same topics.
        self.net_fft_topic   = self.get_topic_name("fft")
        self.net_power_topic = self.get_topic_name("power")
        self.sync_topic      = "/replay/sync"
        self.config_topic    = "/replay/config"
        self.subscribe("net", self.sync_topic, self._on_sync)
        self.subscribe("net", self.config_topic, self._on_config)

        # Load the FLAC recording into memory as a NumPy array.
        # shape: (total_samples, RESPEAKER_CHANNELS)
        print(f"Loading FLAC file: {self.flac_file}")
        # sf.read() has a known bug with 6-channel FLAC files in older versions
        # of libsndfile — it returns 0 frames despite sf.info() reporting the
        # correct frame count. We work around this by reading in blocks.
        info = sf.info(self.flac_file)
        file_sr = info.samplerate
        if file_sr != self.RESPEAKER_RATE:
            print(f"Warning: file sample rate {file_sr} != expected {self.RESPEAKER_RATE}")

        print(f"File reports {info.frames} frames, {info.channels} channels — reading in blocks...")
        BLOCK_SIZE = 16000  # 1 second of audio per block
        blocks = []
        with sf.SoundFile(self.flac_file) as f:
            while True:
                block = f.read(BLOCK_SIZE, dtype='float32', always_2d=True)
                if len(block) == 0:
                    break
                blocks.append(block)
        
        if blocks:
            audio_float = np.vstack(blocks)
        else:
            audio_float = np.zeros((0, info.channels), dtype=np.float32)

        # Convert float32 [-1.0, 1.0] to int16 range to match live driver format
        self.audio_data = (audio_float * 32767).astype(np.int16)
        print(f"Loaded audio: {self.audio_data.shape[0]} samples, {self.audio_data.shape[1]} channels")

        if self.audio_data.shape[1] < self.RESPEAKER_CHANNELS:
            # Pad channels with zeros if fewer channels than expected
            print(f"Warning: file has {self.audio_data.shape[1]} channels, expected {self.RESPEAKER_CHANNELS}. Padding with zeros.")
            pad = np.zeros((self.audio_data.shape[0], self.RESPEAKER_CHANNELS - self.audio_data.shape[1]), dtype=np.int16)
            self.audio_data = np.hstack([self.audio_data, pad])

        # Load and parse the per-frame timestamp CSV produced by the live driver.
        # Columns: Frame, Timestamp
        print(f"Loading timestamp CSV: {self.timestamp_file}")
        self.df_timestamps = pd.read_csv(self.timestamp_file)
        self.df_timestamps["Timestamp"] = pd.to_datetime(self.df_timestamps["Timestamp"])
        # Convert to elapsed seconds from the first frame
        t0 = self.df_timestamps["Timestamp"].iloc[0]
        self.df_timestamps["elapsed"] = (
            self.df_timestamps["Timestamp"] - t0
        ).dt.total_seconds()

        total_frames = len(self.df_timestamps)
        total_duration = self.df_timestamps["elapsed"].iloc[-1]

        # Clamp playback window to valid range
        if self.playback_end_time == -1 or self.playback_end_time > total_duration:
            self.playback_end_time = total_duration

        self.playback_start_frame = int(
            self.df_timestamps["elapsed"].searchsorted(self.playback_start_time)
        )
        self.playback_end_frame = int(
            self.df_timestamps["elapsed"].searchsorted(self.playback_end_time)
        )
        self.total_playback_frames = self.playback_end_frame - self.playback_start_frame

        total_audio_samples   = self.audio_data.shape[0]

        # Derive sample_per_frame, fps, and rate from the actual recording.
        # This avoids assuming a fixed 10fps capture rate.
        self.sample_per_frame = total_audio_samples // total_frames
        self.fps              = round(self.RESPEAKER_RATE / self.sample_per_frame)
        self.rate             = self.sample_per_frame / self.RESPEAKER_RATE  # seconds per frame

        expected_samples     = total_frames * self.sample_per_frame
        playback_samples     = self.total_playback_frames * self.sample_per_frame

        print(f"Total recording duration : {total_duration:.2f}s  ({total_frames} frames)")
        print(f"Derived fps              : {self.fps:.2f}  sample_per_frame={self.sample_per_frame}")
        print(f"Audio array shape        : {self.audio_data.shape}  (expected ~{expected_samples} samples for {total_frames} frames)")
        print(f"Playback window          : {self.playback_start_time:.2f}s -> {self.playback_end_time:.2f}s")
        print(f"Playback frames          : {self.playback_start_frame} -> {self.playback_end_frame}  ({self.total_playback_frames} frames, ~{playback_samples} samples)")
        print(f"Audio samples available  : {total_audio_samples}  ({'OK' if total_audio_samples >= playback_samples else 'WARNING: insufficient'})")
        print("Done initializing respeaker replay service")
        self._loaded_flac_file   = self.flac_file
        self._loaded_start_time  = self.playback_start_time
        self._loaded_end_time    = self.playback_end_time
        self._pending_flac_file  = self.flac_file
        self._pending_start_time = self.playback_start_time
        self._pending_end_time   = self.playback_end_time

    def _on_config(self, topic, payload):
        """MQTT callback for scenario config messages.
        Payload: {"scenario": "20250815_083306", "start_time": "0", "end_time": "-1"}
        Updates file paths and playback window.
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
                print("[ReSpeaker] Config message has blank scenario — keeping current files.")
                new_flac = self._loaded_flac_file if self._loaded_flac_file else self.flac_file
                new_csv  = self.timestamp_file
            else:
                new_flac = f"/output/{scenario}_{self.hostname}_respeaker.flac"
                new_csv  = f"/output/{scenario}_{self.hostname}_respeaker.csv"

            self.flac_file           = new_flac
            self.timestamp_file      = new_csv
            self.playback_start_time = new_start
            self.playback_end_time   = new_end

            print(f"[ReSpeaker] Config updated — scenario='{scenario}' start={new_start} end={new_end}")

            # Trigger reload only if this config differs from what is already loaded OR pending.
            loaded_match  = (new_flac == self._loaded_flac_file 
                             and new_start == self._loaded_start_time 
                             and new_end   == self._loaded_end_time)
            pending_match = (new_flac == self._pending_flac_file 
                             and new_start == self._pending_start_time 
                             and new_end   == self._pending_end_time)

            if not loaded_match and not pending_match:
                self._pending_flac_file  = new_flac
                self._pending_start_time = new_start
                self._pending_end_time   = new_end
                self._config_event.set()
            else:
                print(f"[ReSpeaker] Config is same as loaded/pending — ignoring duplicate.")

        except Exception as e:
            print(f"[ReSpeaker] Config parse error: {e}")


    def _on_sync(self, topic, payload):
        """MQTT callback for playback controller sync messages.
        The sync message carries an absolute 'start_at' wall time at which all
        nodes should begin playing frame 0. This eliminates drift caused by
        different file-load times on different nodes.
        """
        try:
            msg = json.loads(payload)
            self.sync_start_at   = float(msg.get('start_at', time.time()))
            self.loop_wall_start = self.sync_start_at
            self.sync_received   = True
            self._sync_event.set()
            print(f"[ReSpeaker] Sync received — start_at={self.sync_start_at:.3f} "
                  f"(in {self.sync_start_at - time.time():.2f}s)")
        except Exception as e:
            print(f"[ReSpeaker] Sync parse error: {e}")

    def service_initialize_collect(self):
        print("Calling respeaker replay service collection init")

    def service_stop_collect(self):
        print("Calling respeaker replay service stop collection")

    def service_stop(self):
        print("Stopping respeaker replay service")

    # ------------------------------------------------------------------
    # Main replay loop
    # ------------------------------------------------------------------

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
        State machine:
          1. If no data loaded, wait for config then load.
          2. Wait for sync from the playback controller.
          3. Play frames until end of window, sync interrupt, or config interrupt.
          4. At end of window: wait for next sync (same as step 2).
          5. On config: reload data, then wait for sync (same as step 2).
          Sync always wins — it resets loop_wall_start and starts playback.
        """
        service_step_start = time.time()
        step_frame_count   = 0
        data_loaded        = False
        sync_ready         = False  # True when we have a valid loop_wall_start

        while True:  # top-level state machine
            if self.state == state.quit:
                return

            # ── LOAD DATA ────────────────────────────────────────────────────
            if self._config_event.is_set() or not data_loaded:
                self._config_event.clear()
                self._sync_event.clear()
                self.sync_received = False
                print("[ReSpeaker] Loading data...")
                try:
                    self.service_initialize()
                    data_loaded = True
                except FileNotFoundError as e:
                    print(f"[ReSpeaker] File not found: {e} — waiting for new config...")
                    data_loaded = False
                    # Drain any queued config/sync events and wait for a new config
                    self._config_event.clear()
                    self._sync_event.clear()
                    while not self._config_event.is_set():
                        if self.state == state.quit:
                            return
                        time.sleep(0.1)
                    continue
                except Exception as e:
                    print(f"[ReSpeaker] Init failed: {e}")
                    time.sleep(2)
                    continue
                sync_ready = False  # new data loaded, need a fresh sync

            # ── WAIT FOR SYNC ─────────────────────────────────────────────────
            # Skip if we already have a valid sync (e.g. interrupted mid-playback)
            if not sync_ready:
                if self._sync_event.is_set():
                    print(f"[ReSpeaker] Sync already received — starting playback immediately.")
                    self._sync_event.clear()
                else:
                    print(f"[ReSpeaker] Waiting for sync on {self.sync_topic}...")
                    result = self._wait_for_sync_or_config()
                    if result == 'quit':
                        return
                    if result == 'config':
                        continue  # reload data at top of loop
                    # result == 'sync' — fall through to playback
            sync_ready = False  # consume the sync; next iteration must wait again

            # ── PLAY ──────────────────────────────────────────────────────────
            self.frame_count = 0
            loop_duration = (
                self.df_timestamps["elapsed"].iloc[self.playback_end_frame - 1]
                - self.df_timestamps["elapsed"].iloc[self.playback_start_frame]
            )
            if loop_duration <= 0:
                print("[ReSpeaker] Zero loop duration — waiting for config...")
                self._config_event.clear()
                while not self._config_event.is_set():
                    if self.state == state.quit:
                        return
                    time.sleep(0.1)
                continue

            # Wait until start_at if it's in the future
            now = time.time()
            if self.sync_start_at > now:
                wait = self.sync_start_at - now
                print(f"[ReSpeaker] Waiting {wait:.2f}s until synchronized start_at={self.sync_start_at:.3f}")
                elapsed_wait = 0
                while elapsed_wait < wait:
                    if self.state == state.quit: return
                    if self._sync_event.is_set(): break
                    if self._config_event.is_set(): break
                    time.sleep(min(0.05, wait - elapsed_wait))
                    elapsed_wait = time.time() - now
                if self._sync_event.is_set() or self._config_event.is_set():
                    sync_ready = self._sync_event.is_set()
                    if self._sync_event.is_set(): self._sync_event.clear()
                    continue

            # Compute starting frame based on absolute start_at reference.
            # Use modulo so that if start_at is in the past by more than
            # loop_duration (e.g. file load took longer than SYNC_LEAD_S),
            # we still land at the correct position within the window.
            wall_elapsed   = max(0.0, time.time() - self.sync_start_at) % loop_duration
            target_elapsed = self.df_timestamps["elapsed"].iloc[self.playback_start_frame] + wall_elapsed
            current_frame  = int(self.df_timestamps["elapsed"].searchsorted(target_elapsed))
            current_frame  = np.clip(current_frame, self.playback_start_frame, self.playback_end_frame - 1)

            print(f"[ReSpeaker] Starting playback sync_start_at={self.sync_start_at:.3f}")
            print(f"Replay cycle elapsed : {wall_elapsed:.2f}/{loop_duration:.2f}s")
            print(f"Current frame        : {current_frame}/{self.playback_end_frame}")

            interrupted_by_sync   = False
            interrupted_by_config = False

            for frame_idx in range(current_frame, self.playback_end_frame):
                frame_start = time.perf_counter()

                if self.state == state.quit:
                    return

                if self._sync_event.is_set():
                    self._sync_event.clear()
                    interrupted_by_sync = True
                    print("[ReSpeaker] Sync received during playback — restarting.")
                    break
                if self._config_event.is_set():
                    interrupted_by_config = True
                    print("[ReSpeaker] Config received during playback — reinitializing.")
                    break

                relative_frame = frame_idx - self.playback_start_frame
                sample_start   = relative_frame * self.sample_per_frame
                sample_end     = sample_start + self.sample_per_frame

                if sample_end > self.audio_data.shape[0]:
                    print("Reached end of audio data.")
                    break

                chunk     = self.audio_data[sample_start:sample_end, :]
                timestamp = time.time()

                if self.state == state.collecting:
                    self.record_frames.append(chunk.copy())
                    if not self.frame_log_file.closed:
                        self.frame_log_file.write(f"{self.frame_count},{self.ts_to_string(timestamp)}\n")

                self.publish("local", "rawaudio", {"t": timestamp, "waveform": chunk})
                self.samples.append(chunk[:, 1])
                self.power_samples.append(chunk[:, 1:5].mean(axis=1))

                if self.frame_count % self.fps == 0 and self.frame_count > 0:
                    self._send_vis()
                if (5 * self.frame_count) % self.fps == 0 and self.frame_count > 0:
                    self._send_power()

                step_frame_count += 1
                self.frame_count  += 1

                wall_now = time.time()
                if wall_now - service_step_start >= 5:
                    t = datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")
                    print(f"[{t}] State: {self.state}  Frame rate: {step_frame_count/(wall_now-service_step_start):.2f}/s")
                    current_elapsed = max(0.0, wall_now - self.sync_start_at) % max(loop_duration, 1e-6)
                    pct = 100.0 * current_elapsed / max(loop_duration, 1e-6)
                    status_msg = json.dumps({
                        "service":    "respeaker_replay",
                        "node":       self.hostname,
                        "start_time": self.playback_start_time,
                        "end_time":   self.playback_end_time,
                        "current":    round(current_elapsed, 2),
                        "duration":   round(loop_duration, 2),
                        "pct":        round(pct, 1),
                        "t":          self.ts_to_string(wall_now)
                    })
                    self.publish("net", f"/replay/status/respeaker/{self.hostname}", status_msg)
                    service_step_start = time.time()
                    step_frame_count   = 0

                frame_elapsed = time.perf_counter() - frame_start
                sleep_time = self.rate - frame_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # ── END OF PLAYBACK WINDOW ────────────────────────────────────────
            if interrupted_by_config:
                sync_ready = False
                continue  # reload data at top of loop
            elif interrupted_by_sync:
                sync_ready = True  # sync already captured in loop_wall_start
                continue  # skip WAIT SYNC, go straight to PLAY
            else:
                # Natural end — wait for next sync from playback controller
                print("[ReSpeaker] End of playback window — waiting for sync to loop...")
                self.sync_received = False
                result = self._wait_for_sync_or_config()
                if result == 'quit':
                    return
                # Both 'sync' and 'config' are handled at top of loop
                sync_ready = (result == 'sync')
                continue

    # ------------------------------------------------------------------
    # Publishing helpers — identical output format to the live driver
    # ------------------------------------------------------------------

    def _send_vis(self):
        """
        Compute and publish FFT visualization message.
        Matches send_vis() in the live respeaker driver exactly.
        Direction is published as 0.0 since no Tuning USB device is available.
        """
        direction = 0.0   # no physical device; publish neutral value
        sample = np.hstack(self.samples).astype(np.float32)
        f = fft(sample)
        f = np.log(1 + np.abs(f[:len(f) // 2]))

        eq = np.zeros((len(self.fft_bins) - 1,))
        for i in range(len(self.fft_bins) - 1):
            eq[i] = np.mean(f[self.fft_bins[i]:self.fft_bins[i + 1]])

        self.min_db = min(np.min(eq), self.min_db)
        self.max_db = max(np.max(eq), self.max_db)

        # FIX: Use .tolist() instead of list() to convert numpy scalars to native Python ints
        eq_list = (255 * (eq - self.min_db) / (self.max_db + 1e-12)).astype(np.uint8).tolist()

        # Safely convert to a JSON payload
        payload = '{{"fft":{fft}, "dir":{dir:.1f}}}'.format(fft=str(eq_list), dir=direction)
        
        self.publish("net", self.net_fft_topic, payload)
        self.samples = []


    def _send_power(self):
        """
        Compute and publish RMS power message.
        Matches send_power() in the live respeaker driver exactly.
        """
        #print("sending power")
        direction    = 0.0   # no physical device; publish neutral value

        power_sample = np.hstack(self.power_samples[-2:]).astype(np.float32)
        power_sample -= np.mean(power_sample)
        power        = float(np.mean(power_sample ** 2))

        payload = '{{"power":{power:.3f}, "dir":{dir:.1f}}}'.format(power=power, dir=direction)
        #print(payload)
        self.publish("net", self.net_power_topic, payload)

        self.power_samples = []


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ReSpeaker replay service")
    parser.add_argument(
        '--scenario', type=str, default='20241009_120237',
        help='Scenario prefix used to locate the .flac and .csv files '
             '(e.g. 20241009_120237 → /output/20241009_120237_<hostname>_respeaker.flac)'
    )
    parser.add_argument(
        '--start', type=float, default=0.0,
        help='Start offset in seconds within the recording (default: 0)'
    )
    parser.add_argument(
        '--end', type=float, default=-1.0,
        help='End offset in seconds within the recording (-1 = full file, default: -1)'
    )
    args = parser.parse_args()

    service = respeaker_replay(args)
    service.start()


if __name__ == '__main__':
    main()