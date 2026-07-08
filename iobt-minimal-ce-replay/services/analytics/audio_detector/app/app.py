#!/usr/bin/python

#Add MCP python library dir to path
import os, sys
sys.path.append("/lib/iobtmax")

from iobt_max_service import iobt_max_service, state

import time
import numpy as np
import queue
import pickle 
import json
from threading import Timer, Lock 

class audio_detector(iobt_max_service):

    def __init__(self):

        #Name the app and init the base class 
        self.name = "audio_detector"
        iobt_max_service.__init__(self,self.name)

        #Create a queue for receving data
        self.q  = queue.Queue()

        #Get the detection threshold from an environment variable
        #This allows the detector to be configured from docker compose
        self.detection_threshold = float(os.environ["DETECTION_THRESHOLD"])

        #Create topic for issuing detections
        self.net_topic = self.get_topic_name("detections")
        print(f"Issuing detections on {self.net_topic}")

        #Subscribe to the local respeaker data provider
        #and specify a callback to get data
        self.subscribe("local", "respeaker", self.get_respeaker_data)

        #Get a lock for concurrent file operations
        self.lock = Lock()
        self.data_file=None

    def get_respeaker_data(self, data):
        #Get data from callback an enqueue
        self.q.put(data)
        time.sleep(0)

    def service_initialize(self):
        #Must be impemented by sensor service
        print("Calling audio-detector service initialize") 

    def service_stop(self):
        #Must be impemented by sensor service
        print("Calling audio-detecrtor service stop") 

    def service_initialize_collect(self):
        #Initialize collect 
        print("Calling audio-detecrtor initialize collect")
        self.data_file_name = self.get_file_name("csv")

        with self.lock:
            self.data_file      = open(self.data_file_name,"w")
            self.data_file.write("Timestamp,Loudness\n")

    def service_stop_collect(self):
        #Stop collect
        print("Calling audio-detecrtor stop collect")
        with self.lock:
            self.data_file.close()
            self.data_file=None

    def service_step(self):
        #Process received data

        #Initialize data statistics
        frame_count = 0
        data_len    = 0
        loudness    = 0 
        num_det     = 0
        last_report_time = time.time()

        #Loop until quit
        while(not self.state == state.quit):

            #Output statistics once per second
            if(time.time()-last_report_time >1.0):
                delta = time.time()-last_report_time
                frame_rate = frame_count / delta
                data_rate  = data_len / delta
                avg_loudness = loudness / (frame_count+1e-10)
                avg_frame_size = data_len / (frame_count+1e-10)
                ts = self.ts_to_string(time.time())

                print(f"[{ts}] Detections: {num_det}/{frame_count} Avg Loudness: {avg_loudness:0.3f}db Frame Rate: {frame_rate:0.3f}/s Data Rate: {data_rate:0.3f}b/s Avg Frame Size: {avg_frame_size:0.1f}b",flush=True)

                frame_count = 0
                data_len    = 0
                loudness    = 0 
                num_det     = 0 
                last_report_time = time.time()

            try:
                data  = self.q.get(timeout=1)
            except queue.Empty:
                continue  
            
            #Respeaker data are transmitted as a pickled python dictionary
            #Load and process the data. 
            topic     = data["topic"]
            payload   = data["payload"]
            waveform  = payload["waveform"].astype(float)
            timestamp = payload["t"]

            #Accumulate statistics for reporting
            frame_count += 1
            rms          = np.sqrt(np.mean(waveform[:,[1,2,3,4]] **2))
            dbs          = 20 * np.log10(1e-16 + rms/32767) 
            det          = dbs>self.detection_threshold
            loudness    += dbs
            data_len    += 2*np.prod(waveform.shape) 
            num_det     += det

            #If we are in the collecting state, write data
            if(self.state == state.collecting and self.data_file is not None):
                ts_string = self.ts_to_string(timestamp)
                line = f"{ts_string},{dbs:0.3f}\n"
                with self.lock:
                    self.data_file.write(line)

            #If computed loudness exceeds threshold, issue a detection
            if(det):
                msg = json.dumps({"t":timestamp, "db": dbs})
                self.publish("net",self.net_topic,msg)

            time.sleep(0)

        return(True)

def main():     
    node = audio_detector()
    node.start()

if __name__ == '__main__':
    main()