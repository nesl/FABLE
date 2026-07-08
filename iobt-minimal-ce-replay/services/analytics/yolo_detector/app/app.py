
import sys, os, io
sys.path.append("/lib/iobtmax")

from iobt_max_service import iobt_max_service, state

import time
import cv2
import numpy as np
import code
import json
from datetime import datetime
from ultralytics import YOLO
import torch
import spatial_transform_utils as st
import pynng
import pickle
import queue
from threading import Lock
from threading import Timer
import base64
import numpy as np

class yolo_detector(iobt_max_service):
    def __init__(self):

        if("SERVICE_NAME" in os.environ):
            name = os.environ["SERVICE_NAME"]
            if(name=="service"): name="yolo"
        else:
            name="yolo"

        iobt_max_service.__init__(self,name)

        if "SOURCE" in os.environ:
            self.source_host = os.environ["SOURCE"]
            if(self.source_host=="local"): self.source_host = self.hostname
        else:
            self.source_host = self.hostname

        self.node_short_name = self.get_node_short_name(self.source_host)
        self.lock=Lock()
        
        #Load environment
        env_path = os.environ.get("ENV_INFO", "environments/gq/env_info.json")
        with open(env_path) as env_file :
            self.info = json.load(env_file)
        
        if(self.node_short_name in self.info["nodes"]):
            self.this_node_info = self.info["nodes"][self.node_short_name]
            self.point_projector = st.point_projector(self.this_node_info)
        else:
            self.this_node_info = None
            self.point_projector = None
            print(f"No environment info found for {self.node_short_name}")

        # Load yolo model

        if("LOAD_MODEL" in os.environ):
            self.load_model = os.environ["LOAD_MODEL"].lower()=="true"
        else:
            self.load_model = True

        if(self.load_model):
            print("Starting model load (this may take 30s)...")
            if torch.cuda.is_available() and os.environ.get("YOLO_DEVICE", "auto") != "cpu":
                torch.cuda.set_device(0)
                self.yolo_device = 0
            else:
                self.yolo_device = "cpu"
            model_path = os.environ.get("YOLO_MODEL", "/app/yolov8n.pt")
            print(f"Loading YOLO model {model_path} on device={self.yolo_device}")
            self.model = YOLO(model_path)  # load a pretrained model
            print(self.model.info())
            print("Running with more classes")
            #self.model.predict(conf=0.1,classes=[0, 2, 7, 24, 26, 28])
            self.model.predict(conf=0.1, device=self.yolo_device)
            print("Finished loading model")

            #Run a test image to init model
            cv_rgb = cv2.imread("bus.jpg")
            self.detect(cv_rgb)
        else:
            print("Warning: Not loading model")

        self.yolo_topic = f"/{self.source_host}/analytics/yolo/bbox"
        self.q = queue.Queue()

        self.softfail=False
        self.remote_work_topics=[]
        self.remote_work_buffer={}

        if(self.source_host == self.hostname):
            #Subscribe to zed output via ipc
            self.subscribe("local", "zed", self.get_local_zed_data)
        else:
            #Subscribe to zed output via MQTT
            self.subscribe("net",f"/{self.source_host}/zed/rgb_left/compressed",self.get_remote_zed_data)
            self.subscribe("net",f"/{self.source_host}/zed/depth/compressed",self.get_remote_zed_data)

        print(f"Running on {self.hostname} with zed data source {self.source_host} and service name {self.servicename}...")

        self.step_frame_count=0
        self.frame_count=0
        self.max_depth=40

        #Timer(10, self.service_control_callback, args=(self.service_control_topic ,"start_softfail")).start()
        #Timer(20, self.service_control_callback, args=(self.service_control_topic ,"stop_softfail")).start()

    def get_local_zed_data(self, data):

        #if in softfail mode, don't process data
        if(self.softfail): return

        #Get data from nng callback and enqueue
        self.q.put(data)
        time.sleep(0)

    def get_remote_zed_data(self,topic,msg)->None:
        #Get data from MQTT callback and enqueue

        #if in softfail mode, don't process data
        if(self.softfail): return

        node = topic.split("/")[1]
        if node not in self.remote_work_buffer:
            self.remote_work_buffer[node]={"rgb":None,"depth":None,"rgb_time":0,"depth_time":0}
        if("rgb" in topic):
            self.remote_work_buffer[node]["rgb"] = base64.b64decode(msg)
            self.remote_work_buffer[node]["rgb_time"] = time.time()
        if("depth" in topic):
            self.remote_work_buffer[node]["depth"] = base64.b64decode(msg) 
            self.remote_work_buffer[node]["depth_time"] = time.time()   

        if(self.remote_work_buffer[node]["depth"] is not None and self.remote_work_buffer[node]["rgb"] is not None ):

            time_delta = np.abs(self.remote_work_buffer[node]["rgb_time"] - self.remote_work_buffer[node]["depth_time"])

            if(time_delta<0.5):
                #print(f"Have RGB-Depth pair with time detla {time_delta}")
                payload={"t":time.time()*1e6, "i":self.remote_work_buffer[node]["rgb"], "d":self.remote_work_buffer[node]["depth"]}
                msg={"topic":"data","node":node,"payload":payload} 
                self.q.put(msg)
                self.remote_work_buffer[node]["depth"] = None
                self.remote_work_buffer[node]["rgb"]   = None

        time.sleep(0)

    def service_initialize(self):
        #Must be impemented by sensor service
        print("Calling yolo-detector service initialize") 

    def service_stop(self):
        #Must be impemented by sensor service
        print("Calling yolo-detecrtor service stop") 

    def service_initialize_collect(self):
        print("Calling yolo-detector init collect")
        self.frame_count=0
        self.dets_file_name = self.get_file_name("json")
        with self.lock:
            self.dets_file      = open(self.dets_file_name,"w") 
            self.dets_file.write("[\n")
        self.collection_initialized=True 

    def service_stop_collect(self):
        print("Calling yolo-detector stop collect")
        self.collection_initialized=False
        with self.lock:
            self.dets_file.write("\n]")
            self.dets_file.close()    

    def service_control_callback(self, topic, msg)->None:

        print(f"Got node service control callback msg: {msg}")

        if(" " in msg):
            parts = msg.split(" ")
            cmd=parts[0]
            arg=parts[1]
        else:
            cmd = msg

        if(cmd=="start_softfail"):
            self.softfail=True
        elif(cmd=="stop_softfail"):
            self.softfail=False
        else:
            print(f"Got unknown control message: {msg}")


    def service_step(self):
        
        self.step_frame_count=0
        self.detection_count=1e-16
        self.start_time = time.time()
        self.obj_count={}
        self.total_latency=0

        while(not self.state == state.quit and time.time()-self.start_time<5):

            try:
                data  = self.q.get(timeout=1)
            except queue.Empty:
                continue  

            self.process_data(data)

            time.sleep(0)

        self.report_stats()

    def process_data(self,data):

        topic   = data["topic"]
        payload = data["payload"]

        buf    = np.frombuffer(payload["i"], dtype=np.uint8)
        cv_rgb = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        buf      = np.frombuffer(payload["d"], dtype=np.uint8)
        cv_depth = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

        t = self.ts_to_string(payload["t"]/1e6)

        if(self.load_model):
            dets   = self.detect(cv_rgb, cv_depth, t)
            if(len(dets)>0):
                dets_payload = json.dumps(dets)
                self.publish("net",self.yolo_topic,dets_payload)
                if(self.state==state.collecting):
                    with self.lock:
                        if(self.frame_count>=1): self.dets_file.write(",\n")
                        self.dets_file.write(dets_payload)
        else:
            dets=[{"node": self.node_short_name, "model": "no_model", "class": "test", "conf": 1.0, "box": [0,0,0,0], "depth": -1.0, "world": [], "t": t}]
            self.publish("net", self.yolo_topic, json.dumps(dets))

        #Keep track of high-level stats
        self.frame_count+=1
        self.step_frame_count+=1
        self.detection_count+=len(dets)
        self.total_latency += time.time() - payload["t"]/1e6
        for det in dets:
            if(det["class"] in self.obj_count):
                self.obj_count[det["class"]] += 1
            else:
                self.obj_count[det["class"]] = 1

    def detect(self,cv_rgb, cv_depth=None, t=0):
        results = self.model(cv_rgb, verbose=False, device=getattr(self, "yolo_device", None))
        dets=[]
        for r in results:
            boxes = r.boxes
            for box in boxes:
                #code.interact(local=locals())
                c = r.names[int(box.cls.item())]
                b = [int(x) for x in box.xywh[0].cpu().numpy()] #bbox in [cx, cy, w, h] format
                p = box.conf.cpu().item()

                if(cv_depth is None): 
                    d=-1.0
                    w=[]
                else:
                    #code.interact(local=locals())                    

                    dp = cv_depth[b[1],b[0]]
                    if(dp<=6):
                        d=-1.0
                        w=[]
                    else:
                        d = round(self.max_depth*(dp/65535.0),3)
                        s = 1080/cv_rgb.shape[0]
                        points_img = torch.tensor([[s*b[0],s*b[1],d]]).float() 
                        if(self.point_projector is not None):
                            with torch.no_grad():
                                points_world = self.point_projector.image_to_world(points_img)
                            w = [round(float(x),3) for x in points_world[0,:]]
                        else:
                            w = [0.0,0.0,0.0]

                dets.append({"node":self.node_short_name,"model":"yolov8", "class":c, "conf":p, "box": b,"depth":d, "world":w,"t":t})    

        return(dets)

    def report_stats(self):
        t          = datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")
        frame_rate = self.step_frame_count/(time.time()-self.start_time)
        det_rate   = self.detection_count/(time.time() - self.start_time)
        class_info = " ".join([f"{c}: {self.obj_count[c]/self.detection_count:.3f}" for c in self.obj_count])
        avg_latency = self.total_latency/(1e-4+self.step_frame_count)

        print(f"[{t}] Input Rate: {frame_rate:.2f}/fps Det Rate: {det_rate:.2f}/s Latency: {avg_latency:.4f}s [{class_info}]")

def main():     
    node = yolo_detector()
    node.start()

if __name__ == '__main__':
    main()

