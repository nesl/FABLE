
import os
import time
import json
from datetime import datetime, timedelta
import socket
import os, sys
import signal
import sched
from threading import Lock, Timer, Thread
import numpy as np
import paho.mqtt.client as mqtt
import pynng
import pickle
import msgpack
from enum import Enum

from abc import ABC, abstractmethod
from typing import Type, Any, Union, Optional, Callable, Dict, Literal, final
import types

print("Using iob_max_service.py v0.6-readiness")

class state(Enum):
    """
    This class is a enum containing the different states that the iobt_max_service 
    class can be in.
    """
    uninitialized = "uninitialized"
    initialized   = "initialized"
    monitoring    = "monitoring"
    collecting    = "collecting"
    quit          = "quit"


def str_to_bool(s):
    return s.lower() in ['true', '1', 't', 'y', 'yes']

env={
    "SERIALIZER":{"val":"msgpack","cast":str},
    "TEST_CONTROL":{"val":False,"cast": str_to_bool},
    "MCP_CONTAINER_OUTPUT_DIR":{"val":"/output","cast":str},
    "MQTT_HOST_IP":{"val":"localhost","cast":str},
    "MQTT_PORT":{"val":1883,"cast":int},
    "FLASK_HOST_IP":{"val":"localhost","cast":str},
    "FLASK_PORT":{"val":5001,"cast":int},
    "MCP_NODE_NAME":{"val":socket.gethostname(), "cast":str},
    # Optional evaluation/debug logging. This is useful when replay services are
    # running behind NetWaggle because Docker logs otherwise only show service
    # health, not every MQTT/local publish. Kept opt-in so normal runs are quiet.
    "IOBT_LOG_NET_PUBLISH":{"val":False,"cast": str_to_bool},
    "IOBT_LOG_LOCAL_PUBLISH":{"val":False,"cast": str_to_bool},
    "IOBT_LOG_NET_PUBLISH_EVERY_N":{"val":1,"cast": int},
    "IOBT_LOG_LOCAL_PUBLISH_EVERY_N":{"val":30,"cast": int},
    "IOBT_PUBLISH_READINESS":{"val":True,"cast": str_to_bool},
    "IOBT_READINESS_RETAIN":{"val":True,"cast": str_to_bool},
}

for var, data in env.items():
    val = data["val"]
    cast = data["cast"]
    if var in os.environ:
        #cast environment variable to type of default value
        env[var]["val"] = cast(os.environ[var])
        print(f"[IoBT-MAX Service Base]: Setting variable {var}={env[var]['val']} with type {type(env[var]['val'])}.")
    else:
        print(f"[IoBT-MAX Service Base]: Warning - environment variable {var} not set. Using default value {val}.")


class iobt_max_service(ABC):
    """
    This is a base class for defining node services. User classes that inherit from this
        class must define the following methods:
        
        * service_initialize
        * service_stop
        * service_initialize_collect
        * service_stop_collect
        * service_step

    """

    def __init__(self, 
                 servicename:str,
                 serializer:str=env["SERIALIZER"]["val"],
                 test_control:bool=env["TEST_CONTROL"]["val"],
                 output_dir:str=env["MCP_CONTAINER_OUTPUT_DIR"]["val"],
                 mqtt_ip:str=env["MQTT_HOST_IP"]["val"],
                 mqtt_port:int=env["MQTT_PORT"]["val"]) -> None:
        """_summary_

        Parameters
        ----------
        service_name : str
            Name for the service
        serializer : str, optional
            Serializer to use when sending local messages. Must be "pickle" or "msgpack"
        test_control : bool, optional
            Whether to start in control testing mode. When True, the service will execute
            the sequence of state monitoring->collecting->monitoring->quit. When False 
            the service will start normally.
        output_dir : str, optional
            Directory for saving output data files, by default /output
        mqtt_ip : str, optional
            mqtt server ip address or domain name, by default localhost
        mqtt_port : int, optional
            mqtt server port, by default 1883
        """

        self.hostname              = self.get_hostname()
        self.servicename           = servicename
        self.scheduler             = sched.scheduler(time.time, time.sleep)
        self.collection_start_time = None
        self.rate                  = 1
        self.state                 = state.uninitialized
        self.run_thread            = None

        self.output_dir=output_dir

        #Initialize MQTT for off-node communication
        self.host = mqtt_ip
        self.port = mqtt_port
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        self.mqtt_client.on_connect_fail = self._on_mqtt_connect_fail
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect 
        self.mqtt_client.connect(self.host, self.port, 60)
        self.mqtt_client.loop_start()
        self.mqtt_subscriber_callbacks ={}
        self.last_net_pub_time=0
        self.service_control_topic = self.get_topic_name("control")

        self.log_net_publish = env["IOBT_LOG_NET_PUBLISH"]["val"]
        self.log_local_publish = env["IOBT_LOG_LOCAL_PUBLISH"]["val"]
        self.log_net_publish_every_n = max(1, int(env["IOBT_LOG_NET_PUBLISH_EVERY_N"]["val"]))
        self.log_local_publish_every_n = max(1, int(env["IOBT_LOG_LOCAL_PUBLISH_EVERY_N"]["val"]))
        self._publish_log_counts = {}
        self.publish_readiness_enabled = env["IOBT_PUBLISH_READINESS"]["val"]
        self.readiness_retain = env["IOBT_READINESS_RETAIN"]["val"]

        #Initialize pynng for on-node inter-container coms
        self.nng_subscribers = {}
        self.nng_subscriber_callbacks ={}
        self.nng_subscriber_threads = {}

        try:
            self.nng_pub = pynng.Pub0()
            self.nng_addr = f"ipc:///tmp/{self.servicename}.ipc"
            self.nng_pub.listen(self.nng_addr)
            print(f"Created local publisher with address {self.nng_addr}")
        except pynng.exceptions.NotSupported:
            print("Local publisher is not availble. NNG not supported.")

        self.collect_status_topic = self.get_topic_name("collect_status")

        self.serializer=serializer
        print(f"Using serializer {self.serializer}")

        if(test_control):
            print("WARNING: TEST_CONTROL mode is on")
            Timer(10, self._controlCallback,args=("control","collection-start")).start()
            Timer(20, self._controlCallback,args=("control","collection-stop")).start()
            Timer(30, self._controlCallback,args=("control","collection-start")).start()
            Timer(40, self._controlCallback,args=("control","collection-shutdown")).start()

    @final
    def ts_to_string(self,ts:float)->str:
        """
        Convert timestamp is seconds to canonical time string

        Parameters
        ----------
        ts : _type_
            Timestamp in seconds as returned by time.time()

        Returns
        -------
        str
            Timestamp in canonical string representation using the format "%Y-%m-%d %H:%M:%S.%f"
        """
        #
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")

    @final
    def _on_mqtt_connect(self, client:mqtt.Client, userdata:Any, flags: Dict[str, Any], rc: mqtt.ReasonCodes)->None:
        """
        MQTT callback for when the client receives a CONNACK response from the server.
        This indicates that the client is connected and can proceed to subscribe to 
        topics.

        Parameters
        ----------
        client : mqtt.Client
            MQTT client
        userdata : Any
            MQTT CONNACK user user data
        flags : Dict[str, Any]
            MQTT CONNACK flags
        rc : mqtt.reasoncodes.ReasonCode
             MQTT CONNACK reason code value
        """
        print("MQTT client connected with reason code "+str(rc))
        client.subscribe("control")
        client.subscribe(self.service_control_topic)
        self.mqtt_subscriber_callbacks["control"]=self._controlCallback
        self.mqtt_subscriber_callbacks[self.service_control_topic]=self.service_control_callback

    @final
    def _on_mqtt_message(self, client:mqtt.Client, userdata:Any, msg:mqtt.MQTTMessage)->None:
        """
        MQTT callback for when a message is received by the client.

        Parameters
        ----------
        client : mqtt.Client
            MQTT client object
        userdata : Any
            MQTT userdata object
        msg : mqtt.MQTTMessage
            MQTT message object. Contains fields "topic" and "payload"
        """
        #print(f"Got MQTT Message on topic {msg.topic}")

        topic = msg.topic
        body  = msg.payload.decode()

        if(topic in self.mqtt_subscriber_callbacks):
            self.mqtt_subscriber_callbacks[topic](topic,body)

    def service_control_callback(self, topic, msg)->None:
        """
        This function is called when a message is received on the 
        self.service_control_topic, which is specific to an individual
        node and service. This callback can be overridden to implement 
        service-specific control logic. 
        """ 
        print(f"[Warning] Unhandeled node service control message: {msg}")
        pass 


    @final
    def _on_mqtt_connect_fail(self,client:mqtt.Client)->None:
        """
        MQTT connection failed callback.

        Parameters
        ----------
        client : mqtt.Client
            MQTT client object
        """
        print("MQTT Connect Failed. Retrying...")

    @final
    def _on_mqtt_disconnect(self,client:mqtt.Client, userdata:Any, rc: mqtt.ReasonCodes)->None:
        """
        MQTT callback for when the client is disconnected from the server. 

        Parameters
        ----------
        client : mqtt.Client
            MQTT client
        userdata : Any
            MQTT user data
        rc : mqtt.reasoncodes.ReasonCode
            MQTT result code value
        """
        print("MQTT disconnected. Result code: "+str(rc))

    @final
    def _payload_size_bytes(self, payload:Any)->int:
        try:
            if payload is None:
                return 0
            if isinstance(payload, bytes):
                return len(payload)
            if isinstance(payload, bytearray):
                return len(payload)
            if isinstance(payload, str):
                return len(payload.encode("utf-8", errors="replace"))
            return len(str(payload).encode("utf-8", errors="replace"))
        except Exception:
            return -1

    @final
    def _log_publish(self, scope:str, topic:str, payload:Any, every_n:int)->None:
        key = (scope, topic)
        count = self._publish_log_counts.get(key, 0) + 1
        self._publish_log_counts[key] = count
        if count == 1 or count % every_n == 0:
            size = self._payload_size_bytes(payload)
            print(
                f"[publish:{scope}] service={self.servicename} host={self.hostname} "
                f"topic={topic} count={count} bytes={size}",
                flush=True,
            )


    @final
    def publish_readiness(self, service_name:str=None, ready:bool=False, reason:str="", **extra:Any)->None:
        """Publish a retained readiness message for replay orchestration.

        Readiness messages are intentionally separate from debug/status streams so
        the web UI and replay_control.py can decide whether it is safe to publish
        /replay/sync.  The canonical topic is /readiness/<node>/<service>.
        """
        if not getattr(self, "publish_readiness_enabled", True):
            return
        service = str(service_name or self.servicename)
        payload = {
            "kind": "readiness",
            "node": self.hostname,
            "service": service,
            "ready": bool(ready),
            "reason": str(reason or ""),
            "state": getattr(getattr(self, "state", None), "value", str(getattr(self, "state", ""))),
            "pid": os.getpid(),
            "t": time.time(),
        }
        payload.update(extra)
        topic = f"/readiness/{self.hostname}/{service}"
        try:
            wire = json.dumps(payload, default=str)
            result = self.mqtt_client.publish(topic, wire, qos=1, retain=self.readiness_retain)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                print(f"[readiness] publish failed topic={topic} rc={result.rc}", flush=True)
            elif self.log_net_publish:
                self._log_publish("net", topic, wire, 1)
        except Exception as exc:
            print(f"[readiness] publish failed topic={topic}: {exc}", flush=True)

    @final
    def publish(self, scope:Literal["net","local"], topic:str, payload:Union[str,Dict[str, Any]])->None:
        """
        This function publishes a message with the given payload to the specified 
        topic and scope. The scope must be "net" or "local." The topic can be any string.
        The payload must be of type str for scope=net and can be any serializable object
        when the scope is "local".

        Parameters
        ----------
        scope : Literal["net","local"]
            The scope for message distribution. Either "net" or "local." The "net" scope
            will publish a message to the network. The "local" scope will publish the message
            to the local node only.
        topic : str
            The message topic. Can be any string, but it is recommended to construct
            the topic using 
        payload : Union[str,Dict[str, Any]]
            The body of the message. Should be either a string or a dictionary with string keys.
            If the payload is a dictionary, the values need to be serializable using the chosen
            serialization method. 

        Raises
        ------
        ValueError
            If serializer is not defined
        """
        if(scope=="net"):
            if(self.mqtt_client.is_connected()):
                self.mqtt_client.publish(topic,payload)
                if self.log_net_publish:
                    self._log_publish("net", topic, payload, self.log_net_publish_every_n)
                if (topic!=self.collect_status_topic):
                    self.last_net_pub_time=time.time()
            else:
                print(f"The mqtt client is not connected; cannot publish topic={topic}", flush=True)
        elif(scope=="local"):
            try:
                if self.serializer=="pickle":
                    data = pickle.dumps({"topic":topic,"payload":payload})
                elif(self.serializer=="msgpack"):
                    new_payload = self._denumpy_msg(payload)
                    data= msgpack.packb({"topic":topic,"payload":new_payload})
                else:
                    raise ValueError(f"Serializer {self.serializer} is not known")

                self.nng_pub.send(data)
                if self.log_local_publish:
                    self._log_publish("local", topic, data, self.log_local_publish_every_n)
            except pynng.exceptions.Closed:
                print("The nng publisher is already closed")
        else:
            raise ValueError(f"Publishing scope {scope} is not recognized.")

    @final
    def _denumpy_msg(self,payload:Dict[str, Any]) -> Dict[str, Any]:
        """
        Looks for numpy arrays in the values of the input dictionary and replaces them 
        with a cross-platform representation based on the shape and type of the numpy 
        array followed by a packed byte representation of the values.

        Parameters
        ----------
        payload : Dict[str, Any]
            A message payload dictionary.

        Returns
        -------
        Dict[str, Any]
            A message payload dictionary where values that are numpy arrays have 
            been replaced with a cross-platform packed byte representation with
            shape and type meta data.

        """
        new_msg={}
        for k,v in payload.items():
            v = payload[k]
            if(type(v)!=np.ndarray):
                new_msg[k]=v
            else:
                new_data = v.tobytes()
                new_msg[f"numpy_{k}"]={"shape": list(v.shape), "data":new_data, "dtype":v.dtype.name}
        return(new_msg)

    @final
    def _renumpy_msg(self,payload:Dict[str,Any])->Dict[str,Any]:
        """
        Inspect a message payload for values that correspond to a cross-platform packed 
        byte representation of a numpy array and convert them back to a regular numpy
        array.

        Parameters
        ----------
        payload : Dict[str,Any]
            A message payload dictionary.

        Returns
        -------
        Dict[str,Any]
            The input message payload dictionary with packed numpy byte array values
            converted back to regular numpy arrays.
        """
        new_msg={}
        for k,v in payload.items():
            v = payload[k]
            if("numpy" not in k):
                new_msg[k]=v
            else:
                new_k = k.split("_")[1]
                shape = v["shape"]
                dtype_str  = v["dtype"]
                dtype = getattr(np, dtype_str, None)
                bytes = v["data"]
                arr = np.frombuffer(bytes, dtype=dtype).reshape(shape)
                new_msg[new_k]=arr
        return(new_msg)

    @final
    def subscribe(self, scope: Literal["net","local"], topic:str, callback:Callable[[dict], None] )->None:
        """
        This function registers a subscriber to a topic. The scope must either be local or net. 
        The topic can be any message stream available within the scope. The callback function
        must accept a dictionary. Note that this function launches a thread to listen for 
        messages. The specified callback will be called from this thread. The callback must be
        thread safe. It is recommended to do as little work as possible within the callback such as
        pushing the incoming message to a thread safe queue to be processed in a separate thread.

        Parameters
        ----------
        scope : Literal[&quot;net&quot;,&quot;local&quot;]
            The scope in which to register the subscriber
        topic : str
            The topic to subscribe to
        callback : Callable[[dict], None]
            The callback function that will be executed when a message on the specified
            topic and in the specified scope is received. 

        Raises
        ------
        ValueError
            If the specified scope is not valid.
        """
        if(scope=="net"):
            self.mqtt_client.subscribe(topic)
            self.mqtt_subscriber_callbacks[topic]=callback
        elif(scope=="local"):

            addr = f"ipc:///tmp/{topic}.ipc"
            print(f"Attempting to connect to local publisher {addr}")

            # In replay mode, analytics containers often start before the replay
            # child has created /tmp/<service>.ipc. The original implementation
            # tried once and then permanently gave up. Retry in a lightweight
            # background thread so persistent detector containers can stay up
            # while scenarios are selected later from the web UI.
            def _connect_local_with_retry():
                last_error_time = 0
                while self.state != state.quit:
                    if topic in self.nng_subscribers:
                        return
                    try:
                        sub = pynng.Sub0(dial=addr, recv_timeout=1000)
                        sub.subscribe(b"")
                        self.nng_subscribers[topic] = sub
                        self.nng_subscriber_callbacks[topic] = callback

                        listener = Thread(target=self._nng_topic_listener, args=(topic,))
                        listener.daemon = True
                        listener.start()
                        self.nng_subscriber_threads[topic] = listener
                        print(f"Subscribed to local topic: {topic}", flush=True)
                        hook = getattr(self, "on_local_subscription_ready", None)
                        if callable(hook):
                            try:
                                hook(topic, addr)
                            except Exception as exc:
                                print(f"Local subscription readiness hook failed for {topic}: {exc}", flush=True)
                        return
                    except Exception as e:
                        now = time.time()
                        if now - last_error_time > 5:
                            print(f"Local publisher {addr} not ready yet; retrying. Last error: {e}")
                            last_error_time = now
                        time.sleep(1)

            connector = Thread(target=_connect_local_with_retry)
            connector.daemon = True
            connector.start()


    @final
    def unsubscribe(self, scope: Literal["net","local"], topic:str)->None:

        if(scope=="net"):
            if(topic in self.mqtt_subscriber_callbacks):
                self.mqtt_client.unsubscribe(topic)
                del self.mqtt_subscriber_callbacks[topic]
        elif(scope=="local"):
            if(topic in self.nng_subscribers):
                self.nng_subscribers[topic].close()
                del self.nng_subscribers[topic]

    @final
    def _nng_topic_listener(self, topic:str)->None:
        """
        Starts a local nng subscriber loop listening for the specified topic.
        The subscriber and callback must have been registered previously via
        a call to subscribe().

        Parameters
        ----------
        topic : str
            the topic to listen for messages from

        Raises
        ------
        ValueError
            If the serializer is unknown
        """
        print(f"Starting local subscriber listener thread for {topic}")
        while(self.state != state.quit):
            try:

                if(topic not in self.nng_subscribers):
                    return

                data = self.nng_subscribers[topic].recv()

                if self.serializer=="pickle":
                    msg = pickle.loads(data)
                elif(self.serializer=="msgpack"):
                    msg = msgpack.unpackb(data)
                    msg["payload"] = self._renumpy_msg(msg["payload"])
                else:
                    raise ValueError(f"Serializer {self.serializer} is not known")

                self.nng_subscriber_callbacks[topic](msg)
                
            except pynng.exceptions.Timeout:
                pass
            except pynng.exceptions.Closed:
                print(f"The subscription to local topic {topic} was closed by the provider.")
                return

            time.sleep(0)

    @final
    def _controlCallback(self, topic:str, msg:str)->None:
        """
        This function is the control callback for IoBT-Max control plane messages.
        It is called when the net scope subscriber for the control topic receives
        a control message. Depending on the control message, different calls are
        made to update the state of a running service. The msg value passed to this
        function is a control message string.

        Parameters
        ----------
        msg : str
            The control message string. 
        """
        print("Got iobt-max control message " +str(msg))

        if(msg == "collection-start"):
            self._change_state(state.collecting)

        elif(msg == "collection-stop"):
            self._change_state(state.monitoring)

        elif(msg == "collection-shutdown"):
            self._change_state(state.quit)

        else:
            print("WARNING: Got unknown control message: %s"%msg)

    @final
    def _sigint_handler(self,sig:int,frame:Optional[types.FrameType])->None:
        print("Caught ctrl+c. Stopping node.")
        self._change_state(state.quit)  

    @final
    def _change_state(self, desired_state:state):
        """
        This function changes the state of the service to the specified state by 
        making an appropriate sequence of calls to state change functions. Some
        state transitions are not allowed and are discarded.


        Parameters
        ----------
        desired_state : state
            The state the service should be changed to.
        """

        if(self.state == desired_state): return

        #Change to initialized state
        if(desired_state == state.initialized):
            if(self.state==state.uninitialized):
                print("Changing state: %s -> %s"%(self.state, desired_state))
                self._framework_initialize()
                self.service_initialize()
                self.state = state.initialized
            else:
                print("Can not change from state %s to %s. Request ignored."%(self.state, desired_state))
                return

        #Change to monitoring state
        elif(desired_state == state.monitoring):
            if(self.state==state.initialized or self.state == state.uninitialized):
                if(self.state == state.uninitialized): self._change_state(state.initialized)
                print("Changing state: %s -> %s"%(self.state, desired_state))
                self.state      = state.monitoring
                self.run_thread = Thread(target=self.run)
                self.run_thread.start()
                return
            elif(self.state==state.collecting):
                print("Changing state: %s -> %s"%(self.state, desired_state))
                self.service_stop_collect()
                self._framework_stop_collect()
                self.state = state.monitoring
                return
            else:
                print("Can not change from state %s to %s. Request ignored."%(self.state, desired_state))
            
        #Change to collecting state 
        elif(desired_state == state.collecting):
            if(self.state in [state.uninitialized, state.initialized, state.monitoring]):
                if(self.state == state.uninitialized): self._change_state(state.initialized)
                if(self.state == state.initialized):   self._change_state(state.monitoring)

                print("Changing state: %s -> %s"%(self.state, desired_state))
                self._framework_initialize_collect()
                self.service_initialize_collect()
                self.state = state.collecting
                return
            else:
                print("Can not change from state %s to %s. Request ignored."%(self.state, desired_state))  

        #Change to quit state
        elif(desired_state == state.quit):
            if(self.state in [state.uninitialized]):
                print("Changing state: %s -> %s"%(self.state, desired_state))
                self.state = state.quit
                return
            elif(self.state in [state.initialized]):
                print("Changing state: %s -> %s"%(self.state, desired_state))
                self.state = state.quit
                self.service_stop()
                self._framework_stop()
                print("Waiting for all threads to stop")
                return
            elif(self.state in [state.collecting,state.monitoring]):
                self._change_state(state.monitoring)
                print("Changing state: %s -> %s"%(self.state, desired_state))
                self.state = state.quit
                self.service_stop()
                self._framework_stop()
                self.run_thread.join()
                print("Waiting for all threads to stop")
                return
            else:
                print("Can not change from state %s to %s. Request ignored."%(self.state, desired_state))  

    @final
    def _get_desired_state(self)->state:
        """
        Return the desired state of the service based on the network controller.
        Warning: Currently not implemented!

        Returns
        -------
        state
            The desired state as defined by the controller 
        """
        return state.monitoring
    
    @final
    def start(self)->None:
        """
        Initialize the framework and change to the state indicated by the
        network controller. 
        """
        self._change_state(state.initialized)
        self._change_state(self._get_desired_state())

    @abstractmethod
    def service_initialize(self)->None:
        """
        This function is called when the service is initialized. It should do
        any final work needed to initialize service components.

        Raises
        ------
        NotImplementedError
            This function must be implemented in a derived class.
        """
        print("Calling base implementation of service initialize") 
        pass

    @abstractmethod
    def service_stop(self)->None:
        """
        This function is called when the service should stop. It should close
        any resources opened when the service was initialized.

        Raises
        ------
        NotImplementedError
            This function must be implemented in a derived class.
        """
        pass

    @abstractmethod
    def service_initialize_collect(self)->None:
        """
        This function is called immediately before data collection should start.
        The implementation should perform any setup needed to open files for
        data storage in this function.

        Raises
        ------
        NotImplementedError
            This function must be implemented in a derived class.
        """
        pass

    @abstractmethod
    def service_stop_collect(self)->None:
        """
        This function is called when data collection should stop.
        The implementation should perform any actions needed to close files 
        opened during the data collection process. 

        Raises
        ------
        NotImplementedError
            This function must be implemented in a derived class.
        """ 
        pass

    @abstractmethod
    def service_step(self)->None:
        """
        This function is called in a loop while in the monitoring or collecting
        state. The service can do any amount of work per call to service_step.
        The implementation should monitor for state changes between the monitoring,
        collecting and quit states and take appropriate actions.

        Raises
        ------
        NotImplementedError
            This function must be implemented in a derived class.
        """ 
        pass 

    @final
    def run(self)->None:
        """
        This function is the main loop for the service. It repeatedly calls the 
        service.step() function while the state of the service is either monitoring
        or collecting. Note that while this main loop is executing, callbacks that
        change the service state can execute on other threads. The service_step()
        implementation must watch for state changes and take appropriate actions.
        """
        while( (self.state==state.monitoring or self.state==state.collecting)):
            self.service_step()

    @final
    def _framework_initialize(self)->None:
        """
        This is the final initialization for the service framework layer.
        """
        print("Calling framework initialize for " + self.get_service_name())
        signal.signal(signal.SIGINT, self._sigint_handler)
        self._send_collect_status()

    @final
    def _framework_stop(self)->None:
        """
        This function perform framework-level shutdown operations such as closing 
        registered pub/sub connections as a final step in stopping the running service.
        """
        print("Calling framework stop")

        for topic in self.mqtt_subscriber_callbacks:
            self.mqtt_client.unsubscribe(topic)
        self.mqtt_client.disconnect()

        self.nng_pub.close()
        for topic in self.nng_subscribers:
            self.nng_subscribers[topic].close()

    @final
    def _framework_initialize_collect(self)->None:
        """
        This function performs framework-level operations when switching to the 
        collect state.
        """
        print("Calling framework start collect")
        self.collection_start_time = datetime.now()

    @final
    def _framework_stop_collect(self)->None:
        """
        This function performs framework-level operations when switching from the
        collect state to the monitoring state.
        """
        print("Calling framework stop collect")

    @final
    def get_hostname(self)->str:
        """
        This function gets a canonical representation of the host name.

        Returns
        -------
        str
            The canonical representation of the host name.
        """
        return env["MCP_NODE_NAME"]["val"].replace("-","_")

    @final
    def get_service_name(self)->str:
        """
        This function get a canonical representation of the full service name. The 
        full service is a concatenation of the canonical hostname name and the 
        name of the service.

        Returns
        -------
        str
            The full service name as <hostname>_<servicename>
        """
        return self.get_hostname() + "_" + self.servicename

    @final
    def get_node_short_name(self,node_name=None)->str:
        """
        Get the short name of the node. The short name of the node is of the form node<number>.

        Returns
        -------
        str
            The short name of the node.
        """
        if(node_name is None):
            node_name  = self.get_hostname()
        node_number = node_name.split("_")[-1]
        return f"node{node_number}"

    @final
    def get_topic_name(self,topic:str)->str:
        """
        Get the canonical full topic name for a given base topic string.

        Parameters
        ----------
        topic : str
            base topic string

        Returns
        -------
        str
            Canonical full topic name containing the hostname, service name, and topic.
        """
        topic_name = "/" + env["MCP_NODE_NAME"]["val"].replace("-","_") + "/" + self.servicename + "/" + topic
        # print(topic_name)
        return topic_name
    
    #def get_custom_topic_name(self,topic):
    #    custom_topic_name = "/" + socket.gethostname().replace("-","_") + "/" + self.name + "/" + topic._type.split("/")[-1]
    #    # print(custom_topic_name)
    #    return custom_topic_name

    @final  
    def get_file_name(self,ext:str,suffix:Optional[str]=None)->str:
        """
        Get a canonical name for a file used to store data. The name is based on the 
        collection start time, the full service name including the hostname, 
        a file extension and an optional suffix. 

        Parameters
        ----------
        ext : str
            The file extension to use.
        suffix : Optional[str], optional
            An optional suffix to add to the file name before the extension, by default None

        Returns
        -------
        str
            The canonical name for a file used to store data.
        """
        date_time = self.collection_start_time.strftime("%Y%m%d-%H%M%S")
        if suffix is None:
            file_name = os.path.join(self.output_dir, ('%s_%s.%s'%(date_time,self.get_service_name(),ext)).replace("-","_"))
        else:
            file_name = os.path.join(self.output_dir, ('%s_%s_%s.%s'%(date_time,self.get_service_name(),suffix,ext)).replace("-","_"))
        return file_name

    @final
    def _send_collect_status(self)->None:
        """
        This function sends a heartbeat signal to the controller once per second 
        indicating whether the service is in the collect state or note. 
        The message sent is the string 'True' or 'False'.
        """
        if(time.time()-self.last_net_pub_time < 2):
            #If service has not published data within the last two seconds, don't send 
            #the heartbeat to indicate to the dashboard that there may be a problem with the
            #service.
            self.publish("net",self.collect_status_topic, str(self.state==state.collecting))
    
        if(not self.state==state.quit):
             Timer(1, self._send_collect_status).start()
        else:
            print("Stopping sending collect status")
