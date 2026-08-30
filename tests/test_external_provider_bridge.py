import sys
import time

from deployment.external_provider import ExternalProviderBridgeRuntime
from fable.execution import DataflowProviderRuntime
from fable.execution.plan_reconciler import ProviderInstanceKey, ProviderInstanceSpec
from fable.execution.stream_bus import StreamBus


def test_external_bridge_validates_and_publishes_detection_frames() -> None:
    bus = StreamBus()
    received = []
    bus.subscribe("detections", lambda _key, value: received.append(value), source_ids=("camera1",))
    child = (
        "import json,time; "
        "print(json.dumps({'type':'ready'}),flush=True); "
        "print(json.dumps({'type':'detection_frame','source_id':'camera1',"
        "'event_time':'2026-01-01T00:00:00+00:00','frame_id':'1',"
        "'image_width':640,'image_height':480,'detections':[{'class_name':'car',"
        "'confidence':0.9,'bbox':[1,2,30,40]}]}),flush=True); time.sleep(1)"
    )
    runtime = ExternalProviderBridgeRuntime(
        DataflowProviderRuntime(bus=bus),
        {"yolo_vehicle_fast_640": (sys.executable, "-u", "-c", child)},
        ready_timeout_seconds=2,
    )
    key = ProviderInstanceKey("yolo_vehicle_fast_640", "jetson", ("camera1",))
    runtime.start(ProviderInstanceSpec(key, "detections"))
    deadline = time.time() + 2
    while not received and time.time() < deadline:
        time.sleep(0.01)
    assert runtime.ready(key)
    assert received[0].detections[0].class_name == "car"
    runtime.stop(key)
