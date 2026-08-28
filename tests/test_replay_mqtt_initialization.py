from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mqtt_callback_state_is_initialized_before_network_loop_starts():
    source = (
        ROOT / "iobt-minimal-ce-replay/lib/iobt_max_service.py"
    ).read_text(encoding="utf-8")
    constructor = source[source.index("    def __init__("):source.index("    @final")]
    loop_start = constructor.index("self.mqtt_client.loop_start()")
    assert constructor.index("self.mqtt_subscriber_callbacks = {}") < loop_start
    assert constructor.index("self.service_control_topic =") < loop_start
