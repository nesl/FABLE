from __future__ import annotations

import os
from pathlib import Path

from evaluation.runner import JsonlEventStore
from evaluation.runtime_logging import (
    EvaluationMessageNormalizer,
    MqttEvaluationLogger,
    RuntimeLoggingContext,
)
from evaluation.schemas import BaselineId


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable {name}")
    return value


def main() -> None:
    run_id = required("FABLE_EVAL_RUN_ID")
    baseline = BaselineId(required("FABLE_EVAL_BASELINE"))
    trace_id = required("FABLE_EVAL_TRACE_ID")
    request_id = required("FABLE_EVAL_REQUEST_ID")
    output_root = Path(os.environ.get("FABLE_EVAL_OUTPUT", "/var/lib/fable/evaluation"))
    run_dir = output_root / run_id / baseline.value
    logger = MqttEvaluationLogger(
        host=os.environ.get("MQTT_HOST_IP", "mqtt"),
        port=int(os.environ.get("MQTT_PORT", "1883")),
        client_id=f"fable-eval-{run_id}-{baseline.value.lower()}",
        store=JsonlEventStore(run_dir),
        normalizer=EvaluationMessageNormalizer(
            RuntimeLoggingContext(
                run_id=run_id,
                baseline_id=baseline,
                trace_id=trace_id,
                default_request_id=request_id,
            )
        ),
    )
    print(f"FABLE evaluation logger writing to {run_dir}", flush=True)
    logger.run_forever()


if __name__ == "__main__":
    main()
