#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

FABLE_ROOT = Path(__file__).resolve().parents[2]
if str(FABLE_ROOT) not in sys.path:
    sys.path.insert(0, str(FABLE_ROOT))

from evaluation.runner import JsonlEventStore
from evaluation.runtime_logging import (
    EvaluationMessageNormalizer,
    MqttEvaluationLogger,
    RuntimeLoggingContext,
)
from evaluation.schemas import BaselineId


def main() -> None:
    parser = argparse.ArgumentParser(description="Log typed FABLE MQTT runtime records for one evaluation run.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--baseline", choices=[item.value for item in BaselineId], required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation-runs"))
    args = parser.parse_args()
    run_dir = args.output / args.run_id / args.baseline
    logger = MqttEvaluationLogger(
        host=args.host,
        port=args.port,
        client_id=f"fable-eval-{args.run_id}-{args.baseline}",
        store=JsonlEventStore(run_dir),
        normalizer=EvaluationMessageNormalizer(
            RuntimeLoggingContext(
                run_id=args.run_id,
                baseline_id=BaselineId(args.baseline),
                trace_id=args.trace_id,
                default_request_id=args.request_id,
            )
        ),
    )
    print(f"logging to {run_dir}")
    logger.run_forever()


if __name__ == "__main__":
    main()
