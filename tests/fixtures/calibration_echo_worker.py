import json
import sys

request = json.load(sys.stdin)
assert request["schema_version"] == "fable.calibration_worker_request.v1"
print(
    json.dumps(
        {
            "schema_version": "fable.calibration_worker_response.v1",
            "successful": True,
            "quality_score": request["fixture"]["quality_score"],
            "ambiguity_score": request["fixture"]["ambiguity_score"],
        }
    )
)
