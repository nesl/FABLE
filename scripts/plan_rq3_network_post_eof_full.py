#!/usr/bin/env python3
"""Generate the full nominal/post-EOF sensor-disconnect RQ3 matrix."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.catalog import ExperimentCatalog  # noqa: E402
from evaluation.experiments.matrix import PlannedRun  # noqa: E402
from evaluation.experiments.specs import ExperimentQuestion  # noqa: E402
from evaluation.schemas import BaselineId, EvaluationMode  # noqa: E402
from fable.common.ids import deterministic_id  # noqa: E402


OUTPUT_ROOT = ROOT / "evaluation/manifests/adaptation/rq3_post_eof_disconnect_full"
TOPOLOGY = ROOT / "netwaggle/configs/site_evaluation_29node.json"
NETWORK_PROFILE = (
    ROOT / "netwaggle/configs/profiles/site_evaluation_29node/N0.json"
)
POLICIES = (
    BaselineId.B1_STATIC_WHOLE_EVENT,
    BaselineId.B2_FRONTIER_FIXED_REALIZATION,
    BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
    BaselineId.B4_GREEDY_FRONTIER,
    BaselineId.FABLE,
)

# Fixed before evaluation from the validated representative run for each CE
# family. This selects the sensor whose loss is meaningful; it is not changed
# using the outcome of an individual test cell.
SWITCH_BY_VARIANT = {
    "Pass-follow-clear convoy": "s_orin13",
    "Vehicle convergence": "s_orin1",
    "Vehicle rendezvous": "s_mob6",
    "Cross-sensor robbery": "s_orin15",
    "Three-visit stalking": "s_orin16",
    "Route convoy": "s_orin4",
    "Two-vehicle chase": "s_orin1",
    "Robbery with alarm": "s_mob4",
    "Talking/rendezvous": "s_mob6",
}

EOF_MARGIN_SECONDS = 5.0
RECOVERY_ALLOWANCE_SECONDS = 225.0
B1_RECOVERY_GRACE_SECONDS = 15.0


def recovery_allowance_seconds(baseline: BaselineId) -> float:
    """Return the post-restore observation budget for one policy.

    B1 is a fixed authored pipeline and has no explicit bounded historical
    recovery mechanism.  The adaptive/ablation policies share FABLE's live
    execution substrate and may legitimately finish a recovery after link
    restoration.
    """

    if baseline == BaselineId.B1_STATIC_WHOLE_EVENT:
        return B1_RECOVERY_GRACE_SECONDS
    return RECOVERY_ALLOWANCE_SECONDS


def requested_replay_window_seconds(experiment) -> float:
    """Mirror run_replay_accuracy's automatic labeled replay window."""

    # The accuracy driver requests five seconds before the exact labeled
    # recording interval and retains its 30-second semantic deadline after it.
    # Catalog scenarios containing recommended experiments all begin before
    # that five-second prefix, so the requested interval is the exact label
    # span plus 35 seconds. Use the timestamps rather than the rounded
    # duration_seconds column (the convoy pilot is 50 + 35 = 85 seconds).
    exact_span = (
        experiment.recording_end - experiment.recording_start
    ).total_seconds()
    return max(0.0, exact_span + 35.0)


def post_eof_trace(
    experiment_id: str,
    switch_id: str,
    replay_window: float,
    recording_span: float,
    recovery_allowance: float = RECOVERY_ALLOWANCE_SECONDS,
) -> dict:
    slug = experiment_id.replace("_", "-")
    disconnect_at = float(recording_span * 0.25)
    restore_at = float(math.ceil(replay_window + EOF_MARGIN_SECONDS))
    return {
        "schema_version": "fable.condition_trace.v1",
        "trace_id": f"rq3-network-post-eof-disconnect-{slug}",
        "initial_network_profile": "N0",
        "initial_compute_profile": "N0",
        # Anchor to the synchronized replay clock.  Admission can lag seed
        # processing, especially for static pipelines, which previously let a
        # baseline finish before an offset-zero admission-relative outage was
        # applied.  TRACE_START makes the disturbance schedule independent of
        # request/planner latency.
        "anchor": "TRACE_START",
        "transitions": [
            {
                "transition_id": f"{slug}:post-eof-disconnect",
                "offset_s": disconnect_at,
                "action": "FAIL_LINK",
                "target_id": f"link:{switch_id}:s_edge",
            },
            {
                "transition_id": f"{slug}:post-eof-restore",
                "offset_s": restore_at,
                "action": "RESTORE_LINK",
                "target_id": f"link:{switch_id}:s_edge",
            },
        ],
        "duration_s": restore_at + recovery_allowance,
        "random_seed": 31034,
    }


def main() -> int:
    catalog = ExperimentCatalog.from_csv(
        ROOT / "evaluation/labels/filtered_complex_event_experiments.csv"
    )
    experiments = tuple(catalog.recommended())
    variants = {item.ce_variant for item in experiments}
    if variants != set(SWITCH_BY_VARIANT):
        raise RuntimeError(
            "post-EOF switch mapping does not exactly cover recommended variants: "
            f"missing={sorted(variants - set(SWITCH_BY_VARIANT))}, "
            f"extra={sorted(set(SWITCH_BY_VARIANT) - variants)}"
        )

    topology = json.loads(TOPOLOGY.read_text(encoding="utf-8"))
    topology_links = {
        frozenset((str(item["from"]), str(item["to"])))
        for item in topology["links"]
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[PlannedRun] = []
    cases: list[dict[str, object]] = []
    for experiment in experiments:
        switch_id = SWITCH_BY_VARIANT[experiment.ce_variant]
        if frozenset((switch_id, "s_edge")) not in topology_links:
            raise RuntimeError(f"topology has no link between {switch_id} and s_edge")
        recording_span = (
            experiment.recording_end - experiment.recording_start
        ).total_seconds()
        replay_window = requested_replay_window_seconds(experiment)
        trace = post_eof_trace(
            experiment.experiment_id, switch_id, replay_window, recording_span
        )
        cases.append(
            {
                "experiment_id": experiment.experiment_id,
                "ce_variant": experiment.ce_variant,
                "campaign_year": experiment.campaign_year,
                "labeled_duration_seconds": experiment.duration_seconds,
                "requested_replay_window_seconds": replay_window,
                "disconnect_switch": switch_id,
                "disconnect_link": f"link:{switch_id}:s_edge",
                "disconnect_offset_seconds": trace["transitions"][0]["offset_s"],
                "disconnect_fraction_of_recording": 0.25,
                "restore_offset_seconds": trace["transitions"][1]["offset_s"],
                "post_eof_margin_seconds": EOF_MARGIN_SECONDS,
                "recovery_allowance_seconds_by_baseline": {
                    baseline.value: recovery_allowance_seconds(baseline)
                    for baseline in POLICIES
                },
            }
        )
        for condition in ("N0", "POST_EOF_SENSOR_DISCONNECT"):
            for baseline in POLICIES:
                baseline_trace = post_eof_trace(
                    experiment.experiment_id,
                    switch_id,
                    replay_window,
                    recording_span,
                    recovery_allowance_seconds(baseline),
                )
                trace_path = OUTPUT_ROOT / (
                    f"post_eof_{experiment.experiment_id}_{baseline.value}.json"
                )
                trace_path.write_text(
                    json.dumps(baseline_trace, indent=2) + "\n", encoding="utf-8"
                )
                identity = {
                    "matrix": "rq3_network_post_eof_full_v1",
                    "experiment_id": experiment.experiment_id,
                    "condition": condition,
                    "baseline": baseline.value,
                    "seed": 31034,
                }
                condition_kwargs: dict[str, object] = {}
                if condition == "POST_EOF_SENSOR_DISCONNECT":
                    condition_kwargs = {
                        "condition_trace_id": baseline_trace["trace_id"],
                        "condition_trace_path": str(trace_path),
                    }
                rows.append(
                    PlannedRun(
                        run_id=deterministic_id("eval_run", identity, length=32),
                        question=ExperimentQuestion.RQ3_OPERATING_ADAPTATION,
                        experiment_id=experiment.experiment_id,
                        baseline_id=baseline,
                        mode=EvaluationMode.FULL_STACK,
                        network_profile_id="good_network",
                        network_profile_path=str(NETWORK_PROFILE),
                        spatial_metrics_enabled=False,
                        repetition=1,
                        random_seed=31034,
                        playback_mode="realtime",
                        provider_profile_version="rq3-network-post-eof-v1",
                        disturbance_profile_id=condition,
                        ce_start_offset_seconds=0.0,
                        provider_execution_mode="real",
                        vlm_mode="replayed_response",
                        warnings=experiment.spatial_notes,
                        campaign_year=experiment.campaign_year,
                        **condition_kwargs,
                    )
                )

    manifest = OUTPUT_ROOT / "rq3_network_post_eof_full_830.jsonl"
    manifest.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )
    semantic_keys = {
        (row.experiment_id, row.baseline_id.value, row.disturbance_profile_id)
        for row in rows
    }
    if len(experiments) != 83 or len(rows) != 830 or len(semantic_keys) != 830:
        raise RuntimeError(
            f"unexpected matrix size: traces={len(experiments)} "
            f"cells={len(rows)} unique={len(semantic_keys)}"
        )
    if any(
        float(item["restore_offset_seconds"])
        <= float(item["requested_replay_window_seconds"])
        for item in cases
    ):
        raise RuntimeError("a post-EOF restore is not strictly after replay EOF")

    counts = Counter(item["ce_variant"] for item in cases)
    summary = {
        "schema_version": "fable.rq3_network_post_eof_plan.v1",
        "manifest": str(manifest),
        "topology": str(TOPOLOGY),
        "planned_runs": len(rows),
        "traces": len(experiments),
        "trace_counts_by_ce_variant": dict(sorted(counts.items())),
        "conditions": ["N0", "POST_EOF_SENSOR_DISCONNECT"],
        "systems": [item.value for item in POLICIES],
        "runs_per_trace": 10,
        "repetitions": 1,
        "playback_mode": "realtime",
        "playback_speed": 1.0,
        "condition_anchor": "TRACE_START",
        "disconnect_schedule": (
            "fail selected link 25% into the labeled recording span; restore "
            "strictly after the trace-specific requested replay window"
        ),
        "ordinary_evidence_boundary": "close at replay EOF",
        "offline_evidence_policy": "drop; explicit bounded raw recovery only",
        "recovery_budget_policy": {
            "B1_STATIC_WHOLE_EVENT": B1_RECOVERY_GRACE_SECONDS,
            "B2_FRONTIER_FIXED_REALIZATION": RECOVERY_ALLOWANCE_SECONDS,
            "B3_TASK_RESOURCE_ADAPTIVE": RECOVERY_ALLOWANCE_SECONDS,
            "B4_GREEDY_FRONTIER": RECOVERY_ALLOWANCE_SECONDS,
            "FABLE": RECOVERY_ALLOWANCE_SECONDS,
            "unit": "seconds after restoration",
            "terminal_exit": "immediate",
        },
        "execution_order": (
            "ce-round-robin; post-EOF baselines then nominal baselines adjacent "
            "within each trace"
        ),
        "condition_order": "disturbed-first",
        "recommended_output_root": (
            "/media/brianw/Extreme SSD2/fable_results/"
            "rq3_network_post_eof_full_830_v1_20260808"
        ),
        "cases": cases,
    }
    summary_path = OUTPUT_ROOT / "rq3_network_post_eof_full_830.summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "manifest": str(manifest),
        "summary": str(summary_path),
        "traces": len(experiments),
        "planned_runs": len(rows),
        "maximum_restore_offset_seconds": max(
            item["restore_offset_seconds"] for item in cases
        ),
        "maximum_condition_duration_seconds": max(
            json.loads(
                (
                    OUTPUT_ROOT
                    / f"post_eof_{item['experiment_id']}_{baseline.value}.json"
                ).read_text()
            )["duration_s"]
            for item in cases
            for baseline in POLICIES
        ),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
