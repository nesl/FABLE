#!/usr/bin/env python3
"""Generate a frozen nominal/causal sensor-disconnect campaign.

The disturbance is derived before evaluation and is identical for every
policy on a trace.  A trace's accepted nominal event start is used when a
successful reference exists; otherwise the median event-start fraction from
successful references in the same CE family is applied to that trace.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.catalog import ExperimentCatalog  # noqa: E402
from evaluation.experiments.matrix import PlannedRun  # noqa: E402
from evaluation.experiments.specs import ExperimentQuestion  # noqa: E402
from evaluation.schemas import BaselineId, EvaluationMode  # noqa: E402
from fable.common.ids import deterministic_id  # noqa: E402

from scripts.plan_rq3_network_post_eof_full import (  # noqa: E402
    EOF_MARGIN_SECONDS,
    NETWORK_PROFILE,
    RECOVERY_ALLOWANCE_SECONDS,
    SWITCH_BY_VARIANT,
    recovery_allowance_seconds,
    TOPOLOGY,
    requested_replay_window_seconds,
)


OUTPUT_ROOT = ROOT / "evaluation/manifests/adaptation/causal_disconnect_full"
REFERENCE_ROOT = (
    ROOT
    / "evaluation/results/rq1_full_83_traces_all_six_policies_static_fixed_20260806_v2"
)
SUPPLEMENTAL_REFERENCE_ROOTS = (
    ROOT / "evaluation/results/rq3a_network_full_coverage_20260806",
)
POLICIES = (
    BaselineId.B1_STATIC_WHOLE_EVENT,
    BaselineId.B2_FRONTIER_FIXED_REALIZATION,
    BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
    BaselineId.B4_GREEDY_FRONTIER,
    BaselineId.FABLE,
)


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _result_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory, _, names in os.walk(root, followlinks=True):
        parent = Path(directory)
        for name in names:
            if name.endswith(".json") and name not in {"plan.json", "report.json"}:
                files.append(parent / name)
    return files


def _reference(path: Path) -> dict[str, object] | None:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    predictions = result.get("predictions") or []
    timing = result.get("timing") or {}
    observed_start = timing.get("observed_event_time_start")
    if (
        result.get("classification") != "TRUE_POSITIVE"
        or not predictions
        or not observed_start
        or (result.get("condition_trace") or {}).get("transitions")
        or result.get("network_disturbance")
    ):
        return None
    event_start = predictions[0].get("event_start_time")
    if not event_start:
        return None
    offset = (_time(event_start) - _time(observed_start)).total_seconds()
    span = float(timing.get("observed_event_time_span_seconds") or 0)
    if offset < 0 or span <= 0:
        return None
    return {
        "path": str(path.resolve()),
        "baseline": str(result.get("baseline") or ""),
        "offset_seconds": round(offset, 3),
        "observed_span_seconds": span,
        "fraction": min(0.95, max(0.01, offset / span)),
        "event_start_time": event_start,
        "observed_start_time": observed_start,
    }


def _reference_rank(item: dict[str, object]) -> tuple[int, str]:
    baseline = str(item["baseline"])
    rank = 0 if baseline == "FABLE" else 1 if baseline == "B1_STATIC_WHOLE_EVENT" else 2
    return rank, str(item["path"])


def condition_trace(
    experiment,
    *,
    switch_id: str,
    offset: float,
    recovery_allowance: float = RECOVERY_ALLOWANCE_SECONDS,
) -> dict:
    replay_window = requested_replay_window_seconds(experiment)
    restore_at = float(math.ceil(replay_window + EOF_MARGIN_SECONDS))
    slug = experiment.experiment_id.replace("_", "-")
    return {
        "schema_version": "fable.condition_trace.v1",
        "trace_id": f"causal-first-observation-{slug}",
        "initial_network_profile": "N0",
        "initial_compute_profile": "N0",
        "anchor": "TRACE_START",
        "transitions": [
            {
                "transition_id": f"{slug}:causal-first-observation-disconnect",
                "offset_s": round(offset, 3),
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
    experiment_by_id = {item.experiment_id: item for item in experiments}
    references_by_experiment: dict[str, list[dict[str, object]]] = defaultdict(list)
    for reference_root in (REFERENCE_ROOT, *SUPPLEMENTAL_REFERENCE_ROOTS):
        for path in _result_files(reference_root):
            experiment = experiment_by_id.get(path.stem)
            if experiment is None:
                continue
            reference = _reference(path)
            if reference is not None:
                references_by_experiment[experiment.experiment_id].append(reference)

    family_fractions: dict[str, list[float]] = defaultdict(list)
    selected_reference: dict[str, dict[str, object]] = {}
    for experiment_id, candidates in references_by_experiment.items():
        chosen = min(candidates, key=_reference_rank)
        selected_reference[experiment_id] = chosen
        family_fractions[experiment_by_id[experiment_id].ce_variant].append(
            float(chosen["fraction"])
        )
    missing_families = set(SWITCH_BY_VARIANT) - set(family_fractions)
    if missing_families:
        raise RuntimeError(
            "no successful nominal timing exemplar for CE families: "
            + ", ".join(sorted(missing_families))
        )

    topology = json.loads(TOPOLOGY.read_text(encoding="utf-8"))
    links = {
        frozenset((str(item["from"]), str(item["to"])))
        for item in topology["links"]
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[PlannedRun] = []
    provenance: list[dict[str, object]] = []
    for experiment in experiments:
        switch_id = SWITCH_BY_VARIANT[experiment.ce_variant]
        if frozenset((switch_id, "s_edge")) not in links:
            raise RuntimeError(f"topology lacks {switch_id}<->s_edge")
        exact = selected_reference.get(experiment.experiment_id)
        recording_span = (
            experiment.recording_end - experiment.recording_start
        ).total_seconds()
        if exact is not None:
            offset = float(exact["offset_seconds"])
            timing_basis = "trace_successful_nominal_accepted_event_start"
            source = exact
        else:
            fraction = median(family_fractions[experiment.ce_variant])
            offset = round(recording_span * fraction, 3)
            timing_basis = "frozen_ce_family_exemplar_median_fraction"
            source = {
                "family_fraction": fraction,
                "family_reference_count": len(family_fractions[experiment.ce_variant]),
            }
        # Keep the cut inside the labeled recording and after replay begins.
        offset = min(max(0.1, offset), max(0.1, recording_span - 0.1))
        trace = condition_trace(experiment, switch_id=switch_id, offset=offset)
        provenance.append(
            {
                "experiment_id": experiment.experiment_id,
                "ce_variant": experiment.ce_variant,
                "disconnect_offset_seconds": offset,
                "disconnect_link": f"link:{switch_id}:s_edge",
                "timing_basis": timing_basis,
                "source": source,
                "baseline_independent": True,
            }
        )
        for condition in ("CAUSAL_SENSOR_DISCONNECT", "N0"):
            for baseline in POLICIES:
                baseline_trace = condition_trace(
                    experiment,
                    switch_id=switch_id,
                    offset=offset,
                    recovery_allowance=recovery_allowance_seconds(baseline),
                )
                trace_path = OUTPUT_ROOT / (
                    f"causal_{experiment.experiment_id}_{baseline.value}.json"
                )
                trace_path.write_text(
                    json.dumps(baseline_trace, indent=2) + "\n", encoding="utf-8"
                )
                identity = {
                    "matrix": "causal_disconnect_full_v1",
                    "experiment": experiment.experiment_id,
                    "condition": condition,
                    "baseline": baseline.value,
                    "seed": 31034,
                }
                condition_kwargs: dict[str, object] = {}
                if condition == "CAUSAL_SENSOR_DISCONNECT":
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
                        provider_profile_version="causal-disconnect-full-v1",
                        disturbance_profile_id=condition,
                        ce_start_offset_seconds=0.0,
                        provider_execution_mode="real",
                        vlm_mode="replayed_response",
                        warnings=experiment.spatial_notes,
                        campaign_year=experiment.campaign_year,
                        **condition_kwargs,
                    )
                )

    manifest = OUTPUT_ROOT / "causal_disconnect_full_830.jsonl"
    manifest.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )
    (OUTPUT_ROOT / "causal_disconnect_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    exact_count = sum(item["timing_basis"].startswith("trace_") for item in provenance)
    summary = {
        "schema_version": "fable.causal_disconnect_full_plan.v1",
        "trace_count": len(experiments),
        "cell_count": len(rows),
        "policies": [item.value for item in POLICIES],
        "conditions": ["CAUSAL_SENSOR_DISCONNECT", "N0"],
        "trace_specific_timing_count": exact_count,
        "family_exemplar_fallback_count": len(experiments) - exact_count,
        "same_condition_for_all_baselines": True,
        "b1_fanout_allowed": False,
        "recovery_budget_policy": {
            baseline.value: recovery_allowance_seconds(baseline)
            for baseline in POLICIES
        },
        "terminal_exit": "immediate",
        "manifest": str(manifest),
    }
    (OUTPUT_ROOT / "causal_disconnect_full_830.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if len(experiments) != 83 or len(rows) != 830:
        raise RuntimeError(f"unexpected matrix size traces={len(experiments)} rows={len(rows)}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
