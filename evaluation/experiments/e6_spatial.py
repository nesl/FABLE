"""E6 spatial-coordination planned-run builder."""

from pathlib import Path

from .matrix import build_run_matrix
from .matrix import PlannedRun
from .specs import ExperimentQuestion
from evaluation.coordination_logging import CoordinationEpisodeTracker

QUESTION = ExperimentQuestion.RQ3_SPATIAL_COORDINATION


def build(catalog, **kwargs):
    return build_run_matrix(catalog, QUESTION, **kwargs)


def coordination_tracker_for_run(
    run,
    *,
    upstream_predicate_ids: tuple[str, ...],
    downstream_predicate_ids: tuple[str, ...],
) -> CoordinationEpisodeTracker:
    if not run.spatial_metrics_enabled or run.campaign_year not in (2024, 2025):
        raise ValueError("coordination tracker requires an eligible E6 run")
    return CoordinationEpisodeTracker(
        campaign_year=run.campaign_year,
        spatial_evaluation_eligible=True,
        replay_supported_sensor_ids=run.replay_supported_sensor_ids,
        unavailable_mobile_sensor_ids=run.unavailable_mobile_sensor_ids,
        upstream_predicate_ids=upstream_predicate_ids,
        downstream_predicate_ids=downstream_predicate_ids,
        topology_confidence=(
            "measured"
            if len(run.topology_deployment_ids) == 1
            else "ambiguous"
        ),
        route_ambiguity=max(1, len(run.topology_deployment_ids)),
    )


def coordination_tracker_from_manifest(
    manifest_path: str | Path,
    run_id: str,
    *,
    upstream_predicate_ids: tuple[str, ...],
    downstream_predicate_ids: tuple[str, ...],
) -> CoordinationEpisodeTracker:
    """Resolve exactly one immutable E6 run and build its record deriver."""

    matches = []
    for line_number, line in enumerate(
        Path(manifest_path).read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        try:
            run = PlannedRun.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(
                f"invalid planned run at line {line_number}: {exc}"
            ) from exc
        if run.run_id == run_id:
            matches.append(run)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one planned run {run_id!r}; found {len(matches)}"
        )
    return coordination_tracker_for_run(
        matches[0],
        upstream_predicate_ids=upstream_predicate_ids,
        downstream_predicate_ids=downstream_predicate_ids,
    )
