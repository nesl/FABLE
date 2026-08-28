from dataclasses import replace

import pytest

from evaluation.baselines import (
    AlwaysOnPolicy,
    BaselinePlanningCase,
    FablePolicy,
    GreedyFrontierPolicy,
    StaticWholeEventPolicy,
    TaskResourceAdaptivePolicy,
)
from evaluation.orchestration import ControlledPlanningCoordinator, PlanningTrigger
from evaluation.schemas import BaselineId
from fable.common.examples import BASE_TIME
from fable.planning import BoundedLabelPlanner
from fable.planning.phase4_testing import fake_follow_alternative_graph
from fable.planning.testing import fake_artifact_catalog, fake_deployment, fake_provider_registry


def _fixture() -> tuple[BaselinePlanningCase, BoundedLabelPlanner]:
    providers = fake_provider_registry()
    artifacts = fake_artifact_catalog()
    deployment = fake_deployment()
    graph, demand = fake_follow_alternative_graph(
        provider_registry=providers,
        artifact_catalog=artifacts,
        deployment=deployment,
    )
    return (
        BaselinePlanningCase(
            run_id="run",
            trace_id="trace",
            request_id="request",
            event_family="route_convoy",
            frontier_demands=(demand,),
            all_task_demands=(demand,),
            frontier_graph=graph,
            whole_event_graph=graph,
            now=BASE_TIME,
        ),
        BoundedLabelPlanner(
            provider_registry=providers,
            artifact_catalog=artifacts,
            deployment=deployment,
        ),
    )


@pytest.mark.parametrize(
    "policy_factory",
    [
        lambda planner: AlwaysOnPolicy(),
        lambda planner: StaticWholeEventPolicy(planner),
    ],
)
def test_admission_policies_remain_frozen(policy_factory) -> None:
    case, planner = _fixture()
    coordinator = ControlledPlanningCoordinator(policy_factory(planner))
    admitted = coordinator.decide(case, trigger=PlanningTrigger.ADMISSION)
    later = coordinator.decide(
        replace(case, resource_epoch=2, semantic_epoch=3),
        trigger=PlanningTrigger.RESOURCE_EPOCH,
    )
    assert later is admitted


def test_b3_replans_only_on_changed_resource_epoch() -> None:
    case, planner = _fixture()
    coordinator = ControlledPlanningCoordinator(TaskResourceAdaptivePolicy(planner))
    admitted = coordinator.decide(case, trigger=PlanningTrigger.ADMISSION)
    semantic = coordinator.decide(
        replace(case, semantic_epoch=1),
        trigger=PlanningTrigger.SEMANTIC_FRONTIER,
    )
    resource = coordinator.decide(
        replace(case, resource_epoch=1, semantic_epoch=1),
        trigger=PlanningTrigger.RESOURCE_EPOCH,
    )
    assert semantic is admitted
    assert resource is not admitted
    assert resource.resource_epoch == 1


@pytest.mark.parametrize(
    "policy_factory,baseline_id",
    [
        (lambda planner: GreedyFrontierPolicy(), BaselineId.B4_GREEDY_FRONTIER),
        (lambda planner: FablePolicy(planner), BaselineId.FABLE),
    ],
)
def test_frontier_policies_replan_on_semantic_epoch(policy_factory, baseline_id) -> None:
    case, planner = _fixture()
    coordinator = ControlledPlanningCoordinator(policy_factory(planner))
    coordinator.decide(case, trigger=PlanningTrigger.ADMISSION)
    decision = coordinator.decide(
        replace(case, semantic_epoch=1),
        trigger=PlanningTrigger.SEMANTIC_FRONTIER,
    )
    assert decision.baseline_id == baseline_id
    assert decision.semantic_epoch == 1


def test_non_admission_trigger_requires_existing_request() -> None:
    case, _ = _fixture()
    coordinator = ControlledPlanningCoordinator(AlwaysOnPolicy())
    with pytest.raises(ValueError, match="no admission-time"):
        coordinator.decide(case, trigger=PlanningTrigger.RESOURCE_EPOCH)
