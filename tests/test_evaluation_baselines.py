from fable.common.examples import BASE_TIME
from fable.planning import BoundedLabelPlanner
from fable.planning.phase4_testing import fake_follow_alternative_graph
from fable.planning.testing import fake_artifact_catalog, fake_deployment, fake_provider_registry

from evaluation.baselines import (
    AlwaysOnPolicy,
    BaselinePlanningCase,
    ExhaustiveOraclePolicy,
    FablePolicy,
    GreedyFrontierPolicy,
    StaticWholeEventPolicy,
    TaskResourceAdaptivePolicy,
)


def _fixture(*, resource_epoch: int = 0, semantic_epoch: int = 0):
    providers = fake_provider_registry()
    artifacts = fake_artifact_catalog()
    deployment = fake_deployment()
    graph, demand = fake_follow_alternative_graph(
        provider_registry=providers,
        artifact_catalog=artifacts,
        deployment=deployment,
    )
    planner = BoundedLabelPlanner(
        provider_registry=providers,
        artifact_catalog=artifacts,
        deployment=deployment,
    )
    case = BaselinePlanningCase(
        run_id="run-eval",
        trace_id="trace-eval",
        request_id="request-eval",
        event_family="route_convoy",
        frontier_demands=(demand,),
        all_task_demands=(demand,),
        frontier_graph=graph,
        whole_event_graph=graph,
        now=BASE_TIME,
        resource_epoch=resource_epoch,
        semantic_epoch=semantic_epoch,
    )
    return case, planner


def test_always_on_activates_more_than_frontier_planners() -> None:
    case, planner = _fixture()
    always = AlwaysOnPolicy().plan(case)
    greedy = GreedyFrontierPolicy().plan(case)
    fable = FablePolicy(planner).plan(case)
    assert len(always.selected_alternative_ids) > len(greedy.selected_alternative_ids)
    assert len(greedy.selected_alternative_ids) == 1
    assert len(fable.selected_alternative_ids) == 1


def test_static_whole_event_freezes_plan_across_epochs() -> None:
    case, planner = _fixture()
    policy = StaticWholeEventPolicy(planner)
    first = policy.plan(case)
    second_case = BaselinePlanningCase(
        **{**case.__dict__, "resource_epoch": 4, "semantic_epoch": 9}
    )
    second = policy.plan(second_case)
    assert first.frozen and second.frozen
    assert first.selected_alternative_ids == second.selected_alternative_ids
    assert second.resource_epoch == 4
    assert "frozen plan" in second.reason


class CountingPlanner:
    def __init__(self, wrapped: BoundedLabelPlanner) -> None:
        self.wrapped = wrapped
        self.calls = 0

    def search(self, *args, **kwargs):
        self.calls += 1
        return self.wrapped.search(*args, **kwargs)


def test_task_adaptive_replans_only_for_resource_epoch() -> None:
    case, planner = _fixture(resource_epoch=1, semantic_epoch=1)
    counting = CountingPlanner(planner)
    policy = TaskResourceAdaptivePolicy(counting)  # type: ignore[arg-type]
    policy.plan(case)
    policy.plan(BaselinePlanningCase(**{**case.__dict__, "semantic_epoch": 8}))
    assert counting.calls == 1
    policy.plan(
        BaselinePlanningCase(
            **{**case.__dict__, "resource_epoch": 2, "semantic_epoch": 8}
        )
    )
    assert counting.calls == 2


def test_exhaustive_oracle_returns_a_complete_plan() -> None:
    case, planner = _fixture()
    decision = ExhaustiveOraclePolicy(planner).plan(case)
    assert len(decision.selected_alternative_ids) == 1
    assert "exhaustive" in decision.reason.lower()


def test_mobile_source_is_excluded_when_replay_is_orin_only() -> None:
    case, _ = _fixture()
    original = case.whole_event_graph.alternatives[0]
    inputs = list(original.external_inputs)
    index = next(i for i, item in enumerate(inputs) if item.source_id is not None)
    inputs[index] = inputs[index].model_copy(update={"source_id": "d3"})
    mobile = original.model_copy(
        update={"alternative_id": "mobile-only", "external_inputs": tuple(inputs)}
    )
    graph = case.whole_event_graph.model_copy(update={"alternatives": (mobile,)})
    mobile_case = BaselinePlanningCase(
        **{
            **case.__dict__,
            "frontier_graph": graph,
            "whole_event_graph": graph,
            "replay_supported_sensor_ids": ("orin_1", "orin_5"),
        }
    )
    decision = AlwaysOnPolicy().plan(mobile_case)
    assert decision.selected_alternative_ids == ()
    assert decision.excluded_mobile_or_unavailable_sources == ("d3",)
