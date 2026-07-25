from fable.common.examples import BASE_TIME
from fable.planning.phase4_testing import fake_follow_alternative_graph

from evaluation.baselines import AlwaysOnPolicy, BaselinePlanningCase
from evaluation.runner import EvaluationRunner
from evaluation.schemas import EvaluationMode


def test_runner_writes_common_plan_decision(tmp_path) -> None:
    graph, demand = fake_follow_alternative_graph()
    case = BaselinePlanningCase(
        run_id="run-1",
        trace_id="trace-1",
        request_id="request-1",
        event_family="route_convoy",
        frontier_demands=(demand,),
        all_task_demands=(demand,),
        frontier_graph=graph,
        whole_event_graph=graph,
        now=BASE_TIME,
    )
    runner = EvaluationRunner(tmp_path, mode=EvaluationMode.FULL_STACK)
    decision = runner.run_planning_case(AlwaysOnPolicy(), case)
    records = runner.store.read("plan_decision")
    assert decision.selected_alternative_ids
    assert len(records) == 1
    assert records[0]["baseline_id"] == "B0_ALWAYS_ON"
    assert records[0]["metadata"]["evaluation_mode"] == "FULL_STACK"
