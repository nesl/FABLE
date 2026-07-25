from __future__ import annotations

import unittest
from datetime import timedelta

from fable.common.examples import BASE_TIME
from fable.common.ids import uuid7
from fable.llm import (
    BranchPriorityAdjustment,
    CheckpointAdvisorHint,
    CheckpointAdvisorRequest,
    CheckpointHintValidator,
)
from fable.semantic import (
    EventRequestCompiler,
    InterpretedEventRequest,
    RequestCompilationMode,
    RequestCompileError,
)


class _FakeInterpreter:
    def interpret(self, text: str, *, available_family_ids: tuple[str, ...]):
        assert "robbery" in available_family_ids
        return InterpretedEventRequest(
            family_id="robbery",
            rationale="Mapped the requested incident to the authored robbery family.",
            inferred_fields=("family_id",),
        )


class RequestCompilerTests(unittest.TestCase):
    def test_detect_a_convoy_compiles_to_authored_graph(self) -> None:
        result = EventRequestCompiler().compile("detect a convoy")
        self.assertEqual(result.family_id, "convoy")
        self.assertEqual(result.mode, RequestCompilationMode.NATURAL_LANGUAGE_ALIAS)
        self.assertEqual(result.graph.name, "Pass-follow-clear convoy")
        self.assertTrue(result.warnings)

    def test_unknown_free_form_request_requires_typed_interpreter(self) -> None:
        with self.assertRaises(RequestCompileError):
            EventRequestCompiler().compile("look for suspicious coordinated driving")

    def test_interpreter_can_only_select_an_authored_family(self) -> None:
        result = EventRequestCompiler(interpreter=_FakeInterpreter()).compile(
            "look for a store incident with a threat or alarm"
        )
        self.assertEqual(result.family_id, "robbery")
        self.assertEqual(result.mode, RequestCompilationMode.LLM_ASSISTED)
        self.assertIn("family_id", result.inferred_fields)


class CheckpointHintTests(unittest.TestCase):
    def test_validator_accepts_only_active_replayable_branch_hints(self) -> None:
        request = CheckpointAdvisorRequest(
            request_id="request_1",
            graph_hash="graph_hash",
            hypothesis_id=uuid7(now_ms=100),
            hypothesis_version=2,
            frontier_id=uuid7(now_ms=101),
            checkpoint_ids=(uuid7(now_ms=102),),
            active_graph_node_ids=("node_a", "node_b"),
            eligible_branch_ids=("branch_a", "branch_b"),
            replayable_branch_ids=("branch_a",),
            requested_at=BASE_TIME,
        )
        hint = CheckpointAdvisorHint(
            ordered_branch_ids=("branch_a", "branch_b", "branch_unknown"),
            priority_adjustments=(
                BranchPriorityAdjustment(
                    branch_id="branch_a",
                    adjustment=1,
                    reason="Spatial evidence is consistent with this branch.",
                ),
                BranchPriorityAdjustment(
                    branch_id="branch_b",
                    adjustment=-1,
                    reason="Defer this replayable-looking branch.",
                ),
            ),
            explanation="Bounded branch ordering only.",
            expires_at=BASE_TIME + timedelta(seconds=30),
        )
        validated = CheckpointHintValidator().validate(
            request=request,
            hint=hint,
            now=BASE_TIME,
        )
        self.assertEqual(validated.accepted_branch_order, ("branch_a",))
        self.assertEqual(
            tuple(item.branch_id for item in validated.accepted_adjustments),
            ("branch_a",),
        )
        self.assertGreaterEqual(len(validated.ignored_reasons), 2)

    def test_expired_hint_is_rejected(self) -> None:
        request = CheckpointAdvisorRequest(
            request_id="request_1",
            graph_hash="graph_hash",
            hypothesis_id=uuid7(now_ms=200),
            hypothesis_version=0,
            frontier_id=uuid7(now_ms=201),
            checkpoint_ids=(uuid7(now_ms=202),),
            active_graph_node_ids=("node_a",),
            requested_at=BASE_TIME,
        )
        hint = CheckpointAdvisorHint(
            expires_at=BASE_TIME - timedelta(seconds=1),
        )
        with self.assertRaises(ValueError):
            CheckpointHintValidator().validate(
                request=request,
                hint=hint,
                now=BASE_TIME,
            )


if __name__ == "__main__":
    unittest.main()
