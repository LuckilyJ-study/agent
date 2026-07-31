from __future__ import annotations

import unittest

from robot_agent.action_executors import (
    DryRunPolicyBackend,
    DryRunRobotController,
    PolicyExecutor,
    PolicyRegistry,
    RobotPrimitiveExecutor,
)
from robot_agent.capabilities import CapabilityError, CapabilityRegistry
from robot_agent.closed_loop import PlaceholderActionVerifier
from robot_agent.perception import AgentStateProvider, NullPerceptionProvider
from robot_agent.planner import JsonSchemaPlannerMixin, RuleBasedPlanner
from robot_agent.runtime import build_agent_runtime
from robot_agent.gateway import Pi05ServiceGateway


class CapturingPlanner(JsonSchemaPlannerMixin):
    allowed_skills = ("pick", "place")

    def __init__(self):
        self.message = ""

    def _request_plan(self, user_message):
        self.message = user_message
        return [
            {
                "step_id": 1,
                "action_type": "pick",
                "target": "object",
                "executor": "policy",
                "expected_result": "object is held",
                "status": "pending",
                "parameters": {},
            }
        ]


class GenericAgentTests(unittest.TestCase):
    def test_unknown_capability_is_rejected_before_execution(self):
        with self.assertRaises(CapabilityError):
            CapabilityRegistry().normalize_plan(
                [
                    {
                        "step_id": 1,
                        "action_type": "unknown_action",
                        "target": "robot",
                        "expected_result": "done",
                    }
                ]
            )

    def test_robot_primitive_executor_calls_controller(self):
        controller = DryRunRobotController()
        executor = RobotPrimitiveExecutor(controller)
        result = executor.execute(
            {
                "step_id": 1,
                "action_type": "open_gripper",
                "target": "gripper",
                "expected_result": "gripper open",
            },
            {},
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(controller.get_state()["gripper"], "open")
        self.assertFalse(result["physical_result_verified"])

    def test_policy_executor_fails_closed_when_model_is_missing(self):
        controller = DryRunRobotController()
        state_provider = AgentStateProvider(NullPerceptionProvider(), controller)
        executor = PolicyExecutor(PolicyRegistry(), state_provider)
        step = CapabilityRegistry().normalize_step(
            {
                "step_id": 1,
                "action_type": "pick",
                "target": "object",
                "expected_result": "object held",
            },
            1,
        )
        result = executor.execute(step, state_provider.observe())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "POLICY_NOT_AVAILABLE")

    def test_registered_policy_runs_without_pi05_dependency(self):
        controller = DryRunRobotController()
        state_provider = AgentStateProvider(NullPerceptionProvider(), controller)
        policies = PolicyRegistry({"pick": DryRunPolicyBackend()})
        executor = PolicyExecutor(policies, state_provider)
        step = CapabilityRegistry().normalize_step(
            {
                "step_id": 1,
                "action_type": "pick",
                "target": "object",
                "expected_result": "object held",
            },
            1,
        )
        result = executor.execute(step, state_provider.observe())
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["details"]["policy_id"], "pick")

    def test_placeholder_verifier_marks_command_scope(self):
        result = PlaceholderActionVerifier().verify(
            {"available": False},
            {},
            {"status": "success", "physical_result_verified": False},
            "object held",
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["verification_scope"], "command")
        self.assertFalse(result["details"]["physical_result_verified"])

    def test_runtime_executes_generic_dry_run(self):
        runtime = build_agent_runtime(RuleBasedPlanner())
        result = runtime.agent.run_safe("move the arm left by 0.5 cm")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.memory.completed_steps[0]["executor"], "robot")
        self.assertEqual(
            result.memory.records[0]["verification"]["verification_scope"], "command"
        )

    def test_failure_replanning_prompt_requests_unfinished_suffix(self):
        planner = CapturingPlanner()
        planner.revise_from_failure(
            {
                "original_task": "pick and place",
                "completed_steps": [{"action_type": "move_to"}],
                "current_step": {"action_type": "pick"},
            }
        )
        self.assertIn("ONLY the unfinished suffix", planner.message)
        self.assertIn("Never repeat completed_steps", planner.message)

    def test_legacy_pi05_gateway_fails_closed_by_default(self):
        gateway = Pi05ServiceGateway(
            endpoint="http://127.0.0.1:9/predict",
            timeout_seconds=1,
        )
        result = gateway.execute(
            "pick object",
            {"id": 1, "skill": "pick", "target": "object", "expected_result": "held"},
            0,
            observation={},
        )
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["details"]["robot_execution"])


if __name__ == "__main__":
    unittest.main()
