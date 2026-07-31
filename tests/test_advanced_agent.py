from __future__ import annotations

import tempfile
import unittest

from robot_agent.action_executors import (
    DryRunPolicyBackend,
    DryRunRobotController,
    PolicyExecutor,
    PolicyRegistry,
)
from robot_agent.capabilities import CapabilityError, CapabilityRegistry
from robot_agent.control import ExecutionControl
from robot_agent.perception import AgentStateProvider, NullPerceptionProvider
from robot_agent.persistence import JsonTaskStore
from robot_agent.planner import RuleBasedPlanner
from robot_agent.policy_metadata import PolicyMetadata
from robot_agent.runtime import build_agent_runtime
from robot_agent.safety_monitor import SoftwareSafetyMonitor
from robot_agent.task_verifier import TaskVerification


class StaticPlanner:
    def __init__(self, plan):
        self.plan = plan

    def create_plan(self, task):
        return self.plan

    def create_plan_with_context(self, context):
        return self.plan

    def revise_from_failure(self, context):
        return self.plan


class FailingTaskVerifier:
    def verify(self, original_task, completed_steps, world_state):
        return TaskVerification(False, "Goal condition is false.", "symbolic")


class InterruptOnceController(DryRunRobotController):
    def __init__(self):
        super().__init__()
        self.interrupted = False

    def move_relative(self, direction, distance_cm):
        if not self.interrupted:
            self.interrupted = True
            raise KeyboardInterrupt()
        return super().move_relative(direction, distance_cm)


class AdvancedAgentTests(unittest.TestCase):
    def test_dependency_cycle_is_rejected_before_execution(self):
        with self.assertRaises(CapabilityError):
            CapabilityRegistry().normalize_plan(
                [
                    {
                        "step_id": 1,
                        "action_type": "move_home",
                        "target": "home",
                        "expected_result": "home",
                        "depends_on": [2],
                    },
                    {
                        "step_id": 2,
                        "action_type": "open_gripper",
                        "target": "gripper",
                        "expected_result": "open",
                        "depends_on": [1],
                    },
                ]
            )

    def test_invalid_world_effect_is_rejected_before_execution(self):
        with self.assertRaises(CapabilityError):
            CapabilityRegistry().normalize_plan(
                [
                    {
                        "step_id": 1,
                        "action_type": "move_home",
                        "target": "home",
                        "expected_result": "home",
                        "effects": [{"path": "", "operation": "set", "value": True}],
                    }
                ]
            )

    def test_dependency_graph_and_world_effects(self):
        plan = [
            {
                "step_id": 1,
                "action_type": "open_gripper",
                "target": "gripper",
                "expected_result": "gripper open",
                "effects": [
                    {"path": "gripper.open", "operation": "set", "value": True}
                ],
            },
            {
                "step_id": 2,
                "action_type": "close_gripper",
                "target": "gripper",
                "expected_result": "gripper closed",
                "depends_on": [1],
                "conditions": [
                    {"path": "gripper.open", "operator": "eq", "value": True}
                ],
                "effects": [
                    {"path": "gripper.open", "operation": "set", "value": False}
                ],
            },
        ]
        runtime = build_agent_runtime(StaticPlanner(plan))
        result = runtime.agent.run_safe("exercise gripper")
        self.assertEqual(result.status, "completed")
        self.assertFalse(result.memory.world_state["values"]["gripper"]["open"])

    def test_false_condition_can_skip_optional_step(self):
        plan = [
            {
                "step_id": 1,
                "action_type": "open_gripper",
                "target": "gripper",
                "expected_result": "gripper open",
            },
            {
                "step_id": 2,
                "action_type": "move_home",
                "target": "home",
                "expected_result": "at home",
                "depends_on": [1],
                "conditions": [
                    {"path": "optional.enabled", "operator": "eq", "value": True}
                ],
                "on_condition_false": "skip",
            },
        ]
        result = build_agent_runtime(StaticPlanner(plan)).agent.run_safe("optional step")
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.memory.completed_steps), 1)
        self.assertEqual(result.memory.plan[1]["status"], "skipped")

    def test_cancel_is_persisted_as_structured_status(self):
        control = ExecutionControl()
        control.cancel()
        runtime = build_agent_runtime(RuleBasedPlanner(), control=control)
        result = runtime.agent.run_safe("move the arm left by 0.5 cm")
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.memory.status, "cancelled")

    def test_software_safety_monitor_blocks_forbidden_target(self):
        plan = [
            {
                "step_id": 1,
                "action_type": "move_to",
                "target": "forbidden_zone",
                "expected_result": "arrived",
            }
        ]
        runtime = build_agent_runtime(
            StaticPlanner(plan),
            safety_monitor=SoftwareSafetyMonitor(
                forbidden_targets={"forbidden_zone"}
            ),
        )
        result = runtime.agent.run_safe("unsafe move")
        self.assertEqual(result.status, "safety_stopped")
        self.assertIn("FORBIDDEN_TARGET", result.reason)

    def test_task_level_verifier_can_reject_plan_completion(self):
        runtime = build_agent_runtime(
            RuleBasedPlanner(), task_verifier=FailingTaskVerifier()
        )
        result = runtime.agent.run_safe("move the arm left by 0.5 cm")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "Goal condition is false.")

    def test_policy_metadata_rejects_missing_perception(self):
        controller = DryRunRobotController()
        provider = AgentStateProvider(NullPerceptionProvider(), controller)
        registry = PolicyRegistry()
        registry.register(
            "vision_pick",
            DryRunPolicyBackend(),
            PolicyMetadata(
                policy_id="vision_pick",
                action_type="pick",
                required_inputs=("robot_state", "perception"),
            ),
        )
        executor = PolicyExecutor(registry, provider)
        result = executor.execute(
            {
                "step_id": 1,
                "action_type": "pick",
                "target": "object",
                "expected_result": "held",
                "policy_id": "vision_pick",
            },
            provider.observe(),
        )
        self.assertEqual(result["reason"], "POLICY_INPUTS_UNAVAILABLE")
        self.assertEqual(result["details"]["missing_inputs"], ["perception"])

    def test_persisted_running_step_can_resume_after_process_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonTaskStore(directory)
            task = "move the arm left by 0.5 cm"
            task_id = "resume-test"
            first = build_agent_runtime(
                RuleBasedPlanner(),
                controller=InterruptOnceController(),
                task_store=store,
            )
            with self.assertRaises(KeyboardInterrupt):
                first.agent.run(task, task_id=task_id)

            snapshot = store.load(task_id)
            self.assertEqual(snapshot["plan"][0]["status"], "running")

            resumed = build_agent_runtime(
                RuleBasedPlanner(),
                controller=DryRunRobotController(),
                task_store=store,
            )
            result = resumed.agent.run_safe(task, task_id=task_id, resume=True)
            self.assertEqual(result.status, "completed")
            self.assertIn(
                "task.resumed",
                [event["type"] for event in result.memory.events],
            )


if __name__ == "__main__":
    unittest.main()
