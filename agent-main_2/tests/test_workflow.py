from __future__ import annotations

import unittest
from unittest.mock import patch

from robot_agent.graph import build_robot_graph
from robot_agent.planner import OllamaPlanner, QwenApiPlanner, RuleBasedPlanner, create_default_planner
from robot_agent.run_cli import build_parser, run_workflow
from robot_agent.simple_workflow import run_simple_workflow
from robot_agent.skills import SkillRegistry, SkillValidationError
from robot_agent.state import PlanStep


class InvalidSkillPlanner:
    def create_plan(self, user_task: str) -> list[PlanStep]:
        return [
            {
                "id": 1,
                "skill": "open_unsafe_motor_control",
                "target": "arm",
                "expected_result": "not allowed",
            }
        ]


class OnePickPlanner:
    def create_plan(self, user_task: str) -> list[PlanStep]:
        return [
            {
                "id": 1,
                "skill": "pick",
                "target": "dough",
                "expected_result": "dough is held by the gripper",
            }
        ]


class ReplanningPickPlanner(OnePickPlanner):
    def revise_plan(
        self,
        user_task: str,
        previous_plan: list[PlanStep],
        failed_step: PlanStep,
        feedback: dict,
    ) -> list[PlanStep]:
        return [
            {
                "id": 1,
                "skill": "inspect",
                "target": "dough",
                "expected_result": "dough location is confirmed",
            },
            {
                "id": 2,
                "skill": "pick",
                "target": "dough",
                "expected_result": "dough is held by the gripper",
            },
        ]


class NonRecoverableGateway:
    def execute(self, task_text: str, step: PlanStep, retry_count: int) -> dict:
        return {
            "status": "failed",
            "reason": "COLLISION_RISK",
            "details": {"source": "simulated_safety_monitor"},
        }


class FailOnceGateway:
    def __init__(self) -> None:
        self.has_failed = False

    def execute(self, task_text: str, step: PlanStep, retry_count: int) -> dict:
        if not self.has_failed:
            self.has_failed = True
            return {
                "status": "failed",
                "reason": "COLLISION_RISK",
                "details": {"source": "simulated_safety_monitor"},
            }
        return {"status": "success", "reason": "OK", "details": {}}


class ObservationAwareGateway:
    def __init__(self) -> None:
        self.observation = None

    def execute(self, task_text: str, step: PlanStep, retry_count: int, observation=None) -> dict:
        self.observation = observation
        return {
            "status": "success",
            "reason": "OK",
            "details": {"action": {"type": "move", "target": step["target"]}},
        }


class RobotWorkflowTests(unittest.TestCase):
    def test_ollama_plan_json_is_parsed(self) -> None:
        plan = OllamaPlanner._parse_plan(
            """
            {
              "steps": [
                {
                  "id": 1,
                  "skill": "pick",
                  "target": "dough",
                  "expected_result": "dough is held by the gripper"
                }
              ]
            }
            """
        )

        self.assertEqual(plan[0]["skill"], "pick")
        self.assertEqual(plan[0]["target"], "dough")

    def test_unknown_skill_is_blocked_before_execution(self) -> None:
        graph = build_robot_graph(planner=InvalidSkillPlanner())

        result = graph.invoke({"user_task": "unsafe request", "history": [], "events": []})

        self.assertEqual(result["status"], "blocked_by_safety")
        self.assertEqual(result["events"][-1]["type"], "plan.rejected")

    def test_nonrecoverable_failure_requests_agent_replan(self) -> None:
        graph = build_robot_graph(planner=OnePickPlanner(), gateway=NonRecoverableGateway())

        result = graph.invoke(
            {
                "user_task": "pick dough",
                "max_retries": 2,
                "history": [],
                "events": [],
            }
        )

        self.assertEqual(result["status"], "needs_agent_replan")
        self.assertEqual(result["events"][-1]["type"], "agent.replan_requested")

    def test_replanning_planner_replaces_plan_and_continues(self) -> None:
        graph = build_robot_graph(planner=ReplanningPickPlanner(), gateway=FailOnceGateway())

        result = graph.invoke(
            {
                "user_task": "pick dough",
                "max_retries": 1,
                "max_replans": 1,
                "history": [],
                "events": [],
            }
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["replan_count"], 1)
        self.assertIn("agent.replanned", [event["type"] for event in result["events"]])

    def test_default_planner_factory_selects_ollama(self) -> None:
        with patch.dict("os.environ", {"ROBOT_AGENT_PLANNER_PROVIDER": "ollama"}, clear=False):
            planner = create_default_planner()
        self.assertIsInstance(planner, OllamaPlanner)

    def test_default_planner_factory_selects_qwen_api(self) -> None:
        env = {
            "ROBOT_AGENT_PLANNER_PROVIDER": "qwen_api",
            "QWEN_API_KEY": "test-key",
        }
        with patch.dict("os.environ", env, clear=False):
            planner = create_default_planner()
        self.assertIsInstance(planner, QwenApiPlanner)

    def test_rule_based_planner_creates_small_relative_motion(self) -> None:
        plan = RuleBasedPlanner().create_plan("机械臂向左移动0.5厘米")

        self.assertEqual(plan[0]["skill"], "move_relative")
        self.assertEqual(plan[0]["target"], "left 0.5 cm")

    def test_large_relative_motion_is_rejected(self) -> None:
        with self.assertRaises(SkillValidationError):
            SkillRegistry().validate_step(
                {
                    "id": 1,
                    "skill": "move_relative",
                    "target": "left 5 cm",
                    "expected_result": "arm moved left",
                }
            )

    def test_basic_motion_demo_flow_completes(self) -> None:
        graph = build_robot_graph(planner=RuleBasedPlanner())

        result = graph.invoke({"user_task": "机械臂向右移动5毫米", "history": [], "events": []})

        self.assertEqual(result["status"], "completed")
        self.assertIn("pi05.input_prepared", [event["type"] for event in result["events"]])

    def test_cli_accepts_natural_language_task(self) -> None:
        args = build_parser().parse_args(["--planner", "rule_based", "机械臂向左移动0.5厘米"])

        self.assertEqual(args.planner, "rule_based")
        self.assertEqual(" ".join(args.task), "机械臂向左移动0.5厘米")


    def test_cli_simple_engine_does_not_require_langgraph(self) -> None:
        result = run_workflow(
            user_task="move the arm left by 0.5 cm",
            planner=RuleBasedPlanner(),
            max_retries=0,
            max_replans=0,
            engine="simple",
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["plan"][0]["skill"], "move_relative")

    def test_simple_workflow_passes_structured_pi05_observation_to_gateway(self) -> None:
        gateway = ObservationAwareGateway()
        result = run_simple_workflow(
            initial_state={"user_task": "机械臂向左移动0.5厘米", "history": [], "events": []},
            planner=RuleBasedPlanner(),
            gateway=gateway,
        )

        self.assertIsNotNone(gateway.observation)
        self.assertEqual(result["status"], "completed")
        self.assertIn("pi05_action", result)
        self.assertEqual(result["pi05_action"]["type"], "move")


if __name__ == "__main__":
    unittest.main()
