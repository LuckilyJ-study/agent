"""End-to-end Robot Agent tests that require no robot, camera, or model server.

Run from this directory:
    python test_agent_simulation.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent
SOURCE_ROOT = WORKSPACE / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from robot_agent.runtime import build_agent_runtime
from robot_agent.simulation import build_pick_and_place_simulation


class RobotAgentSimulationTests(unittest.TestCase):
    def test_local_retry_then_complete(self) -> None:
        """The first grasp fails, local recovery retries it, and the task completes."""
        simulation = build_pick_and_place_simulation(pick_failures=1)
        runtime = build_agent_runtime(
            simulation.planner,
            controller=simulation.controller,
            policies=simulation.policies,
            dry_run=False,
        )

        result = runtime.agent.run_safe("Put the demo object on the demo tray.")

        self.assertEqual(result.status, "completed")
        self.assertEqual(simulation.pick_backend.calls, 2)
        self.assertEqual(result.memory.replan_count, 0)
        self.assertEqual(
            result.memory.world_state["values"]["objects"]["demo_object"]["location"],
            "demo_tray",
        )
        self.assertIsNone(
            result.memory.world_state["values"]["gripper"]["holding"]
        )

    def test_repeated_failure_triggers_suffix_replanning(self) -> None:
        """Two grasp failures escalate to Planner without repeating completed work."""
        simulation = build_pick_and_place_simulation(pick_failures=2)
        runtime = build_agent_runtime(
            simulation.planner,
            controller=simulation.controller,
            policies=simulation.policies,
            dry_run=False,
        )

        result = runtime.agent.run_safe("Put the demo object on the demo tray.")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.memory.replan_count, 1)
        self.assertEqual(simulation.pick_backend.calls, 3)
        completed_before_replan = simulation.planner.last_failure_context[
            "completed_steps"
        ]
        self.assertEqual(
            [step["action_type"] for step in completed_before_replan],
            ["move_to"],
        )

    def test_user_input_changes_object_and_destination(self) -> None:
        simulation = build_pick_and_place_simulation(pick_failures=0)
        runtime = build_agent_runtime(
            simulation.planner,
            controller=simulation.controller,
            policies=simulation.policies,
            dry_run=False,
        )

        result = runtime.agent.run_safe("把杯子放到托盘")

        self.assertEqual(result.status, "completed")
        self.assertEqual(simulation.planner.object_name, "杯子")
        self.assertEqual(simulation.planner.destination, "托盘")
        self.assertEqual(
            result.memory.world_state["values"]["objects"]["杯子"]["location"],
            "托盘",
        )

    def test_unsupported_user_task_is_rejected(self) -> None:
        simulation = build_pick_and_place_simulation()
        runtime = build_agent_runtime(
            simulation.planner,
            controller=simulation.controller,
            policies=simulation.policies,
            dry_run=False,
        )

        result = runtime.agent.run_safe("画一幅画")

        self.assertEqual(result.status, "failed")
        self.assertIn("supports pick-and-place tasks only", result.reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
