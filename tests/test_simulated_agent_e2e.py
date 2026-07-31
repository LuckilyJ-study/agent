from __future__ import annotations

import unittest

from robot_agent.runtime import build_agent_runtime
from robot_agent.simulation import build_pick_and_place_simulation


class SimulatedAgentEndToEndTests(unittest.TestCase):
    def test_pick_failure_is_retried_then_task_completes(self):
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
        self.assertGreaterEqual(simulation.pick_backend.stop_calls, 1)
        self.assertEqual(simulation.place_backend.calls, 1)
        self.assertEqual(result.memory.replan_count, 0)
        self.assertEqual(
            [record["status"] for record in result.memory.records],
            ["completed", "failed", "completed", "completed", "completed"],
        )
        self.assertEqual(
            result.memory.world_state["values"]["objects"]["demo_object"]["location"],
            "demo_tray",
        )
        self.assertIsNone(
            result.memory.world_state["values"]["gripper"]["holding"]
        )

    def test_repeated_failure_escalates_to_suffix_replanning(self):
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
        self.assertEqual(simulation.planner.replan_calls, 1)
        completed_before_replan = simulation.planner.last_failure_context[
            "completed_steps"
        ]
        self.assertEqual(
            [step["action_type"] for step in completed_before_replan],
            ["move_to"],
        )
        self.assertEqual(simulation.pick_backend.calls, 3)


if __name__ == "__main__":
    unittest.main()
