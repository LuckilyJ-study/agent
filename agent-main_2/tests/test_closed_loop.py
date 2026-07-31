from __future__ import annotations

import unittest

from robot_agent.closed_loop import ClosedLoopAgent, SafetyStop


class FakeStateProvider:
    def observe(self):
        return {"frame": "camera"}

    def robot_state(self):
        return {"pose": [0, 0, 0]}


class FakeExecutor:
    def __init__(self, results):
        self.results = list(results)
        self.steps = []
        self.stopped = False

    def execute(self, step, observation):
        self.steps.append(step["action_type"])
        return self.results.pop(0)

    def stop(self):
        self.stopped = True


class Replanner:
    def __init__(self):
        self.context = None

    def create_plan(self, task):
        return [
            {"step_id": 1, "action_type": "move_to", "target": "object", "expected_result": "near object", "executor": "robot", "status": "pending"},
            {"step_id": 2, "action_type": "pick", "target": "object", "expected_result": "object held", "executor": "policy", "status": "pending"},
            {"step_id": 3, "action_type": "place", "target": "tray", "expected_result": "object on tray", "executor": "policy", "status": "pending"},
        ]

    def revise_from_failure(self, context):
        self.context = context
        return [
            {"step_id": 4, "action_type": "pick", "target": "object", "expected_result": "object held"},
            {"step_id": 5, "action_type": "place", "target": "tray", "expected_result": "object on tray"},
        ]


class ClosedLoopTests(unittest.TestCase):
    def test_routes_stops_and_replans_only_remaining_suffix(self):
        planner = Replanner()
        robot = FakeExecutor([{"status": "success", "reason": "OK"}])
        policy = FakeExecutor([
            {"status": "failed", "reason": "GRASP_FAILED"},
            {"status": "failed", "reason": "GRASP_FAILED"},
            {"status": "success", "reason": "OK"},
            {"status": "success", "reason": "OK"},
        ])
        memory = ClosedLoopAgent(planner, robot, policy, FakeStateProvider()).run("pick and place")

        self.assertEqual(robot.steps, ["move_to"])
        self.assertEqual(policy.steps, ["pick", "pick", "pick", "place"])
        self.assertTrue(policy.stopped)
        self.assertEqual([step["action_type"] for step in planner.context["completed_steps"]], ["move_to"])
        self.assertEqual(memory.current_step_id, 5)
        self.assertEqual(
            [record["status"] for record in memory.records],
            ["completed", "failed", "failed", "completed", "completed"],
        )

    def test_local_retry_does_not_call_planner(self):
        planner = Replanner()
        robot = FakeExecutor([{"status": "success", "reason": "OK"}])
        policy = FakeExecutor([
            {"status": "failed", "reason": "GRASP_FAILED"},
            {"status": "success", "reason": "OK"},
            {"status": "success", "reason": "OK"},
        ])
        memory = ClosedLoopAgent(planner, robot, policy, FakeStateProvider()).run("pick and place")
        self.assertIsNone(planner.context)
        self.assertEqual(memory.current_step_id, 3)

    def test_collision_stops_without_replanning(self):
        planner = Replanner()
        robot = FakeExecutor([{"status": "failed", "reason": "COLLISION_RISK"}])
        policy = FakeExecutor([])
        with self.assertRaises(SafetyStop):
            ClosedLoopAgent(planner, robot, policy, FakeStateProvider()).run("pick and place")
        self.assertTrue(robot.stopped)
        self.assertIsNone(planner.context)

    def test_rejects_wrong_executor_route(self):
        with self.assertRaises(ValueError):
            ClosedLoopAgent._normalize_plan([
                {"step_id": 1, "action_type": "pick", "target": "x", "executor": "robot", "expected_result": "held"}
            ])


if __name__ == "__main__":
    unittest.main()
