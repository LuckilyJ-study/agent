from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from robot_agent.gateway import Pi05ServiceGateway
from robot_agent.observation import (
    CAMERA_NAMES,
    StaticObservationSource,
    build_structured_observation,
    summarize_observation,
)
from robot_agent.planner import RuleBasedPlanner
from robot_agent.robot_controller import PrintRobotController
from robot_agent.simple_workflow import run_simple_workflow


class ObservationBuilderTest(unittest.TestCase):
    def test_builds_service_shaped_observation(self) -> None:
        step = {
            "id": 1,
            "skill": "pick",
            "target": "dough",
            "expected_result": "dough is held by the gripper",
        }
        observation = build_structured_observation(step, "Pick up the dough.")
        self.assertEqual(observation["task_text"], "Pick up the dough.")
        self.assertEqual(len(observation["state"]), 8)
        self.assertEqual(sorted(observation["images"]), sorted(CAMERA_NAMES))
        for camera in CAMERA_NAMES:
            self.assertTrue(observation["images"][camera])

    def test_summary_hides_image_payloads(self) -> None:
        step = {"id": 1, "skill": "pick", "target": "dough", "expected_result": "held"}
        observation = build_structured_observation(step, "Pick up the dough.")
        summary = summarize_observation(observation)
        self.assertEqual(summary["state_dim"], 8)
        self.assertNotIn("images", summary)
        self.assertEqual(sorted(summary["image_bytes"]), sorted(CAMERA_NAMES))


class PrintRobotControllerTest(unittest.TestCase):
    def test_prints_each_control_step(self) -> None:
        controller = PrintRobotController()
        actions = [[0.1 * index] * 7 for index in range(50)]
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            summary = controller.execute_action_chunk(actions)
        output = buffer.getvalue()
        self.assertIn("step 01/50", output)
        self.assertIn("chunk finished: 50 actions executed", output)
        self.assertEqual(summary["steps_total"], 50)


class Pi05ServiceGatewayTest(unittest.TestCase):
    def _step(self) -> dict:
        return {
            "id": 1,
            "skill": "pick",
            "target": "dough",
            "expected_result": "dough is held by the gripper",
        }

    def test_uses_service_actions_when_available(self) -> None:
        gateway = Pi05ServiceGateway(endpoint="http://127.0.0.1:9/predict")
        service_actions = [[0.5] * 7 for _ in range(50)]

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"act": service_actions}).encode("utf-8")

        observation = build_structured_observation(self._step(), "Pick up the dough.")
        buffer = io.StringIO()
        with patch("robot_agent.gateway.urlopen", return_value=_Response()):
            with redirect_stdout(buffer):
                result = gateway.execute("Pick up the dough.", self._step(), 0, observation=observation)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["reason"], "PI05_ACTIONS_EXECUTED")
        self.assertEqual(result["details"]["action"]["chunk_size"], 50)
        self.assertEqual(result["details"]["action"]["first_action"], [0.5] * 7)

    def test_falls_back_to_stub_actions_when_service_is_down(self) -> None:
        gateway = Pi05ServiceGateway(
            endpoint="http://127.0.0.1:9/predict",
            timeout_seconds=1,
            allow_stub_actions=True,
        )
        observation = build_structured_observation(self._step(), "Pick up the dough.")
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            result = gateway.execute("Pick up the dough.", self._step(), 0, observation=observation)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["reason"], "PI05_STUB_ACTIONS_EXECUTED")
        self.assertEqual(result["details"]["action"]["chunk_size"], 50)
        self.assertEqual(result["details"]["action"]["first_action"], [0.0] * 7)
        self.assertIn("stub", result["details"]["policy_note"])


class ClosedLoopWorkflowTest(unittest.TestCase):
    def test_full_loop_completes_with_feedback(self) -> None:
        gateway = Pi05ServiceGateway(
            endpoint="http://127.0.0.1:9/predict",
            timeout_seconds=1,
            allow_stub_actions=True,
        )
        initial_state = {
            "user_task": "Make a pizza by picking dough and placing it on a tray.",
            "max_retries": 2,
            "max_replans": 1,
            "history": [],
            "events": [],
        }
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            result = run_simple_workflow(initial_state, RuleBasedPlanner(), gateway=gateway)
        self.assertEqual(result["status"], "completed")
        event_types = [event["type"] for event in result["events"]]
        self.assertIn("plan.created", event_types)
        self.assertIn("pi05.input_prepared", event_types)
        self.assertIn("execution.finished", event_types)
        self.assertIn("feedback.success", event_types)
        self.assertIn("task.completed", event_types)
        for event in result["events"]:
            self.assertLess(len(str(event["data"])), 4000, "events must not embed base64 images")


if __name__ == "__main__":
    unittest.main()
