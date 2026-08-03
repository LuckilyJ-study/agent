from __future__ import annotations

import unittest
import json
from unittest.mock import patch

from robot_agent.gateway import Pi05ServiceGateway
from robot_agent.libero_integration import (
    LIBERO_ACTION_SCHEMA,
    LiberoActionChunkController,
    LiberoPerceptionProvider,
    LiberoServiceError,
    LiberoTaskVerifier,
    build_libero_action_guard,
    validate_libero_action_schema,
)


class _FakeClient:
    def __init__(self) -> None:
        self.success = False
        self.chunks = []
        self.stop_calls = 0

    def health(self):
        return {
            "status": "ok",
            "action_schema": {"id": LIBERO_ACTION_SCHEMA, "dimension": 7},
        }

    def observe(self):
        return {
            "available": True,
            "source": "libero_bridge",
            "success": self.success,
            "state": [0.0] * 8,
            "images": {"primary": "abc", "secondary": "abc", "wrist": "abc"},
            "robot_state": {"available": True, "source": "libero_bridge"},
        }

    def execute_action_chunk(self, actions):
        self.chunks.append(actions)
        return {
            "status": "success",
            "reason": "ACTION_CHUNK_EXECUTED",
            "steps_executed": len(actions),
        }

    def stop(self):
        self.stop_calls += 1
        return {"status": "stopping"}


class LiberoIntegrationTests(unittest.TestCase):
    def test_perception_reads_cached_bridge_observation(self) -> None:
        provider = LiberoPerceptionProvider(_FakeClient())
        observation = provider.observe()
        self.assertTrue(observation["available"])
        self.assertEqual(len(observation["state"]), 8)
        self.assertFalse(provider.hardware_ready)

    def test_controller_fails_closed_until_schema_is_confirmed(self) -> None:
        client = _FakeClient()
        controller = LiberoActionChunkController(client)
        result = controller.execute_action_chunk([[0.0] * 7])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            result["reason"], "LIBERO_ACTION_SCHEMA_CONFIRMATION_REQUIRED"
        )
        self.assertEqual(client.chunks, [])

    def test_controller_executes_after_schema_confirmation(self) -> None:
        client = _FakeClient()
        controller = LiberoActionChunkController(
            client,
            action_schema_confirmed=True,
        )
        result = controller.execute_action_chunk([[0.0] * 7, [1.0] * 7])
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["steps_executed"], 2)
        self.assertEqual(controller.get_action_state(), [1.0] * 7)

    def test_controller_blocks_out_of_range_action(self) -> None:
        client = _FakeClient()
        controller = LiberoActionChunkController(
            client,
            action_schema_confirmed=True,
        )
        result = controller.execute_action_chunk([[1.1] * 7])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "ACTION_VALUE_LIMIT_EXCEEDED")
        self.assertEqual(client.chunks, [])

    def test_pi05_gateway_propagates_controller_rejection(self) -> None:
        controller = LiberoActionChunkController(_FakeClient())
        gateway = Pi05ServiceGateway(robot=controller)

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"act": [[0.0] * 7]}).encode("utf-8")

        step = {
            "id": 1,
            "skill": "pick",
            "target": "object",
            "expected_result": "object is held",
        }
        with patch("robot_agent.gateway.urlopen", return_value=_Response()):
            result = gateway.execute("Pick object", step, 0, observation={})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            result["reason"], "LIBERO_ACTION_SCHEMA_CONFIRMATION_REQUIRED"
        )

    def test_schema_mismatch_is_rejected(self) -> None:
        with self.assertRaises(LiberoServiceError):
            validate_libero_action_schema(
                {"action_schema": {"id": "joint_positions", "dimension": 7}}
            )

    def test_simulation_guard_accepts_full_normalized_transition(self) -> None:
        guard = build_libero_action_guard(max_chunk_size=2)
        result = guard.check([[-1.0] * 7, [1.0] * 7])
        self.assertTrue(result.safe)
        self.assertFalse(guard.limits.hardware_approved)

    def test_task_verifier_uses_bddl_success(self) -> None:
        client = _FakeClient()
        verifier = LiberoTaskVerifier(client)
        failed = verifier.verify("task", [], None)
        self.assertFalse(failed.success)
        client.success = True
        succeeded = verifier.verify("task", [], None)
        self.assertTrue(succeeded.success)
        self.assertEqual(succeeded.verification_scope, "simulation")


if __name__ == "__main__":
    unittest.main()
