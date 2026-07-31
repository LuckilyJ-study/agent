from __future__ import annotations

import json
import threading
import time
import unittest
from copy import deepcopy
from unittest.mock import patch

from robot_agent.closed_loop import ClosedLoopAgent, RecoveryPolicy
from robot_agent.demo_scenes import DEMO_WORKSPACE_SCENE
from robot_agent.monitor import StructuredActionMonitor
from robot_agent.planner import PlannerServiceError, QwenApiPlanner
from robot_agent.runtime import build_agent_runtime
from robot_agent.skill_grounding import SceneSkillPlanner
from robot_agent.step_protocol import build_plan_json_schema


class _StaticStateProvider:
    def __init__(self, observation=None, robot_state=None):
        self._observation = observation or {
            "available": False,
            "source": "test",
            "frames": [],
        }
        self._robot_state = robot_state or {
            "available": True,
            "source": "test_robot",
            "gripper": "unknown",
        }

    def observe(self):
        return deepcopy(self._observation)

    def robot_state(self):
        return deepcopy(self._robot_state)


class _StaticPlanner:
    def __init__(self, plan):
        self.plan = deepcopy(plan)

    def create_plan(self, task):
        return deepcopy(self.plan)


class _RecordingExecutor:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.executed = []
        self.stop_calls = 0

    def execute(self, step, observation):
        self.executed.append(deepcopy(step))
        if self.results:
            return deepcopy(self.results.pop(0))
        return {
            "status": "success",
            "reason": "TEST_COMMAND_COMPLETED",
            "command_completed": True,
            "physical_result_verified": False,
        }

    def stop(self):
        self.stop_calls += 1


class _CooperativeBlockingExecutor(_RecordingExecutor):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.released = threading.Event()

    def execute(self, step, observation):
        self.executed.append(deepcopy(step))
        self.started.set()
        self.released.wait(5.0)
        return {
            "status": "success",
            "reason": "RELEASED",
            "command_completed": True,
            "physical_result_verified": False,
        }

    def stop(self):
        self.stop_calls += 1
        self.released.set()


class _CapturingReplanner:
    def __init__(self):
        self.context = None

    def create_plan(self, task):
        return [
            {
                "step_id": 1,
                "action_type": "pick",
                "target": "dough",
                "expected_result": "dough is held",
                "timeout_seconds": 1,
                "max_attempts": 2,
            }
        ]

    def revise_from_failure(self, context):
        self.context = deepcopy(context)
        return [
            {
                "step_id": 2,
                "action_type": "move_home",
                "target": "home",
                "expected_result": "robot is at its safe home pose",
                "timeout_seconds": 1,
                "max_attempts": 1,
            }
        ]


class _RepairingHighLevelPlanner:
    def __init__(self):
        self.last_goal = {
            "description": "the brush is held outside the drawer",
            "conditions": [
                {
                    "path": "runtime.holding",
                    "operator": "eq",
                    "value": "brush",
                }
            ],
        }
        self.repair_calls = 0

    def create_plan_with_context(self, context):
        return [
            {
                "step_id": 1,
                "action_type": "pick",
                "target": "刷子",
                "expected_result": "brush is held",
                "parameters": {},
            }
        ]

    def repair_rejected_plan(
        self, context, rejected_plan, validation_error, *, suffix_only=False
    ):
        self.repair_calls += 1
        return [
            {
                "step_id": 1,
                "action_type": "manipulate",
                "target": "抽屉",
                "expected_result": "drawer is open",
                "parameters": {"operation": "open"},
                "effects": [
                    {
                        "path": "objects.pizza.sauce_applied",
                        "operation": "set",
                        "value": True,
                    }
                ],
            },
            {
                "step_id": 2,
                "action_type": "pick",
                "target": "刷子",
                "expected_result": "brush is held outside the drawer",
                "parameters": {},
            },
        ]


class TargetArchitectureTests(unittest.TestCase):
    def test_plan_schema_uses_runtime_whitelist_and_hides_local_routing(self):
        schema = build_plan_json_schema(("zeta_skill", "pick", "alpha_skill", "pick"))
        step_schema = schema["properties"]["steps"]["items"]
        properties = step_schema["properties"]

        self.assertEqual(
            properties["action_type"]["enum"],
            ["alpha_skill", "pick", "zeta_skill"],
        )
        self.assertNotIn("executor", properties)
        self.assertNotIn("policy_id", properties)
        self.assertNotIn("executor", step_schema["required"])
        self.assertNotIn("policy_id", step_schema["required"])

    def test_qwen_cannot_supply_trusted_grounding_or_motor_parameters(self):
        base_step = {
            "step_id": 1,
            "action_type": "pick",
            "target": "ball",
            "expected_result": "ball is held",
            "status": "pending",
        }
        for parameters in (
            {"perception_grounding": {"position_xyz_m": [0.4, 0.1, 0.2]}},
            {"position_xyz_m": [0.4, 0.1, 0.2]},
            {"target_aliases": ["different object"]},
        ):
            with self.subTest(parameters=parameters):
                document = {
                    "goal": {"description": "ball is held", "conditions": []},
                    "steps": [{**base_step, "parameters": parameters}],
                }
                with self.assertRaises(PlannerServiceError):
                    QwenApiPlanner._parse_plan_document(
                        json.dumps(document),
                        ("pick",),
                    )

    def test_qwen_plan_without_executor_is_routed_by_trusted_router(self):
        raw_steps = [
            {
                "step_id": 1,
                "action_type": "pick",
                "target": "dough",
                "expected_result": "dough is held",
                "status": "pending",
                "parameters": {},
            },
            {
                "step_id": 2,
                "action_type": "move_home",
                "target": "home",
                "expected_result": "robot is at its safe home pose",
                "status": "pending",
                "parameters": {},
            },
        ]
        self.assertTrue(all("executor" not in step for step in raw_steps))
        qwen_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "goal": {
                                    "description": "hold the dough, then return home",
                                    "conditions": [],
                                },
                                "steps": raw_steps,
                            }
                        )
                    }
                }
            ]
        }
        planner = QwenApiPlanner(
            api_key="offline-test-key",
            allowed_skills=("pick", "move_home"),
            review_plans=False,
        )

        with patch.object(
            planner, "_post_json", return_value=qwen_response
        ) as mock_post:
            result = build_agent_runtime(planner).agent.run_safe(
                "Pick up the dough and return home."
            )

        self.assertEqual(result.status, "completed")
        mock_post.assert_called_once()
        routed = [
            event["data"]
            for event in result.memory.events
            if event["type"] == "step.routed"
        ]
        self.assertEqual(
            [(event["step_id"], event["executor"]) for event in routed],
            [(1, "policy"), (2, "robot")],
        )
        self.assertEqual(
            [event["route"] for event in routed],
            ["policy:pick", "robot_control"],
        )

    def test_unknown_skill_is_rejected_before_any_executor_runs(self):
        planner = _StaticPlanner(
            [
                {
                    "step_id": 1,
                    "action_type": "fly_to_moon",
                    "target": "moon",
                    "expected_result": "robot is on the moon",
                }
            ]
        )
        robot = _RecordingExecutor()
        policy = _RecordingExecutor()
        result = ClosedLoopAgent(
            planner,
            robot,
            policy,
            _StaticStateProvider(),
        ).run_safe("Fly to the moon.")

        self.assertEqual(result.status, "failed")
        self.assertIn("not registered", result.reason)
        self.assertEqual(robot.executed, [])
        self.assertEqual(policy.executed, [])
        self.assertIn(
            "plan.rejected",
            [event["type"] for event in result.memory.events],
        )
        self.assertNotIn(
            "step.started",
            [event["type"] for event in result.memory.events],
        )

    def test_structured_monitor_classifies_temporal_failures(self):
        monitor = StructuredActionMonitor(target_lost_frames=3)
        step = {
            "step_id": 1,
            "action_type": "pick",
            "target": "dough",
            "expected_result": "dough is held",
        }

        target_lost = monitor.before_action(
            step,
            {
                "available": True,
                "frames": [
                    {
                        "detections": [{"label": "dough", "confidence": 0.9}],
                        "signals": {"target_visible": True},
                    },
                    {"detections": [], "signals": {"target_visible": False}},
                    {"detections": [], "signals": {"target_visible": False}},
                    {"detections": [], "signals": {"target_visible": False}},
                ],
            },
            {},
        )
        self.assertIsNotNone(target_lost)
        self.assertEqual(target_lost["error_type"], "TARGET_LOST")

        grasp_failed = monitor.verify(
            {
                "available": True,
                "frames": [
                    {
                        "detections": [
                            {
                                "label": "dough",
                                "confidence": 0.9,
                                "following_gripper": False,
                            }
                        ],
                        "signals": {
                            "target_visible": True,
                            "grasp_succeeded": False,
                            "gripper_lifted": True,
                        },
                    }
                ],
            },
            {"gripper": "closed", "lifting": True},
            {"status": "success", "_monitor_step": step},
            "dough is held",
        )
        self.assertEqual(grasp_failed["error_type"], "GRASP_FAILED")

        object_dropped = monitor.during_action(
            step,
            {
                "available": True,
                "frames": [
                    {
                        "detections": [
                            {
                                "label": "dough",
                                "confidence": 0.9,
                                "following_gripper": True,
                            }
                        ]
                    },
                    {
                        "detections": [
                            {
                                "label": "dough",
                                "confidence": 0.9,
                                "following_gripper": False,
                                "falling": True,
                            }
                        ]
                    },
                ],
            },
            {"gripper": "closed"},
        )
        self.assertIsNotNone(object_dropped)
        self.assertEqual(object_dropped["error_type"], "OBJECT_DROPPED")

    def test_recovery_policy_separates_retry_replan_and_stop(self):
        recovery = RecoveryPolicy(max_local_retries=1)

        self.assertEqual(recovery.decide("GRASP_FAILED", 1).action, "retry")
        self.assertEqual(recovery.decide("OBJECT_DROPPED", 1).action, "replan")
        self.assertEqual(recovery.decide("COLLISION_RISK", 1).action, "stop")

    def test_action_timeout_calls_stop_without_waiting_for_blocked_executor(self):
        planner = _StaticPlanner(
            [
                {
                    "step_id": 1,
                    "action_type": "move_home",
                    "target": "home",
                    "expected_result": "robot is at home",
                    "timeout_seconds": 0.05,
                    "max_attempts": 1,
                }
            ]
        )
        robot = _CooperativeBlockingExecutor()
        policy = _RecordingExecutor()
        agent = ClosedLoopAgent(
            planner,
            robot,
            policy,
            _StaticStateProvider(),
            monitor_interval_seconds=0.005,
            stop_grace_seconds=0.2,
        )

        started = time.monotonic()
        result = agent.run_safe("Return home.")
        elapsed = time.monotonic() - started

        self.assertEqual(result.status, "safety_stopped")
        self.assertTrue(robot.started.is_set())
        self.assertGreaterEqual(robot.stop_calls, 1)
        self.assertLess(elapsed, 1.0)
        failures = [
            event["data"].get("error_type")
            for event in result.memory.events
            if event["type"] == "step.failed"
        ]
        self.assertEqual(failures, ["ACTION_TIMEOUT"])

    def test_replanning_context_contains_only_compact_observation_data(self):
        image_secret = "RAW_IMAGE_PAYLOAD_MUST_NOT_LEAK"
        frame_secret = "RAW_FRAME_PAYLOAD_MUST_NOT_LEAK"
        observation = {
            "available": True,
            "source": "simulated_yolo_world",
            "images": {"wrist": image_secret},
            "frames": [
                {
                    "raw_pixels": frame_secret,
                    "detections": [
                        {
                            "label": "dough",
                            "confidence": 0.91,
                            "track_id": 7,
                        }
                    ],
                    "signals": {"target_visible": True},
                }
            ],
        }
        planner = _CapturingReplanner()
        robot = _RecordingExecutor()
        policy = _RecordingExecutor(
            [
                {
                    "status": "failed",
                    "reason": "OBJECT_DROPPED",
                    "command_completed": False,
                    "physical_result_verified": False,
                    "images": {"policy_camera": image_secret},
                    "frames": [{"raw_pixels": frame_secret}],
                }
            ]
        )
        result = ClosedLoopAgent(
            planner,
            robot,
            policy,
            _StaticStateProvider(observation=observation),
        ).run_safe("Pick up the dough.")

        self.assertEqual(result.status, "completed")
        self.assertIsNotNone(planner.context)
        serialized = json.dumps(planner.context, ensure_ascii=False, default=str)
        self.assertNotIn(image_secret, serialized)
        self.assertNotIn(frame_secret, serialized)
        self._assert_no_raw_media_keys(planner.context)
        self.assertEqual(
            planner.context["current_observation"]["frame_count"],
            1,
        )
        self.assertEqual(
            planner.context["current_observation"]["detections"][0]["label"],
            "dough",
        )

    def test_semantic_skill_contract_repairs_invalid_order_without_task_recipe(self):
        planner = _RepairingHighLevelPlanner()
        grounded = SceneSkillPlanner(planner, DEMO_WORKSPACE_SCENE)
        context = {
            "original_task": "从抽屉中拿出刷子",
            "world_state": {
                "version": 0,
                "values": deepcopy(DEMO_WORKSPACE_SCENE),
            },
            "capabilities": [],
        }

        plan = grounded.create_plan_with_context(context)

        self.assertEqual(planner.repair_calls, 1)
        self.assertEqual(
            [(step["action_type"], step["target"]) for step in plan],
            [("manipulate", "drawer"), ("pick", "brush")],
        )
        self.assertEqual(
            plan[0]["effects"],
            [
                {
                    "path": "objects.drawer.state",
                    "operation": "set",
                    "value": "open",
                }
            ],
        )
        self.assertNotIn(
            {
                "path": "objects.pizza.sauce_applied",
                "operation": "set",
                "value": True,
            },
            plan[0]["effects"],
        )

    def test_generic_pick_place_grounding_uses_scene_entities_not_kitchen_names(self):
        scene = {
            "scene": {"skill_contracts": [], "goal_predicates": []},
            "runtime": {"holding": None},
            "objects": {
                "sample_vial": {
                    "type": "container",
                    "aliases": ["vial", "样品瓶"],
                    "location": "bench",
                },
                "centrifuge_rack": {
                    "type": "placement_target",
                    "aliases": ["rack", "离心管架"],
                },
            },
        }

        class Planner:
            last_goal = {"description": "vial is on rack", "conditions": []}

            def create_plan_with_context(self, context):
                return [
                    {
                        "step_id": 1,
                        "action_type": "pick",
                        "target": "样品瓶",
                        "expected_result": "vial held",
                    },
                    {
                        "step_id": 2,
                        "action_type": "place",
                        "target": "离心管架",
                        "expected_result": "vial on rack",
                    },
                ]

        plan = SceneSkillPlanner(Planner(), scene).create_plan_with_context(
            {
                "original_task": "把样品瓶放到离心管架",
                "world_state": {"version": 0, "values": deepcopy(scene)},
                "capabilities": [],
            }
        )

        self.assertEqual(
            [step["target"] for step in plan],
            ["sample_vial", "centrifuge_rack"],
        )
        self.assertEqual(
            plan[1]["effects"][-1],
            {
                "path": "objects.sample_vial.location",
                "operation": "set",
                "value": "centrifuge_rack",
            },
        )

    def _assert_no_raw_media_keys(self, value):
        if isinstance(value, dict):
            self.assertTrue(
                {"images", "frames"}.isdisjoint(value),
                "Replanner context must not contain raw images or frame arrays.",
            )
            for child in value.values():
                self._assert_no_raw_media_keys(child)
        elif isinstance(value, list):
            for child in value:
                self._assert_no_raw_media_keys(child)


if __name__ == "__main__":
    unittest.main()
