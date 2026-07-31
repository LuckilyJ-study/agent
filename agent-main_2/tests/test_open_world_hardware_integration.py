from __future__ import annotations

import io
import json
import os
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import mock_open, patch

import run_agent_simulation as simulation_runner

from robot_agent.action_executors import PolicyRegistry
from robot_agent.closed_loop import PlaceholderActionVerifier
from robot_agent.perception import ScriptedPerceptionProvider
from robot_agent.physical_target_gate import PhysicalPerceptionConfigurationError
from robot_agent.policy_metadata import PolicyMetadata
from robot_agent.runtime import build_agent_runtime
from robot_agent.motion_safety import JointSafetyLimits, MotionSafetyLimits
from robot_agent.safety_monitor import SoftwareSafetyMonitor
from robot_agent.skill_grounding import SceneSkillPlanner


class _StaticHighLevelPlanner:
    last_goal = {
        "description": "the requested object is held",
        "conditions": [],
    }

    def __init__(self, target: str = "小球") -> None:
        self.target = target

    def create_plan_with_context(self, context):
        del context
        return [
            {
                "step_id": 1,
                "action_type": "pick",
                "target": self.target,
                "expected_result": "the requested object is held",
                "parameters": {},
                "max_attempts": 1,
            }
        ]


class _PhysicalController:
    hardware_ready = True

    def __init__(self) -> None:
        self.stop_calls = 0

    def get_state(self):
        return {
            "available": True,
            "source": "physical_test_controller",
            "connected": True,
            "gripper": "closed",
            "telemetry": {
                "joint_positions_rad": [0.0],
                "joint_velocities_rad_s": [0.0],
                "joint_accelerations_rad_s2": [0.0],
                "joint_torques_nm": [0.0],
            },
        }

    def stop(self) -> None:
        self.stop_calls += 1

    def move_home(self):
        return {
            "status": "success",
            "reason": "AT_HOME",
            "command_completed": True,
            "physical_result_verified": True,
        }


class _HardwareReadyPerception:
    hardware_ready = True
    supports_target_configuration = True
    supports_localization = True
    localization_modes = frozenset({"bbox_2d", "robot_base_xyz"})

    def __init__(self, *, localized: bool) -> None:
        self.localized = localized
        self.targets: list[str] = []
        self.observe_calls = 0

    def configure_targets(self, labels) -> None:
        self.targets = [str(label) for label in labels]

    def observe(self):
        self.observe_calls += 1
        observed_at = datetime.now(timezone.utc).isoformat()
        detection = {
            "entity_id": "small_ball",
            "label": "small ball",
            "confidence": 0.96,
            "bbox_xyxy": [10, 20, 40, 60],
        }
        if self.localized:
            detection.update(
                {
                    "position_xyz_m": [0.42, 0.08, 0.12],
                    "coordinate_frame": "robot_base",
                }
            )
        return {
            "available": True,
            "source": "physical_test_yolo",
            "timestamp": observed_at,
            "frames": [
                {
                    "timestamp": observed_at,
                    "detections": [detection],
                }
            ],
        }


class _CountingPhysicalPolicy:
    def __init__(self) -> None:
        self.calls = 0
        self.stop_calls = 0
        self.last_step = None

    def execute(self, step, observation, robot_state):
        del observation, robot_state
        self.calls += 1
        self.last_step = deepcopy(step)
        return {
            "status": "success",
            "reason": "PHYSICAL_POLICY_COMMAND_COMPLETED",
            "command_completed": True,
            "physical_result_verified": False,
        }

    def stop(self) -> None:
        self.stop_calls += 1


class _PhysicalVerifier:
    def __init__(self) -> None:
        self.calls = 0

    def verify(self, observation, robot_state, action, expected_result):
        del observation, robot_state, expected_result
        self.calls += 1
        success = action.get("status") == "success"
        return {
            "success": success,
            "error_type": "NONE" if success else "EXECUTION_FAILED",
            "confidence": 1.0,
            "verification_scope": "physical",
            "details": {"physical_result_verified": success},
        }


def _known_ball_scene():
    return {
        "scene": {
            "scene_id": "physical_ball_test",
            "skill_contracts": [],
            "goal_predicates": [],
        },
        "runtime": {
            "holding": None,
            "gripper": "unknown",
            "robot_location": "home",
        },
        "objects": {
            "small_ball": {
                "type": "movable_object",
                "aliases": ["ball", "small ball", "小球"],
                "location": "table",
            }
        },
    }


def _physical_runtime(*, localized: bool):
    scene = _known_ball_scene()
    planner = SceneSkillPlanner(
        _StaticHighLevelPlanner(),
        scene,
        allow_one_repair=False,
    )
    controller = _PhysicalController()
    perception = _HardwareReadyPerception(localized=localized)
    policy = _CountingPhysicalPolicy()
    policies = PolicyRegistry()
    policies.register(
        "pick",
        policy,
        PolicyMetadata(
            policy_id="pick",
            version="physical-test",
            action_type="pick",
            required_inputs=("perception", "robot_state"),
            supports_stop=True,
        ),
    )
    verifier = _PhysicalVerifier()
    joint_limits = JointSafetyLimits(
        position_min_rad=(-2.0,),
        position_max_rad=(2.0,),
        max_velocity_rad_s=(1.0,),
        max_acceleration_rad_s2=(2.0,),
        max_torque_nm=(10.0,),
        max_cumulative_motion_rad=(1.0,),
        joint_names=("test_joint",),
    )
    safety_monitor = SoftwareSafetyMonitor(
        limits=MotionSafetyLimits(
            joint_limits=joint_limits,
            require_joint_telemetry=True,
            hardware_approved=True,
            profile_name="physical-integration-test",
        )
    )
    runtime = build_agent_runtime(
        planner,
        controller=controller,
        perception=perception,
        policies=policies,
        verifier=verifier,
        initial_world_state=scene,
        dry_run=False,
        hardware_mode=True,
        safety_monitor=safety_monitor,
        max_replans=0,
    )
    return runtime, controller, perception, policy, verifier


class OpenWorldAndHardwareIntegrationTests(unittest.TestCase):
    def test_unknown_task_selects_open_world_simulation_scene(self):
        scene = simulation_runner._load_scene(None, "抓小球")

        self.assertEqual(scene["scene"]["scene_id"], "open_world_simulation")
        self.assertTrue(scene["scene"]["open_world_simulation"])
        self.assertEqual(scene["objects"], {})

    def test_bad_explicit_scene_fails_before_api_key_prompt(self):
        args = simulation_runner.build_parser().parse_args(
            ["--task", "抓小球", "--scene-file", "invalid-scene.json"]
        )
        prompt = patch.object(
            simulation_runner.getpass,
            "getpass",
            return_value="must-not-be-requested",
        )
        bad_scene = mock_open(read_data=json.dumps([]))

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(simulation_runner.Path, "open", bad_scene),
            prompt as prompt_mock,
            redirect_stdout(io.StringIO()),
        ):
            return_code = simulation_runner._run_qwen(args, "抓小球", None)

        self.assertEqual(return_code, 2)
        prompt_mock.assert_not_called()
        bad_scene.assert_called_once()

    def test_virtual_entity_patch_reaches_runtime_world_state_and_pick_completes(self):
        scene = simulation_runner._load_scene(None, "抓小球")
        semantic_scene = simulation_runner._planner_scene(scene)
        planner = SceneSkillPlanner(
            _StaticHighLevelPlanner(),
            semantic_scene,
            allow_one_repair=False,
            allow_virtual_entities=True,
        )
        runtime = build_agent_runtime(
            planner,
            initial_world_state=semantic_scene,
            dry_run=True,
        )

        result = runtime.agent.run_safe("抓小球")

        self.assertEqual(result.status, "completed")
        completed_step = result.memory.completed_steps[0]
        entity_id = completed_step["target"]
        self.assertRegex(entity_id, r"^virtual_entity_[0-9a-f]{12}$")
        world = result.memory.world_state["values"]
        self.assertTrue(world["objects"][entity_id]["virtual"])
        self.assertEqual(world["objects"][entity_id]["display_name"], "小球")
        self.assertEqual(world["objects"][entity_id]["location"], "held")
        self.assertEqual(world["runtime"]["holding"], entity_id)
        registrations = [
            event
            for event in result.memory.events
            if event["type"] == "world.virtual_entities_registered"
        ]
        self.assertEqual(registrations[0]["data"]["entity_ids"], [entity_id])

    def test_automatic_builtin_scene_can_extend_with_a_new_virtual_target(self):
        scene = simulation_runner._load_scene(None, "抓取刷子旁边的新盒子")
        self.assertEqual(
            scene["scene"]["scene_id"],
            "demo_kitchen_workspace",
        )
        self.assertTrue(simulation_runner._allows_virtual_entities(scene))
        semantic_scene = simulation_runner._planner_scene(scene)
        planner = SceneSkillPlanner(
            _StaticHighLevelPlanner("新盒子"),
            semantic_scene,
            allow_one_repair=False,
            allow_virtual_entities=True,
        )

        result = build_agent_runtime(
            planner,
            initial_world_state=semantic_scene,
            dry_run=True,
        ).agent.run_safe("抓取刷子旁边的新盒子")

        self.assertEqual(result.status, "completed")
        entity_id = result.memory.completed_steps[0]["target"]
        self.assertTrue(
            result.memory.world_state["values"]["objects"][entity_id]["virtual"]
        )

    def test_hardware_runtime_rejects_dry_run_and_scripted_provider(self):
        with self.assertRaisesRegex(
            ValueError,
            "hardware_mode cannot use dry_run",
        ):
            build_agent_runtime(
                _StaticHighLevelPlanner(),
                hardware_mode=True,
            )

        with self.assertRaises(PhysicalPerceptionConfigurationError):
            build_agent_runtime(
                _StaticHighLevelPlanner(),
                controller=_PhysicalController(),
                perception=ScriptedPerceptionProvider([]),
                verifier=PlaceholderActionVerifier(),
                dry_run=False,
                hardware_mode=True,
            )

    def test_hardware_gate_blocks_unlocalized_target_without_policy_call(self):
        runtime, _, perception, policy, verifier = _physical_runtime(
            localized=False
        )

        result = runtime.agent.run_safe("抓小球")

        self.assertEqual(result.status, "failed")
        self.assertEqual(policy.calls, 0)
        self.assertEqual(verifier.calls, 0)
        self.assertIn("small_ball", perception.targets)
        blocked = [
            event["data"]["error_type"]
            for event in result.memory.events
            if event["type"] == "monitor.blocked"
        ]
        self.assertEqual(blocked, ["TARGET_NOT_LOCALIZED"])
        self.assertNotIn(
            "step.routed",
            [event["type"] for event in result.memory.events],
        )

    def test_hardware_gate_allows_one_policy_call_with_fresh_localization(self):
        runtime, _, perception, policy, verifier = _physical_runtime(
            localized=True
        )

        result = runtime.agent.run_safe("抓小球")

        self.assertEqual(result.status, "completed")
        self.assertEqual(policy.calls, 1)
        self.assertEqual(verifier.calls, 1)
        self.assertIn("small_ball", perception.targets)
        grounding = policy.last_step["parameters"]["perception_grounding"]
        self.assertEqual(grounding["position_xyz_m"], [0.42, 0.08, 0.12])
        self.assertEqual(grounding["coordinate_frame"], "robot_base")
        self.assertEqual(
            result.memory.records[0]["verification"]["verification_scope"],
            "physical",
        )


if __name__ == "__main__":
    unittest.main()
