from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_agent.action_safety import ActionChunkGuard, ActionChunkSafetyLimits
from robot_agent.capabilities import CapabilityError, CapabilityRegistry
from robot_agent.gateway import Pi05ServiceGateway
from robot_agent.motion_safety import JointSafetyLimits, MotionSafetyLimits
from robot_agent.safety_monitor import SoftwareSafetyMonitor
from robot_agent.safety_config import load_safety_profiles


def _z_rotation(degrees: float) -> list[float]:
    half_angle = math.radians(degrees) / 2.0
    return [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]


def _pose_step(degrees: float) -> dict:
    return {
        "step_id": 1,
        "action_type": "move_to_pose",
        "target": f"pose_{degrees:g}",
        "expected_result": "pose reached",
        "parameters": {
            "position_xyz_m": [0.3, 0.0, 0.35],
            "orientation_xyzw": _z_rotation(degrees),
            "coordinate_frame": "robot_base",
        },
    }


def _robot_state(degrees: float = 0.0) -> dict:
    return {
        "connected": True,
        "cartesian_pose": {
            "position_xyz_m": [0.3, 0.0, 0.35],
            "orientation_xyzw": _z_rotation(degrees),
            "coordinate_frame": "robot_base",
        },
    }


class MotionSafetyTests(unittest.TestCase):
    def test_example_safety_profile_loads_but_is_not_hardware_approved(self) -> None:
        profile_path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "safety_profile.example.json"
        )
        profiles = load_safety_profiles(profile_path)
        self.assertFalse(profiles.motion.hardware_approved)
        self.assertFalse(profiles.policy_action.hardware_approved)
        self.assertEqual(profiles.policy_action.action_dim, 7)

    def test_capability_rejects_non_unit_and_non_finite_quaternions(self) -> None:
        registry = CapabilityRegistry()
        for quaternion in ([100.0, 0.0, 0.0, 0.0], [math.nan, 0.0, 0.0, 1.0]):
            step = _pose_step(0.0)
            step["parameters"]["orientation_xyzw"] = quaternion
            with self.assertRaisesRegex(CapabilityError, "unit quaternion"):
                registry.normalize_step(step, 1)

    def test_capability_forces_shortest_path_and_bounded_angular_speed(self) -> None:
        normalized = CapabilityRegistry().normalize_step(_pose_step(10.0), 1)
        self.assertEqual(normalized["parameters"]["rotation_path"], "shortest")
        self.assertEqual(
            normalized["parameters"]["max_angular_speed_rad_s"],
            0.5,
        )

        unsafe = _pose_step(10.0)
        unsafe["parameters"]["rotation_path"] = "long"
        with self.assertRaisesRegex(CapabilityError, "shortest"):
            CapabilityRegistry().normalize_step(unsafe, 1)

    def test_monitor_blocks_large_single_rotation(self) -> None:
        monitor = SoftwareSafetyMonitor(
            limits=MotionSafetyLimits(max_orientation_step_degrees=30.0)
        )
        step = CapabilityRegistry().normalize_step(_pose_step(90.0), 1)
        result = monitor.before_action(step, _robot_state())
        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "ORIENTATION_STEP_LIMIT_EXCEEDED")

    def test_monitor_blocks_cumulative_rotation_and_home_resets_it(self) -> None:
        monitor = SoftwareSafetyMonitor(
            limits=MotionSafetyLimits(
                max_orientation_step_degrees=40.0,
                max_cumulative_orientation_degrees=60.0,
            )
        )
        registry = CapabilityRegistry()

        first = registry.normalize_step(_pose_step(30.0), 1)
        self.assertTrue(monitor.before_action(first, _robot_state(0.0)).safe)
        monitor.after_action(first, {"status": "success"}, _robot_state(30.0))

        second = registry.normalize_step(_pose_step(60.0), 2)
        self.assertTrue(monitor.before_action(second, _robot_state(30.0)).safe)
        monitor.after_action(second, {"status": "success"}, _robot_state(60.0))

        third = registry.normalize_step(_pose_step(90.0), 3)
        result = monitor.before_action(third, _robot_state(60.0))
        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "ORIENTATION_CUMULATIVE_LIMIT_EXCEEDED")

        home = {
            "step_id": 4,
            "action_type": "move_home",
            "target": "home",
            "expected_result": "home reached",
        }
        monitor.after_action(home, {"status": "success"}, _robot_state())
        self.assertEqual(monitor.cumulative_rotation_degrees, 0.0)

    def test_joint_telemetry_watchdog_blocks_velocity_limit(self) -> None:
        joints = JointSafetyLimits(
            position_min_rad=(-2.0, -2.0),
            position_max_rad=(2.0, 2.0),
            max_velocity_rad_s=(1.0, 1.0),
            max_acceleration_rad_s2=(2.0, 2.0),
            max_torque_nm=(10.0, 10.0),
            max_cumulative_motion_rad=(0.5, 0.5),
        )
        monitor = SoftwareSafetyMonitor(
            limits=MotionSafetyLimits(
                joint_limits=joints,
                require_joint_telemetry=True,
                hardware_approved=True,
                profile_name="two-joint-test-rig",
            )
        )
        state = {
            **_robot_state(),
            "telemetry": {
                "joint_positions_rad": [0.0, 0.0],
                "joint_velocities_rad_s": [0.2, 1.2],
                "joint_accelerations_rad_s2": [0.0, 0.0],
                "joint_torques_nm": [0.0, 0.0],
            },
        }
        result = monitor.during_action({}, {}, state)
        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "JOINT_VELOCITY_LIMIT_EXCEEDED")

    def test_joint_watchdog_blocks_cumulative_back_and_forth_rotation(self) -> None:
        joints = JointSafetyLimits(
            position_min_rad=(-2.0,),
            position_max_rad=(2.0,),
            max_velocity_rad_s=(1.0,),
            max_acceleration_rad_s2=(2.0,),
            max_torque_nm=(10.0,),
            max_cumulative_motion_rad=(0.5,),
        )
        monitor = SoftwareSafetyMonitor(
            limits=MotionSafetyLimits(
                joint_limits=joints,
                require_joint_telemetry=True,
                hardware_approved=True,
                profile_name="cumulative-joint-test",
            )
        )

        def state(position: float) -> dict:
            return {
                **_robot_state(),
                "telemetry": {
                    "joint_positions_rad": [position],
                    "joint_velocities_rad_s": [0.2],
                    "joint_accelerations_rad_s2": [0.0],
                    "joint_torques_nm": [0.0],
                },
            }

        step = {
            "step_id": 1,
            "action_type": "move_home",
            "target": "home",
            "expected_result": "home reached",
        }
        self.assertTrue(monitor.before_action(step, state(0.0)).safe)
        self.assertTrue(monitor.during_action(step, {}, state(0.3)).safe)
        result = monitor.during_action(step, {}, state(0.0))
        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "JOINT_CUMULATIVE_MOTION_EXCEEDED")


class _Response:
    def __init__(self, actions):
        self.actions = actions

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({"act": self.actions}).encode("utf-8")


class _HardwareRobot:
    hardware_ready = True

    def __init__(self) -> None:
        self.execute_calls = 0
        self.stopped = False

    def execute_action_chunk(self, actions):
        self.execute_calls += 1
        return {"steps_total": len(actions)}

    def get_action_state(self):
        return [0.0, 0.0]

    def stop(self) -> None:
        self.stopped = True


def _hardware_action_guard(*, cumulative: float = 0.5) -> ActionChunkGuard:
    return ActionChunkGuard(
        ActionChunkSafetyLimits(
            semantics="joint_position_rad",
            lower_bounds=(-2.0, -2.0),
            upper_bounds=(2.0, 2.0),
            max_step_changes=(0.2, 0.2),
            max_cumulative_changes=(cumulative, cumulative),
            max_chunk_size=10,
            require_reference=True,
            hardware_approved=True,
            profile_name="two-joint-test-controller",
            dimension_names=("joint_1", "joint_2"),
        )
    )


class ActionChunkSafetyTests(unittest.TestCase):
    def _step(self) -> dict:
        return {
            "id": 1,
            "skill": "pick",
            "target": "object",
            "expected_result": "held",
        }

    def test_guard_rejects_nan_and_dimension_mismatch(self) -> None:
        guard = ActionChunkGuard(ActionChunkSafetyLimits.normalized_simulation(2))
        self.assertEqual(
            guard.check([[0.0, math.nan]]).reason,
            "ACTION_DIMENSION_OR_VALUE_INVALID",
        )
        self.assertEqual(
            guard.check([[0.0]]).reason,
            "ACTION_DIMENSION_OR_VALUE_INVALID",
        )

    def test_hardware_gateway_requires_explicit_action_profile(self) -> None:
        robot = _HardwareRobot()
        gateway = Pi05ServiceGateway(robot=robot, stub_action_dim=2)
        with patch(
            "robot_agent.gateway.urlopen",
            return_value=_Response([[0.1, 0.1]]),
        ):
            result = gateway.execute("pick object", self._step(), 0, observation={})
        self.assertEqual(result["reason"], "ACTION_SAFETY_CONFIG_REQUIRED")
        self.assertEqual(robot.execute_calls, 0)

    def test_gateway_blocks_cumulative_joint_motion_before_robot_call(self) -> None:
        robot = _HardwareRobot()
        gateway = Pi05ServiceGateway(
            robot=robot,
            stub_action_dim=2,
            action_guard=_hardware_action_guard(cumulative=0.3),
        )
        actions = [[0.1, 0.0], [0.2, 0.0], [0.1, 0.0], [0.0, 0.0]]
        with patch("robot_agent.gateway.urlopen", return_value=_Response(actions)):
            result = gateway.execute("pick object", self._step(), 0, observation={})
        self.assertEqual(result["reason"], "ACTION_CUMULATIVE_DELTA_EXCEEDED")
        self.assertEqual(robot.execute_calls, 0)

    def test_gateway_sends_safe_hardware_actions(self) -> None:
        robot = _HardwareRobot()
        gateway = Pi05ServiceGateway(
            robot=robot,
            stub_action_dim=2,
            action_guard=_hardware_action_guard(),
        )
        actions = [[0.1, 0.0], [0.15, 0.05], [0.2, 0.1]]
        with patch("robot_agent.gateway.urlopen", return_value=_Response(actions)):
            result = gateway.execute("pick object", self._step(), 0, observation={})
        self.assertEqual(result["reason"], "PI05_ACTIONS_EXECUTED")
        self.assertEqual(robot.execute_calls, 1)
        self.assertEqual(
            result["details"]["action_safety"]["profile_name"],
            "two-joint-test-controller",
        )


if __name__ == "__main__":
    unittest.main()
