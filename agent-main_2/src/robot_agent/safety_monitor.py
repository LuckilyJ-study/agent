from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from .motion_safety import (
    MotionSafetyLimits,
    finite_vector,
    quaternion_shortest_angle_degrees,
    unit_quaternion_xyzw,
    vector_norm,
)
from .state import PlanStep


@dataclass(frozen=True)
class SafetyCheck:
    safe: bool
    reason: str = "SAFE"


class RuntimeSafetyMonitor(Protocol):
    def before_action(
        self, step: PlanStep, robot_state: dict[str, Any]
    ) -> SafetyCheck: ...

    def after_action(
        self,
        step: PlanStep,
        action_result: dict[str, Any],
        robot_state: dict[str, Any],
    ) -> SafetyCheck: ...

    def during_action(
        self,
        step: PlanStep,
        observation: dict[str, Any],
        robot_state: dict[str, Any],
    ) -> SafetyCheck: ...


class SoftwareSafetyMonitor:
    """Independent command envelope plus a telemetry safety watchdog.

    A real robot driver must still enforce its native hard limits. This monitor
    prevents unsafe high-level commands from reaching that boundary and stops
    execution when supplied joint telemetry exceeds the configured profile.
    """

    def __init__(
        self,
        max_timeout_seconds: float = 300.0,
        forbidden_targets: set[str] | None = None,
        limits: MotionSafetyLimits | None = None,
    ) -> None:
        self.max_timeout_seconds = max_timeout_seconds
        self.forbidden_targets = forbidden_targets or set()
        self.limits = limits or MotionSafetyLimits()
        self._pending_rotation_degrees = 0.0
        self._cumulative_rotation_degrees = 0.0
        self._last_joint_positions: tuple[float, ...] | None = None
        self._cumulative_joint_motion: list[float] = []

    @property
    def cumulative_rotation_degrees(self) -> float:
        return self._cumulative_rotation_degrees

    @property
    def hardware_ready(self) -> bool:
        return self.limits.hardware_approved

    def before_action(self, step: PlanStep, robot_state: dict[str, Any]) -> SafetyCheck:
        self._pending_rotation_degrees = 0.0
        if robot_state.get("emergency_stop"):
            return SafetyCheck(False, "EMERGENCY_STOP")
        if robot_state.get("connected") is False:
            return SafetyCheck(False, "ROBOT_DISCONNECTED")
        telemetry = self._check_joint_telemetry(robot_state)
        if not telemetry.safe:
            return telemetry
        self._begin_joint_motion_tracking(robot_state)
        if str(step.get("target")) in self.forbidden_targets:
            return SafetyCheck(False, "FORBIDDEN_TARGET")
        action_type = str(step.get("action_type") or step.get("skill") or "")
        parameters = dict(step.get("parameters") or {})
        position = parameters.get("position_xyz_m")
        grounding = parameters.get("perception_grounding")
        if (
            position is None
            and action_type == "move_to"
            and isinstance(grounding, dict)
        ):
            position = grounding.get("position_xyz_m")
        if position is not None:
            parsed_position = finite_vector(position, 3)
            if parsed_position is None:
                return SafetyCheck(False, "INVALID_CARTESIAN_POSITION")
            if not self._inside_workspace(parsed_position):
                return SafetyCheck(False, "CARTESIAN_WORKSPACE_EXCEEDED")
            current_position = self._current_position(robot_state)
            if current_position is None:
                return SafetyCheck(False, "CARTESIAN_STATE_UNAVAILABLE")
            translation = tuple(
                parsed_position[index] - current_position[index]
                for index in range(3)
            )
            if vector_norm(translation) > self.limits.max_linear_step_m:
                return SafetyCheck(False, "CARTESIAN_TRANSLATION_LIMIT_EXCEEDED")
        delta = parameters.get("delta_xyz_m")
        if delta is not None:
            parsed_delta = finite_vector(delta, 3)
            if parsed_delta is None:
                return SafetyCheck(False, "INVALID_LINEAR_DELTA")
            if vector_norm(parsed_delta) > self.limits.max_linear_step_m:
                return SafetyCheck(False, "LINEAR_STEP_LIMIT_EXCEEDED")
            current_position = self._current_position(robot_state)
            if current_position is None:
                return SafetyCheck(False, "CARTESIAN_STATE_UNAVAILABLE")
            destination = tuple(
                current_position[index] + parsed_delta[index] for index in range(3)
            )
            if not self._inside_workspace(destination):
                return SafetyCheck(False, "CARTESIAN_WORKSPACE_EXCEEDED")
        speed = parameters.get("speed_m_s")
        if speed is not None:
            if not isinstance(speed, (int, float)):
                return SafetyCheck(False, "INVALID_LINEAR_SPEED")
            parsed_speed = float(speed)
            if not 0 < parsed_speed <= self.limits.max_linear_speed_m_s:
                return SafetyCheck(False, "LINEAR_SPEED_LIMIT_EXCEEDED")
        orientation = parameters.get("orientation_xyzw")
        if orientation is not None:
            target_orientation = unit_quaternion_xyzw(
                orientation,
                tolerance=self.limits.quaternion_norm_tolerance,
            )
            if target_orientation is None:
                return SafetyCheck(False, "INVALID_ORIENTATION_QUATERNION")
            current_orientation = self._current_orientation(robot_state)
            if current_orientation is None:
                return SafetyCheck(False, "ORIENTATION_STATE_UNAVAILABLE")
            if (
                self.limits.require_shortest_rotation_path
                and parameters.get("rotation_path") != "shortest"
            ):
                return SafetyCheck(False, "ROTATION_PATH_NOT_SHORTEST")
            angular_speed = parameters.get("max_angular_speed_rad_s")
            if not isinstance(angular_speed, (int, float)):
                return SafetyCheck(False, "ANGULAR_SPEED_REQUIRED")
            if not 0 < float(angular_speed) <= self.limits.max_angular_speed_rad_s:
                return SafetyCheck(False, "ANGULAR_SPEED_LIMIT_EXCEEDED")
            rotation_degrees = quaternion_shortest_angle_degrees(
                current_orientation,
                target_orientation,
            )
            if rotation_degrees > self.limits.max_orientation_step_degrees:
                return SafetyCheck(False, "ORIENTATION_STEP_LIMIT_EXCEEDED")
            if (
                self._cumulative_rotation_degrees + rotation_degrees
                > self.limits.max_cumulative_orientation_degrees
            ):
                return SafetyCheck(False, "ORIENTATION_CUMULATIVE_LIMIT_EXCEEDED")
            self._pending_rotation_degrees = rotation_degrees
        timeout = float(step.get("timeout_seconds", 60.0))
        if (
            not math.isfinite(timeout)
            or timeout <= 0
            or timeout > self.max_timeout_seconds
        ):
            return SafetyCheck(False, "INVALID_ACTION_TIMEOUT")
        return SafetyCheck(True)

    def after_action(
        self,
        step: PlanStep,
        action_result: dict[str, Any],
        robot_state: dict[str, Any],
    ) -> SafetyCheck:
        if robot_state.get("emergency_stop"):
            self._pending_rotation_degrees = 0.0
            self._clear_joint_motion_tracking()
            return SafetyCheck(False, "EMERGENCY_STOP")
        if action_result.get("reason") in {"COLLISION_RISK", "HARDWARE_FAULT"}:
            self._pending_rotation_degrees = 0.0
            self._clear_joint_motion_tracking()
            return SafetyCheck(False, str(action_result["reason"]))
        telemetry = self._check_joint_telemetry(robot_state)
        if not telemetry.safe:
            self._pending_rotation_degrees = 0.0
            self._clear_joint_motion_tracking()
            return telemetry
        joint_motion = self._track_joint_motion(robot_state)
        if not joint_motion.safe:
            self._pending_rotation_degrees = 0.0
            self._clear_joint_motion_tracking()
            return joint_motion
        if action_result.get("status") == "success":
            if str(step.get("action_type") or step.get("skill")) == "move_home":
                self._cumulative_rotation_degrees = 0.0
            else:
                self._cumulative_rotation_degrees += self._pending_rotation_degrees
        self._pending_rotation_degrees = 0.0
        self._clear_joint_motion_tracking()
        return SafetyCheck(True)

    def during_action(
        self,
        step: PlanStep,
        observation: dict[str, Any],
        robot_state: dict[str, Any],
    ) -> SafetyCheck:
        if robot_state.get("emergency_stop"):
            return SafetyCheck(False, "EMERGENCY_STOP")
        if robot_state.get("connected") is False:
            return SafetyCheck(False, "ROBOT_DISCONNECTED")
        telemetry = self._check_joint_telemetry(robot_state)
        if not telemetry.safe:
            return telemetry
        joint_motion = self._track_joint_motion(robot_state)
        if not joint_motion.safe:
            return joint_motion
        signals = dict(observation.get("signals") or {})
        frames = observation.get("frames")
        if isinstance(frames, list) and frames and isinstance(frames[-1], dict):
            signals.update(dict(frames[-1].get("signals") or {}))
        if signals.get("collision_risk"):
            return SafetyCheck(False, "COLLISION_RISK")
        if signals.get("hardware_fault"):
            return SafetyCheck(False, "HARDWARE_FAULT")
        return SafetyCheck(True)

    def _inside_workspace(self, position: tuple[float, ...]) -> bool:
        return all(
            float(lower) <= value <= float(upper)
            for value, lower, upper in zip(
                position,
                self.limits.workspace_min_xyz_m,
                self.limits.workspace_max_xyz_m,
            )
        )

    def _current_position(
        self, robot_state: dict[str, Any]
    ) -> tuple[float, ...] | None:
        cartesian_pose = robot_state.get("cartesian_pose")
        if isinstance(cartesian_pose, dict):
            value = cartesian_pose.get("position_xyz_m")
        else:
            value = robot_state.get("position_xyz_m")
        return finite_vector(value, 3)

    def _current_orientation(
        self, robot_state: dict[str, Any]
    ) -> tuple[float, float, float, float] | None:
        cartesian_pose = robot_state.get("cartesian_pose")
        if isinstance(cartesian_pose, dict):
            value = cartesian_pose.get("orientation_xyzw")
        else:
            value = robot_state.get("orientation_xyzw")
        return unit_quaternion_xyzw(
            value,
            tolerance=self.limits.quaternion_norm_tolerance,
        )

    def _check_joint_telemetry(
        self, robot_state: dict[str, Any]
    ) -> SafetyCheck:
        limits = self.limits.joint_limits
        if limits is None:
            return SafetyCheck(True)
        telemetry = robot_state.get("telemetry")
        source = telemetry if isinstance(telemetry, dict) else robot_state
        fields = {
            "positions": finite_vector(source.get("joint_positions_rad"), limits.dimension),
            "velocities": finite_vector(
                source.get("joint_velocities_rad_s"), limits.dimension
            ),
            "accelerations": finite_vector(
                source.get("joint_accelerations_rad_s2"), limits.dimension
            ),
            "torques": finite_vector(source.get("joint_torques_nm"), limits.dimension),
        }
        if self.limits.require_joint_telemetry and any(
            value is None for value in fields.values()
        ):
            return SafetyCheck(False, "JOINT_TELEMETRY_UNAVAILABLE")
        positions = fields["positions"]
        if positions is not None and any(
            value < lower or value > upper
            for value, lower, upper in zip(
                positions, limits.position_min_rad, limits.position_max_rad
            )
        ):
            return SafetyCheck(False, "JOINT_POSITION_LIMIT_EXCEEDED")
        comparisons = (
            ("velocities", limits.max_velocity_rad_s, "JOINT_VELOCITY_LIMIT_EXCEEDED"),
            (
                "accelerations",
                limits.max_acceleration_rad_s2,
                "JOINT_ACCELERATION_LIMIT_EXCEEDED",
            ),
            ("torques", limits.max_torque_nm, "JOINT_TORQUE_LIMIT_EXCEEDED"),
        )
        for field, maxima, reason in comparisons:
            values = fields[field]
            if values is not None and any(
                abs(value) > maximum for value, maximum in zip(values, maxima)
            ):
                return SafetyCheck(False, reason)
        return SafetyCheck(True)

    def _begin_joint_motion_tracking(self, robot_state: dict[str, Any]) -> None:
        limits = self.limits.joint_limits
        if limits is None:
            self._clear_joint_motion_tracking()
            return
        positions = self._joint_positions(robot_state, limits.dimension)
        self._last_joint_positions = positions
        self._cumulative_joint_motion = [0.0] * limits.dimension

    def _track_joint_motion(self, robot_state: dict[str, Any]) -> SafetyCheck:
        limits = self.limits.joint_limits
        if limits is None:
            return SafetyCheck(True)
        positions = self._joint_positions(robot_state, limits.dimension)
        if positions is None:
            if self.limits.require_joint_telemetry:
                return SafetyCheck(False, "JOINT_TELEMETRY_UNAVAILABLE")
            return SafetyCheck(True)
        if self._last_joint_positions is None:
            self._last_joint_positions = positions
            self._cumulative_joint_motion = [0.0] * limits.dimension
            return SafetyCheck(True)
        for index, value in enumerate(positions):
            self._cumulative_joint_motion[index] += abs(
                value - self._last_joint_positions[index]
            )
            if (
                self._cumulative_joint_motion[index]
                > limits.max_cumulative_motion_rad[index]
            ):
                return SafetyCheck(False, "JOINT_CUMULATIVE_MOTION_EXCEEDED")
        self._last_joint_positions = positions
        return SafetyCheck(True)

    def _joint_positions(
        self,
        robot_state: dict[str, Any],
        dimension: int,
    ) -> tuple[float, ...] | None:
        telemetry = robot_state.get("telemetry")
        source = telemetry if isinstance(telemetry, dict) else robot_state
        return finite_vector(source.get("joint_positions_rad"), dimension)

    def _clear_joint_motion_tracking(self) -> None:
        self._last_joint_positions = None
        self._cumulative_joint_motion = []
