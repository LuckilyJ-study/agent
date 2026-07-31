from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


class MotionSafetyConfigurationError(ValueError):
    """Raised when a motion safety profile is internally inconsistent."""


@dataclass(frozen=True)
class JointSafetyLimits:
    """Robot-specific joint limits used by the runtime telemetry watchdog."""

    position_min_rad: tuple[float, ...]
    position_max_rad: tuple[float, ...]
    max_velocity_rad_s: tuple[float, ...]
    max_acceleration_rad_s2: tuple[float, ...]
    max_torque_nm: tuple[float, ...]
    max_cumulative_motion_rad: tuple[float, ...]
    joint_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        dimension = len(self.position_min_rad)
        if dimension < 1:
            raise MotionSafetyConfigurationError("At least one joint limit is required.")
        vectors = {
            "position_max_rad": self.position_max_rad,
            "max_velocity_rad_s": self.max_velocity_rad_s,
            "max_acceleration_rad_s2": self.max_acceleration_rad_s2,
            "max_torque_nm": self.max_torque_nm,
            "max_cumulative_motion_rad": self.max_cumulative_motion_rad,
        }
        for name, values in vectors.items():
            if len(values) != dimension:
                raise MotionSafetyConfigurationError(
                    f"{name} must contain {dimension} values."
                )
        if self.joint_names and len(self.joint_names) != dimension:
            raise MotionSafetyConfigurationError(
                f"joint_names must contain {dimension} values."
            )
        all_values = [
            *self.position_min_rad,
            *self.position_max_rad,
            *self.max_velocity_rad_s,
            *self.max_acceleration_rad_s2,
            *self.max_torque_nm,
            *self.max_cumulative_motion_rad,
        ]
        if any(not math.isfinite(float(value)) for value in all_values):
            raise MotionSafetyConfigurationError("Joint limits must be finite numbers.")
        if any(
            float(lower) >= float(upper)
            for lower, upper in zip(self.position_min_rad, self.position_max_rad)
        ):
            raise MotionSafetyConfigurationError(
                "Every joint position minimum must be below its maximum."
            )
        positive_vectors = (
            self.max_velocity_rad_s,
            self.max_acceleration_rad_s2,
            self.max_torque_nm,
            self.max_cumulative_motion_rad,
        )
        if any(float(value) <= 0 for values in positive_vectors for value in values):
            raise MotionSafetyConfigurationError(
                "Joint velocity, acceleration, torque, and cumulative limits must be positive."
            )

    @property
    def dimension(self) -> int:
        return len(self.position_min_rad)


@dataclass(frozen=True)
class MotionSafetyLimits:
    """High-level motion envelope checked independently of the robot driver.

    The defaults are conservative for the bundled tabletop simulation. A real
    deployment must replace them with values approved for its robot, tooling,
    coordinate frames, and workcell.
    """

    workspace_min_xyz_m: tuple[float, float, float] = (-1.0, -1.0, 0.0)
    workspace_max_xyz_m: tuple[float, float, float] = (1.0, 1.0, 1.5)
    max_linear_step_m: float = 0.30
    max_linear_speed_m_s: float = 0.25
    max_orientation_step_degrees: float = 30.0
    max_cumulative_orientation_degrees: float = 180.0
    max_angular_speed_rad_s: float = 0.5
    quaternion_norm_tolerance: float = 1e-3
    require_shortest_rotation_path: bool = True
    joint_limits: JointSafetyLimits | None = None
    require_joint_telemetry: bool = False
    hardware_approved: bool = False
    profile_name: str = "tabletop-simulation-default"

    def __post_init__(self) -> None:
        if len(self.workspace_min_xyz_m) != 3 or len(self.workspace_max_xyz_m) != 3:
            raise MotionSafetyConfigurationError(
                "Workspace minimum and maximum must each contain three values."
            )
        workspace_values = [*self.workspace_min_xyz_m, *self.workspace_max_xyz_m]
        if any(not math.isfinite(float(value)) for value in workspace_values):
            raise MotionSafetyConfigurationError("Workspace limits must be finite.")
        if any(
            float(lower) >= float(upper)
            for lower, upper in zip(
                self.workspace_min_xyz_m, self.workspace_max_xyz_m
            )
        ):
            raise MotionSafetyConfigurationError(
                "Every workspace minimum must be below its maximum."
            )
        positive_values = (
            self.max_linear_step_m,
            self.max_linear_speed_m_s,
            self.max_orientation_step_degrees,
            self.max_cumulative_orientation_degrees,
            self.max_angular_speed_rad_s,
            self.quaternion_norm_tolerance,
        )
        if any(
            not math.isfinite(float(value)) or float(value) <= 0
            for value in positive_values
        ):
            raise MotionSafetyConfigurationError(
                "Motion, rotation, speed, and tolerance limits must be finite and positive."
            )
        if self.max_orientation_step_degrees > 180.0:
            raise MotionSafetyConfigurationError(
                "max_orientation_step_degrees cannot exceed 180 degrees."
            )
        if (
            self.max_cumulative_orientation_degrees
            < self.max_orientation_step_degrees
        ):
            raise MotionSafetyConfigurationError(
                "The cumulative orientation limit cannot be below the per-step limit."
            )
        if self.require_joint_telemetry and self.joint_limits is None:
            raise MotionSafetyConfigurationError(
                "require_joint_telemetry needs robot-specific joint_limits."
            )
        if not str(self.profile_name).strip():
            raise MotionSafetyConfigurationError("profile_name cannot be empty.")
        if self.hardware_approved and (
            self.joint_limits is None or not self.require_joint_telemetry
        ):
            raise MotionSafetyConfigurationError(
                "A hardware-approved motion profile requires joint limits and telemetry."
            )
        if self.hardware_approved and self.profile_name == "tabletop-simulation-default":
            raise MotionSafetyConfigurationError(
                "A hardware-approved motion profile requires an explicit profile_name."
            )


def finite_vector(value: Any, dimension: int) -> tuple[float, ...] | None:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != dimension
        or any(not isinstance(item, (int, float)) for item in value)
    ):
        return None
    parsed = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in parsed):
        return None
    return parsed


def unit_quaternion_xyzw(
    value: Any,
    *,
    tolerance: float,
) -> tuple[float, float, float, float] | None:
    parsed = finite_vector(value, 4)
    if parsed is None:
        return None
    norm = math.sqrt(sum(item * item for item in parsed))
    if norm <= 0 or abs(norm - 1.0) > tolerance:
        return None
    return tuple(item / norm for item in parsed)  # type: ignore[return-value]


def quaternion_shortest_angle_degrees(
    start_xyzw: Sequence[float],
    end_xyzw: Sequence[float],
) -> float:
    """Return the shortest endpoint rotation in [0, 180] degrees."""

    dot = abs(sum(float(a) * float(b) for a, b in zip(start_xyzw, end_xyzw)))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))
