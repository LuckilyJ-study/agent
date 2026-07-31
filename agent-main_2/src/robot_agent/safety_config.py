from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .action_safety import ActionChunkSafetyLimits
from .motion_safety import JointSafetyLimits, MotionSafetyLimits


@dataclass(frozen=True)
class SafetyProfiles:
    motion: MotionSafetyLimits
    policy_action: ActionChunkSafetyLimits


def load_safety_profiles(path: str | Path) -> SafetyProfiles:
    """Load validated motion and policy-action limits from one JSON file."""

    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load safety profile {config_path}: {error}") from error
    if not isinstance(raw, Mapping):
        raise ValueError("Safety profile root must be a JSON object.")
    motion_raw = _mapping(raw.get("motion"), "motion")
    action_raw = _mapping(raw.get("policy_action"), "policy_action")

    motion_values = dict(motion_raw)
    for name in ("workspace_min_xyz_m", "workspace_max_xyz_m"):
        if name in motion_values:
            motion_values[name] = _number_tuple(motion_values[name], name)
    joint_raw = motion_values.get("joint_limits")
    if joint_raw is not None:
        joint_values = dict(_mapping(joint_raw, "motion.joint_limits"))
        for name in (
            "position_min_rad",
            "position_max_rad",
            "max_velocity_rad_s",
            "max_acceleration_rad_s2",
            "max_torque_nm",
            "max_cumulative_motion_rad",
        ):
            joint_values[name] = _number_tuple(
                joint_values.get(name), f"motion.joint_limits.{name}"
            )
        if "joint_names" in joint_values:
            joint_values["joint_names"] = _string_tuple(
                joint_values["joint_names"], "motion.joint_limits.joint_names"
            )
        motion_values["joint_limits"] = JointSafetyLimits(**joint_values)

    action_values = dict(action_raw)
    for name in (
        "lower_bounds",
        "upper_bounds",
        "max_step_changes",
        "max_cumulative_changes",
    ):
        action_values[name] = _number_tuple(
            action_values.get(name), f"policy_action.{name}"
        )
    if "dimension_names" in action_values:
        action_values["dimension_names"] = _string_tuple(
            action_values["dimension_names"],
            "policy_action.dimension_names",
        )

    try:
        return SafetyProfiles(
            motion=MotionSafetyLimits(**motion_values),
            policy_action=ActionChunkSafetyLimits(**action_values),
        )
    except TypeError as error:
        raise ValueError(f"Safety profile contains invalid or unknown fields: {error}") from error


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _number_tuple(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ValueError(f"{name} must be a JSON array of numbers.")
    return tuple(float(item) for item in value)


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{name} must be a JSON array of non-empty strings.")
    return tuple(item.strip() for item in value)
