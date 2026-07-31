from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

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
    """Hardware-independent checks; real-time limits remain controller-owned."""

    def __init__(
        self,
        max_timeout_seconds: float = 300.0,
        forbidden_targets: set[str] | None = None,
    ) -> None:
        self.max_timeout_seconds = max_timeout_seconds
        self.forbidden_targets = forbidden_targets or set()

    def before_action(self, step: PlanStep, robot_state: dict[str, Any]) -> SafetyCheck:
        if robot_state.get("emergency_stop"):
            return SafetyCheck(False, "EMERGENCY_STOP")
        if robot_state.get("connected") is False:
            return SafetyCheck(False, "ROBOT_DISCONNECTED")
        if str(step.get("target")) in self.forbidden_targets:
            return SafetyCheck(False, "FORBIDDEN_TARGET")
        parameters = dict(step.get("parameters") or {})
        position = parameters.get("position_xyz_m")
        if isinstance(position, list) and any(abs(float(value)) > 2.0 for value in position):
            return SafetyCheck(False, "CARTESIAN_WORKSPACE_EXCEEDED")
        delta = parameters.get("delta_xyz_m")
        if isinstance(delta, list) and any(abs(float(value)) > 0.5 for value in delta):
            return SafetyCheck(False, "LINEAR_STEP_LIMIT_EXCEEDED")
        timeout = float(step.get("timeout_seconds", 60.0))
        if timeout <= 0 or timeout > self.max_timeout_seconds:
            return SafetyCheck(False, "INVALID_ACTION_TIMEOUT")
        return SafetyCheck(True)

    def after_action(
        self,
        step: PlanStep,
        action_result: dict[str, Any],
        robot_state: dict[str, Any],
    ) -> SafetyCheck:
        if robot_state.get("emergency_stop"):
            return SafetyCheck(False, "EMERGENCY_STOP")
        if action_result.get("reason") in {"COLLISION_RISK", "HARDWARE_FAULT"}:
            return SafetyCheck(False, str(action_result["reason"]))
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
        signals = dict(observation.get("signals") or {})
        frames = observation.get("frames")
        if isinstance(frames, list) and frames and isinstance(frames[-1], dict):
            signals.update(dict(frames[-1].get("signals") or {}))
        if signals.get("collision_risk"):
            return SafetyCheck(False, "COLLISION_RISK")
        if signals.get("hardware_fault"):
            return SafetyCheck(False, "HARDWARE_FAULT")
        return SafetyCheck(True)
