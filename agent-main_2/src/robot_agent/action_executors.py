from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .capabilities import CapabilityRegistry
from .motion import parse_relative_motion_target
from .policy_metadata import PolicyMetadata
from .state import PlanStep


class PrimitiveRobotController(Protocol):
    hardware_ready: bool

    def move_relative(self, direction: str, distance_cm: float) -> dict[str, Any]: ...

    def move_to(self, target: str, parameters: dict[str, Any]) -> dict[str, Any]: ...

    def move_to_pose(
        self,
        position_xyz_m: list[float],
        orientation_xyzw: list[float] | None,
        coordinate_frame: str,
        speed_m_s: float | None,
        max_angular_speed_rad_s: float | None,
        shortest_path: bool,
    ) -> dict[str, Any]: ...

    def move_linear(
        self,
        delta_xyz_m: list[float],
        coordinate_frame: str,
        speed_m_s: float | None,
    ) -> dict[str, Any]: ...

    def move_home(self) -> dict[str, Any]: ...

    def open_gripper(self) -> dict[str, Any]: ...

    def close_gripper(self) -> dict[str, Any]: ...

    def stop(self) -> None: ...

    def get_state(self) -> dict[str, Any]: ...


@dataclass
class DryRunRobotController:
    """Safe controller used until a real robot driver is injected."""

    state: dict[str, Any] = field(
        default_factory=lambda: {
            "available": True,
            "source": "dry_run_robot",
            "pose": "home",
            "cartesian_pose": {
                "position_xyz_m": [0.30, 0.00, 0.35],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "coordinate_frame": "robot_base",
            },
            "gripper": "unknown",
            "stopped": False,
        }
    )
    hardware_ready: bool = field(default=False, init=False)

    def move_relative(self, direction: str, distance_cm: float) -> dict[str, Any]:
        self.state.update(
            {"last_motion": {"direction": direction, "distance_cm": distance_cm}, "stopped": False}
        )
        return self._result("move_relative")

    def move_to(self, target: str, parameters: dict[str, Any]) -> dict[str, Any]:
        self.state.update({"pose": target, "parameters": dict(parameters), "stopped": False})
        return self._result("move_to")

    def move_to_pose(
        self,
        position_xyz_m: list[float],
        orientation_xyzw: list[float] | None,
        coordinate_frame: str,
        speed_m_s: float | None,
        max_angular_speed_rad_s: float | None,
        shortest_path: bool,
    ) -> dict[str, Any]:
        self.state.update(
            {
                "cartesian_pose": {
                    "position_xyz_m": [float(value) for value in position_xyz_m],
                    "orientation_xyzw": orientation_xyzw,
                    "coordinate_frame": coordinate_frame,
                },
                "speed_m_s": speed_m_s,
                "max_angular_speed_rad_s": max_angular_speed_rad_s,
                "shortest_path": shortest_path,
                "stopped": False,
            }
        )
        return self._result("move_to_pose")

    def move_linear(
        self,
        delta_xyz_m: list[float],
        coordinate_frame: str,
        speed_m_s: float | None,
    ) -> dict[str, Any]:
        current_pose = dict(self.state.get("cartesian_pose") or {})
        current_position = list(current_pose.get("position_xyz_m") or [0.0, 0.0, 0.0])
        next_position = [
            float(current_position[index]) + float(delta_xyz_m[index])
            for index in range(3)
        ]
        self.state.update(
            {
                "cartesian_pose": {
                    **current_pose,
                    "position_xyz_m": next_position,
                    "coordinate_frame": coordinate_frame,
                },
                "last_linear_motion": {
                    "delta_xyz_m": [float(value) for value in delta_xyz_m],
                    "coordinate_frame": coordinate_frame,
                },
                "speed_m_s": speed_m_s,
                "stopped": False,
            }
        )
        return self._result("move_linear")

    def move_home(self) -> dict[str, Any]:
        self.state.update({"pose": "home", "stopped": False})
        return self._result("move_home")

    def open_gripper(self) -> dict[str, Any]:
        self.state.update({"gripper": "open", "stopped": False})
        return self._result("open_gripper")

    def close_gripper(self) -> dict[str, Any]:
        self.state.update({"gripper": "closed", "stopped": False})
        return self._result("close_gripper")

    def stop(self) -> None:
        self.state["stopped"] = True

    def get_state(self) -> dict[str, Any]:
        return dict(self.state)

    def _result(self, action_type: str) -> dict[str, Any]:
        return {
            "status": "success",
            "reason": "DRY_RUN_COMMAND_COMPLETED",
            "command_completed": True,
            "physical_result_verified": False,
            "details": {"action_type": action_type, "controller": "dry_run"},
        }


class RobotPrimitiveExecutor:
    def __init__(self, controller: PrimitiveRobotController) -> None:
        self.controller = controller

    def execute(self, step: PlanStep, observation: dict[str, Any]) -> dict[str, Any]:
        action_type = str(step["action_type"])
        parameters = dict(step.get("parameters") or {})
        if action_type == "move_relative":
            motion = parse_relative_motion_target(str(step["target"]))
            return self.controller.move_relative(motion.direction, motion.distance_cm)
        if action_type == "move_to":
            return self.controller.move_to(str(step["target"]), parameters)
        if action_type == "move_to_pose":
            return self.controller.move_to_pose(
                [float(value) for value in parameters["position_xyz_m"]],
                (
                    [float(value) for value in parameters["orientation_xyzw"]]
                    if parameters.get("orientation_xyzw") is not None
                    else None
                ),
                str(parameters["coordinate_frame"]),
                (
                    float(parameters["speed_m_s"])
                    if parameters.get("speed_m_s") is not None
                    else None
                ),
                (
                    float(parameters["max_angular_speed_rad_s"])
                    if parameters.get("max_angular_speed_rad_s") is not None
                    else None
                ),
                parameters.get("rotation_path") == "shortest",
            )
        if action_type == "move_linear":
            return self.controller.move_linear(
                [float(value) for value in parameters["delta_xyz_m"]],
                str(parameters["coordinate_frame"]),
                (
                    float(parameters["speed_m_s"])
                    if parameters.get("speed_m_s") is not None
                    else None
                ),
            )
        if action_type == "move_home":
            return self.controller.move_home()
        if action_type == "open_gripper":
            mismatch = self._gripper_pose_mismatch(parameters)
            if mismatch:
                return mismatch
            return self.controller.open_gripper()
        if action_type == "close_gripper":
            mismatch = self._gripper_pose_mismatch(parameters)
            if mismatch:
                return mismatch
            return self.controller.close_gripper()
        return {
            "status": "failed",
            "reason": "UNSUPPORTED_ROBOT_PRIMITIVE",
            "command_completed": False,
            "physical_result_verified": False,
        }

    def stop(self) -> None:
        self.controller.stop()

    def _gripper_pose_mismatch(
        self, parameters: dict[str, Any]
    ) -> dict[str, Any] | None:
        expected = parameters.get("at_position_xyz_m")
        if expected is None:
            return None
        state = self.controller.get_state()
        actual = (state.get("cartesian_pose") or {}).get("position_xyz_m")
        if (
            not isinstance(actual, list)
            or len(actual) != 3
            or any(abs(float(actual[i]) - float(expected[i])) > 0.01 for i in range(3))
        ):
            return {
                "status": "failed",
                "reason": "GRIPPER_POSE_MISMATCH",
                "command_completed": False,
                "physical_result_verified": False,
                "details": {"expected_xyz_m": expected, "actual_xyz_m": actual},
            }
        return None


class PolicyBackend(Protocol):
    def execute(
        self,
        step: PlanStep,
        observation: dict[str, Any],
        robot_state: dict[str, Any],
    ) -> dict[str, Any]: ...

    def stop(self) -> None: ...


class PolicyRegistry:
    def __init__(self, policies: dict[str, PolicyBackend] | None = None) -> None:
        self._policies = dict(policies or {})
        self._metadata: dict[str, PolicyMetadata] = {
            policy_id: PolicyMetadata(
                policy_id=policy_id,
                action_type=(
                    policy_id
                    if policy_id in {"pick", "place", "manipulate", "inspect"}
                    else "manipulate"
                ),
            )
            for policy_id in self._policies
        }

    def register(
        self,
        policy_id: str,
        backend: PolicyBackend,
        metadata: PolicyMetadata | None = None,
    ) -> None:
        if not policy_id.strip():
            raise ValueError("policy_id cannot be empty.")
        self._policies[policy_id] = backend
        self._metadata[policy_id] = metadata or PolicyMetadata(policy_id=policy_id)

    def get(self, policy_id: str) -> PolicyBackend | None:
        return self._policies.get(policy_id)

    def metadata(self, policy_id: str) -> PolicyMetadata | None:
        return self._metadata.get(policy_id)

    def find_for_action(self, action_type: str) -> list[str]:
        return sorted(
            policy_id
            for policy_id, metadata in self._metadata.items()
            if metadata.action_type in {action_type, "*"}
        )

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "policy_id": item.policy_id,
                "version": item.version,
                "action_type": item.action_type,
                "required_inputs": list(item.required_inputs),
                "robot_types": list(item.robot_types),
                "supports_stop": item.supports_stop,
                "timeout_seconds": item.timeout_seconds,
                "description": item.description,
            }
            for item in self._metadata.values()
        ]


class PolicyExecutor:
    def __init__(
        self,
        registry: PolicyRegistry,
        state_provider: Any,
        capabilities: CapabilityRegistry | None = None,
    ) -> None:
        self.registry = registry
        self.state_provider = state_provider
        self.capabilities = capabilities or CapabilityRegistry()
        self._active_backend: PolicyBackend | None = None

    def execute(self, step: PlanStep, observation: dict[str, Any]) -> dict[str, Any]:
        capability = self.capabilities.get(str(step["action_type"]))
        policy_id = str(step.get("policy_id") or capability.policy_id or "")
        backend = self.registry.get(policy_id)
        if backend is None:
            return {
                "status": "failed",
                "reason": "POLICY_NOT_AVAILABLE",
                "command_completed": False,
                "physical_result_verified": False,
                "details": {"policy_id": policy_id, "action_type": step["action_type"]},
            }
        robot_state = self.state_provider.robot_state()
        metadata = self.registry.metadata(policy_id)
        if metadata and metadata.action_type not in {
            str(step["action_type"]),
            "*",
        }:
            return {
                "status": "failed",
                "reason": "POLICY_ACTION_MISMATCH",
                "command_completed": False,
                "physical_result_verified": False,
                "details": {
                    "policy_id": policy_id,
                    "policy_action_type": metadata.action_type,
                    "requested_action_type": step["action_type"],
                },
            }
        missing_inputs = (
            metadata.validate_inputs(observation, robot_state) if metadata else []
        )
        if missing_inputs:
            return {
                "status": "failed",
                "reason": "POLICY_INPUTS_UNAVAILABLE",
                "command_completed": False,
                "physical_result_verified": False,
                "details": {"policy_id": policy_id, "missing_inputs": missing_inputs},
            }
        self._active_backend = backend
        return backend.execute(step, observation, robot_state)

    def stop(self) -> None:
        if self._active_backend is not None:
            self._active_backend.stop()


class DryRunPolicyBackend:
    """Explicitly unverified policy backend for integration tests and demos."""

    def __init__(self) -> None:
        self.stopped = False

    def execute(
        self,
        step: PlanStep,
        observation: dict[str, Any],
        robot_state: dict[str, Any],
    ) -> dict[str, Any]:
        self.stopped = False
        return {
            "status": "success",
            "reason": "DRY_RUN_POLICY_COMPLETED",
            "command_completed": True,
            "physical_result_verified": False,
            "details": {
                "policy_id": step.get("policy_id"),
                "action_type": step["action_type"],
                "target": step["target"],
            },
        }

    def stop(self) -> None:
        self.stopped = True


class Pi05GatewayBackend:
    """Adapt the existing Pi05 service gateway to the new PolicyBackend API."""

    def __init__(self, gateway: Any) -> None:
        self.gateway = gateway
        self._attempts: dict[str, int] = {}

    def execute(
        self,
        step: PlanStep,
        observation: dict[str, Any],
        robot_state: dict[str, Any],
    ) -> dict[str, Any]:
        instance_id = str(step.get("instance_id") or step.get("step_id"))
        retry_count = self._attempts.get(instance_id, 0)
        self._attempts[instance_id] = retry_count + 1
        parameters = dict(step.get("parameters") or {})
        task_text = str(
            parameters.get("instruction")
            or f"{step['action_type']} {step['target']}"
        )
        legacy_step: PlanStep = {
            **step,
            "id": int(step["step_id"]),
            "skill": str(step["action_type"]),
        }
        result = self.gateway.execute(
            task_text,
            legacy_step,
            retry_count,
            observation=observation,
        )
        if not isinstance(result, dict):
            return {
                "status": "failed",
                "reason": "INVALID_PI05_RESULT",
                "command_completed": False,
                "physical_result_verified": False,
            }
        success = result.get("status") == "success"
        return {
            **result,
            "command_completed": bool(
                result.get("command_completed", success)
            ),
            "physical_result_verified": bool(
                result.get("physical_result_verified", False)
            ),
        }

    def stop(self) -> None:
        stop = getattr(self.gateway, "stop", None)
        if callable(stop):
            stop()


def build_pi05_policy_registry(
    gateway: Any,
    action_types: tuple[str, ...] = ("pick", "place", "manipulate", "inspect"),
) -> PolicyRegistry:
    """Register one Pi05 gateway behind trusted skill-specific policy IDs."""

    registry = PolicyRegistry()
    backend = Pi05GatewayBackend(gateway)
    for action_type in action_types:
        policy_id = f"pi05_{action_type}"
        registry.register(
            policy_id,
            backend,
            PolicyMetadata(
                policy_id=policy_id,
                version="gateway",
                action_type=action_type,
                required_inputs=("perception", "robot_state"),
                supports_stop=True,
                description=f"Pi05 gateway backend for {action_type}.",
            ),
        )
    return registry
