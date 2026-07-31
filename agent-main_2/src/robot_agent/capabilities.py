from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from .motion import parse_relative_motion_target
from .motion_safety import unit_quaternion_xyzw
from .domain import Condition, Effect
from .state import PlanStep


ExecutorKind = Literal["robot", "policy"]


class CapabilityError(ValueError):
    """Raised when a plan requests a capability the robot does not expose."""


@dataclass(frozen=True)
class Capability:
    action_type: str
    executor: ExecutorKind
    description: str
    policy_id: str | None = None
    supports_stop: bool = True
    required_parameters: tuple[str, ...] = ()
    default_timeout_seconds: float = 60.0
    default_max_attempts: int = 2

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Capability:
        action_type = str(raw.get("action_type") or "").strip()
        executor = str(raw.get("executor") or "").strip()
        description = str(raw.get("description") or "").strip()
        if not action_type or executor not in {"robot", "policy"} or not description:
            raise CapabilityError(
                "Each capability requires action_type, executor=robot|policy, "
                "and description."
            )
        return cls(
            action_type=action_type,
            executor=executor,  # type: ignore[arg-type]
            description=description,
            policy_id=(
                str(raw["policy_id"]).strip()
                if raw.get("policy_id") is not None
                else (action_type if executor == "policy" else None)
            ),
            supports_stop=bool(raw.get("supports_stop", True)),
            required_parameters=tuple(
                str(value)
                for value in raw.get("required_parameters", [])
                if str(value).strip()
            ),
            default_timeout_seconds=float(
                raw.get("default_timeout_seconds", 60.0)
            ),
            default_max_attempts=int(raw.get("default_max_attempts", 2)),
        )


class CapabilityRegistry:
    """Single source of truth shared by Planner validation and execution routing."""

    def __init__(self, capabilities: Iterable[Capability] | None = None) -> None:
        selected = tuple(capabilities or default_capabilities())
        action_types = [item.action_type for item in selected]
        if len(action_types) != len(set(action_types)):
            raise CapabilityError("Capability action_type values must be unique.")
        self._capabilities = {item.action_type: item for item in selected}
        if not self._capabilities:
            raise CapabilityError("At least one robot capability must be registered.")

    def get(self, action_type: str) -> Capability:
        try:
            return self._capabilities[action_type]
        except KeyError as error:
            raise CapabilityError(f"Capability '{action_type}' is not registered.") from error

    @classmethod
    def from_dicts(
        cls, raw_capabilities: Iterable[dict[str, Any]]
    ) -> CapabilityRegistry:
        return cls(Capability.from_dict(raw) for raw in raw_capabilities)

    def normalize_step(self, step: PlanStep, index: int) -> PlanStep:
        action_type = str(step.get("action_type") or step.get("skill") or "").strip()
        if not action_type:
            raise CapabilityError("Plan step requires action_type (or legacy skill).")
        capability = self.get(action_type)
        supplied_executor = step.get("executor")
        if supplied_executor is not None and supplied_executor != capability.executor:
            raise CapabilityError(
                f"Capability '{action_type}' must use executor='{capability.executor}'."
            )
        target = str(step.get("target") or "").strip()
        if not target:
            raise CapabilityError(f"Capability '{action_type}' requires a non-empty target.")
        expected_result = str(step.get("expected_result") or "").strip()
        if not expected_result:
            raise CapabilityError(
                f"Capability '{action_type}' requires an observable expected_result."
            )
        if action_type == "move_relative":
            parse_relative_motion_target(target)
        if action_type in {"open_gripper", "close_gripper"} and target != "gripper":
            raise CapabilityError(f"Capability '{action_type}' target must be 'gripper'.")
        step_id = int(step.get("step_id", step.get("id", index)))
        normalized: PlanStep = {
            **step,
            "id": step_id,
            "step_id": step_id,
            "skill": action_type,
            "action_type": action_type,
            "target": target,
            "expected_result": expected_result,
            "executor": capability.executor,
            "status": "pending",
            "parameters": dict(step.get("parameters") or {}),
            "timeout_seconds": float(
                step.get("timeout_seconds", capability.default_timeout_seconds)
            ),
            "max_attempts": int(
                step.get("max_attempts", capability.default_max_attempts)
            ),
            "depends_on": [int(value) for value in step.get("depends_on", [])],
            "conditions": list(step.get("conditions", [])),
            "effects": list(step.get("effects", [])),
            "on_condition_false": step.get("on_condition_false", "fail"),
        }
        if normalized["timeout_seconds"] <= 0:
            raise CapabilityError("timeout_seconds must be positive.")
        if normalized["max_attempts"] < 1:
            raise CapabilityError("max_attempts must be at least 1.")
        if normalized["on_condition_false"] not in {"skip", "fail"}:
            raise CapabilityError("on_condition_false must be 'skip' or 'fail'.")
        try:
            for condition in normalized["conditions"]:
                Condition.from_dict(condition)
            for effect in normalized["effects"]:
                Effect.from_dict(effect)
        except (TypeError, ValueError) as error:
            raise CapabilityError(str(error)) from error
        missing_parameters = [
            name for name in capability.required_parameters
            if name not in normalized["parameters"]
        ]
        if missing_parameters:
            raise CapabilityError(
                f"Capability '{action_type}' is missing parameters: {missing_parameters}."
            )
        if action_type == "move_to_pose":
            _validate_xyz(
                normalized["parameters"].get("position_xyz_m"),
                "position_xyz_m",
                max_abs=2.0,
            )
            _validate_coordinate_frame(normalized["parameters"])
            _validate_motion_options(normalized["parameters"])
        if action_type == "move_linear":
            _validate_xyz(
                normalized["parameters"].get("delta_xyz_m"),
                "delta_xyz_m",
                max_abs=0.5,
            )
            _validate_coordinate_frame(normalized["parameters"])
            _validate_motion_options(normalized["parameters"])
        if "at_position_xyz_m" in normalized["parameters"]:
            _validate_xyz(
                normalized["parameters"]["at_position_xyz_m"],
                "at_position_xyz_m",
                max_abs=2.0,
            )
            _validate_coordinate_frame(normalized["parameters"])
        if capability.policy_id and not normalized.get("policy_id"):
            normalized["policy_id"] = capability.policy_id
        return normalized

    def normalize_plan(self, plan: list[PlanStep]) -> list[PlanStep]:
        if not plan:
            raise CapabilityError("Planner returned an empty plan.")
        normalized = [self.normalize_step(step, index) for index, step in enumerate(plan, 1)]
        ids = [int(step["step_id"]) for step in normalized]
        if len(ids) != len(set(ids)):
            raise CapabilityError("Plan step_id values must be unique within one plan.")
        known_ids = set(ids)
        for step in normalized:
            step_id = int(step["step_id"])
            dependencies = set(step.get("depends_on", []))
            if step_id in dependencies:
                raise CapabilityError(f"Step {step_id} cannot depend on itself.")
            unknown = dependencies.difference(known_ids)
            if unknown:
                raise CapabilityError(
                    f"Step {step_id} has unknown dependencies: {sorted(unknown)}."
                )
        _reject_dependency_cycles(normalized)
        return normalized

    def planner_skills(self) -> list[dict[str, Any]]:
        """Return the model-visible skill contract.

        Execution backend and policy identifiers are intentionally excluded:
        those are local deployment details selected by ExecutorRouter.
        """

        return [
            {
                "action_type": item.action_type,
                "description": item.description,
                "required_parameters": list(item.required_parameters),
            }
            for item in self._capabilities.values()
        ]

    def routing_table(self) -> list[dict[str, Any]]:
        """Return trusted local routing data for diagnostics, never for the LLM."""

        return [
            {
                "action_type": item.action_type,
                "executor": item.executor,
                "policy_id": item.policy_id,
                "supports_stop": item.supports_stop,
            }
            for item in self._capabilities.values()
        ]


def default_capabilities() -> tuple[Capability, ...]:
    return (
        Capability("move_relative", "robot", "Move a small bounded distance."),
        Capability("move_to", "robot", "Move to a named target or resolved pose."),
        Capability(
            "move_to_pose",
            "robot",
            "Move the end effector to an environment-provided Cartesian pose in meters.",
            required_parameters=("position_xyz_m", "coordinate_frame"),
        ),
        Capability(
            "move_linear",
            "robot",
            "Move linearly by an environment-derived XYZ delta in meters.",
            required_parameters=("delta_xyz_m", "coordinate_frame"),
        ),
        Capability("move_home", "robot", "Return to the configured safe home pose."),
        Capability("open_gripper", "robot", "Open the gripper."),
        Capability("close_gripper", "robot", "Close the gripper."),
        Capability(
            "pick",
            "policy",
            (
                "Pick the target object with a trained policy. The skill owns "
                "approach, grasp, gripper timing, and lift; the Planner must not "
                "expand those motor actions."
            ),
            "pick",
        ),
        Capability(
            "place",
            "policy",
            (
                "Place the currently held object at the target destination with "
                "a trained policy. Optional parameters may identify the object."
            ),
            "place",
        ),
        Capability(
            "manipulate",
            "policy",
            (
                "Run a registered task-specific manipulation such as opening a "
                "container or using a held tool. parameters.operation should "
                "describe the semantic operation."
            ),
            "manipulate",
        ),
        Capability(
            "inspect",
            "policy",
            "Observe or relocalize a target without claiming a manipulation result.",
            "inspect",
        ),
    )


def _reject_dependency_cycles(plan: list[PlanStep]) -> None:
    graph = {
        int(step["step_id"]): set(step.get("depends_on", []))
        for step in plan
    }
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(step_id: int) -> None:
        if step_id in visiting:
            raise CapabilityError("Plan dependency graph contains a cycle.")
        if step_id in visited:
            return
        visiting.add(step_id)
        for dependency in graph[step_id]:
            visit(dependency)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in graph:
        visit(step_id)


def _validate_xyz(value: Any, name: str, max_abs: float) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(not isinstance(item, (int, float)) for item in value)
    ):
        raise CapabilityError(f"{name} must be a numeric [x, y, z] list.")
    if any(not math.isfinite(float(item)) for item in value):
        raise CapabilityError(f"{name} must contain only finite numbers.")
    if any(abs(float(item)) > max_abs for item in value):
        raise CapabilityError(
            f"Every {name} component must be within +/-{max_abs:g} meters."
        )


def _validate_coordinate_frame(parameters: dict[str, Any]) -> None:
    frame = parameters.get("coordinate_frame")
    if frame not in {"robot_base", "tool", "world"}:
        raise CapabilityError(
            "coordinate_frame must be 'robot_base', 'tool', or 'world'."
        )


def _validate_motion_options(parameters: dict[str, Any]) -> None:
    orientation = parameters.get("orientation_xyzw")
    if orientation is not None:
        normalized = unit_quaternion_xyzw(orientation, tolerance=1e-3)
        if normalized is None:
            raise CapabilityError(
                "orientation_xyzw must be a finite unit quaternion [x,y,z,w]."
            )
        parameters["orientation_xyzw"] = list(normalized)
        rotation_path = str(parameters.setdefault("rotation_path", "shortest"))
        if rotation_path != "shortest":
            raise CapabilityError("rotation_path must be 'shortest'.")
        angular_speed = parameters.setdefault("max_angular_speed_rad_s", 0.5)
        if (
            not isinstance(angular_speed, (int, float))
            or not math.isfinite(float(angular_speed))
            or not 0 < float(angular_speed) <= 1.0
        ):
            raise CapabilityError(
                "max_angular_speed_rad_s must be finite, > 0, and <= 1.0."
            )
    speed = parameters.setdefault("speed_m_s", 0.1)
    if speed is not None and (
        not isinstance(speed, (int, float))
        or not math.isfinite(float(speed))
        or not 0 < float(speed) <= 1.0
    ):
        raise CapabilityError("speed_m_s must be finite, > 0, and <= 1.0.")
