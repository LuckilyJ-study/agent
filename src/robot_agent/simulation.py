from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .action_executors import (
    DryRunRobotController,
    PolicyRegistry,
)
from .policy_metadata import PolicyMetadata
from .state import PlanStep


class PickAndPlaceSimulationPlanner:
    """Deterministic planner for an end-to-end no-hardware demonstration."""

    def __init__(self, object_name: str = "demo_object", destination: str = "demo_tray") -> None:
        self.object_name = object_name
        self.destination = destination
        self.replan_calls = 0
        self.last_failure_context: dict[str, Any] | None = None

    def create_plan(self, user_task: str) -> list[PlanStep]:
        return self.create_plan_with_context({"original_task": user_task})

    def create_plan_with_context(self, context: dict[str, Any]) -> list[PlanStep]:
        task = str(context["original_task"])
        self.object_name, self.destination = _parse_pick_and_place_task(task)
        return [
            {
                "step_id": 1,
                "action_type": "move_to",
                "target": self.object_name,
                "executor": "robot",
                "expected_result": f"arm is near {self.object_name}",
                "effects": [
                    {"path": "robot.at", "operation": "set", "value": self.object_name}
                ],
            },
            {
                "step_id": 2,
                "action_type": "pick",
                "target": self.object_name,
                "executor": "policy",
                "policy_id": "simulated_pick",
                "expected_result": f"{self.object_name} is held",
                "depends_on": [1],
                "conditions": [
                    {"path": "robot.at", "operator": "eq", "value": self.object_name}
                ],
                "effects": [
                    {
                        "path": "gripper.holding",
                        "operation": "set",
                        "value": self.object_name,
                    }
                ],
            },
            {
                "step_id": 3,
                "action_type": "move_to",
                "target": self.destination,
                "executor": "robot",
                "expected_result": f"arm is near {self.destination}",
                "depends_on": [2],
                "conditions": [
                    {
                        "path": "gripper.holding",
                        "operator": "eq",
                        "value": self.object_name,
                    }
                ],
                "effects": [
                    {"path": "robot.at", "operation": "set", "value": self.destination}
                ],
            },
            {
                "step_id": 4,
                "action_type": "place",
                "target": self.destination,
                "executor": "policy",
                "policy_id": "simulated_place",
                "expected_result": f"{self.object_name} is on {self.destination}",
                "depends_on": [3],
                "conditions": [
                    {
                        "path": "gripper.holding",
                        "operator": "eq",
                        "value": self.object_name,
                    }
                ],
                "effects": [
                    {"path": "gripper.holding", "operation": "set", "value": None},
                    {
                        "path": f"objects.{self.object_name}.location",
                        "operation": "set",
                        "value": self.destination,
                    },
                ],
            },
        ]

    def revise_from_failure(self, context: dict[str, Any]) -> list[PlanStep]:
        """Return only the suffix; the already-completed approach step is omitted."""
        self.replan_calls += 1
        self.last_failure_context = context
        base = 100 * self.replan_calls
        return [
            {
                "step_id": base + 1,
                "action_type": "pick",
                "target": self.object_name,
                "executor": "policy",
                "policy_id": "simulated_pick",
                "expected_result": f"{self.object_name} is held",
                "conditions": [
                    {"path": "robot.at", "operator": "eq", "value": self.object_name}
                ],
                "effects": [
                    {
                        "path": "gripper.holding",
                        "operation": "set",
                        "value": self.object_name,
                    }
                ],
            },
            {
                "step_id": base + 2,
                "action_type": "move_to",
                "target": self.destination,
                "executor": "robot",
                "expected_result": f"arm is near {self.destination}",
                "depends_on": [base + 1],
                "conditions": [
                    {
                        "path": "gripper.holding",
                        "operator": "eq",
                        "value": self.object_name,
                    }
                ],
                "effects": [
                    {"path": "robot.at", "operation": "set", "value": self.destination}
                ],
            },
            {
                "step_id": base + 3,
                "action_type": "place",
                "target": self.destination,
                "executor": "policy",
                "policy_id": "simulated_place",
                "expected_result": f"{self.object_name} is on {self.destination}",
                "depends_on": [base + 2],
                "conditions": [
                    {
                        "path": "gripper.holding",
                        "operator": "eq",
                        "value": self.object_name,
                    }
                ],
                "effects": [
                    {"path": "gripper.holding", "operation": "set", "value": None},
                    {
                        "path": f"objects.{self.object_name}.location",
                        "operation": "set",
                        "value": self.destination,
                    },
                ],
            },
        ]


@dataclass
class FaultInjectingPolicyBackend:
    """Policy stand-in that can fail the first N calls deterministically."""

    policy_id: str
    failure_reason: str = "GRASP_FAILED"
    failures_before_success: int = 0
    calls: int = 0
    stop_calls: int = 0

    def execute(
        self,
        step: PlanStep,
        observation: dict[str, Any],
        robot_state: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls += 1
        if self.calls <= self.failures_before_success:
            return {
                "status": "failed",
                "reason": self.failure_reason,
                "command_completed": True,
                "physical_result_verified": False,
                "details": {"policy_id": self.policy_id, "simulated_attempt": self.calls},
            }
        return {
            "status": "success",
            "reason": "SIMULATED_POLICY_COMPLETED",
            "command_completed": True,
            "physical_result_verified": False,
            "details": {"policy_id": self.policy_id, "simulated_attempt": self.calls},
        }

    def stop(self) -> None:
        self.stop_calls += 1


@dataclass
class SimulationComponents:
    planner: PickAndPlaceSimulationPlanner
    controller: DryRunRobotController
    policies: PolicyRegistry
    pick_backend: FaultInjectingPolicyBackend
    place_backend: FaultInjectingPolicyBackend


def build_pick_and_place_simulation(
    *,
    pick_failures: int = 1,
    object_name: str = "demo_object",
    destination: str = "demo_tray",
) -> SimulationComponents:
    if pick_failures < 0:
        raise ValueError("pick_failures cannot be negative.")
    planner = PickAndPlaceSimulationPlanner(object_name, destination)
    controller = DryRunRobotController()
    pick_backend = FaultInjectingPolicyBackend(
        "simulated_pick", failures_before_success=pick_failures
    )
    place_backend = FaultInjectingPolicyBackend("simulated_place")
    policies = PolicyRegistry()
    policies.register(
        "simulated_pick",
        pick_backend,
        PolicyMetadata(
            policy_id="simulated_pick",
            version="simulation-1",
            action_type="pick",
            required_inputs=("robot_state",),
            description="Fault-injecting simulated picking policy.",
        ),
    )
    policies.register(
        "simulated_place",
        place_backend,
        PolicyMetadata(
            policy_id="simulated_place",
            version="simulation-1",
            action_type="place",
            required_inputs=("robot_state",),
            description="Simulated placement policy.",
        ),
    )
    return SimulationComponents(
        planner, controller, policies, pick_backend, place_backend
    )


def _parse_pick_and_place_task(task: str) -> tuple[str, str]:
    """Parse the deliberately small offline simulation vocabulary."""
    chinese = re.search(
        r"把\s*(.+?)\s*(?:放到|放在|放入|移动到)\s*(.+?)\s*[。！？]?$",
        task.strip(),
    )
    if chinese:
        return _normalize_name(chinese.group(1)), _normalize_name(chinese.group(2))

    english = re.search(
        r"\bput\s+(?:the\s+)?(.+?)\s+(?:on|onto|in|into)\s+(?:the\s+)?(.+?)\s*[.!?]?$",
        task.strip(),
        flags=re.IGNORECASE,
    )
    if english:
        return _normalize_name(english.group(1)), _normalize_name(english.group(2))

    raise ValueError(
        "The offline simulation supports pick-and-place tasks only. "
        "Use '把 <物体> 放到 <目标>' or 'Put <object> on <destination>'."
    )


def _normalize_name(value: str) -> str:
    normalized = re.sub(r"\s+", "_", value.strip())
    if not normalized:
        raise ValueError("Object and destination names cannot be empty.")
    return normalized
