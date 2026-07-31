from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyMetadata:
    policy_id: str
    version: str = "unknown"
    action_type: str = "manipulate"
    required_inputs: tuple[str, ...] = ("robot_state",)
    robot_types: tuple[str, ...] = ()
    supports_stop: bool = True
    timeout_seconds: float = 60.0
    description: str = ""

    def validate_inputs(
        self, observation: dict[str, Any], robot_state: dict[str, Any]
    ) -> list[str]:
        available = {
            "observation": bool(observation),
            "perception": bool(observation.get("available", False)),
            "robot_state": bool(robot_state),
        }
        return [name for name in self.required_inputs if not available.get(name, False)]
