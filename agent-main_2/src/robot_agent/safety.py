from __future__ import annotations

from dataclasses import dataclass

from .state import RobotState


@dataclass(frozen=True)
class ExecutionPolicy:
    """Safety policy that must be explicitly relaxed before real-hardware use."""

    simulation_mode: bool = True
    operator_confirmed: bool = False
    max_retries: int = 2

    def can_execute(self, state: RobotState) -> tuple[bool, str]:
        if self.simulation_mode:
            return True, "simulation_mode"
        if not self.operator_confirmed:
            return False, "real hardware requires explicit operator confirmation"
        if state.get("status") != "running":
            return False, f"cannot execute while status is {state.get('status')}"
        return True, "approved"
