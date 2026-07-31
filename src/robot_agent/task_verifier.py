from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .domain import WorldState
from .state import PlanStep


@dataclass(frozen=True)
class TaskVerification:
    success: bool
    reason: str
    verification_scope: str


class TaskVerifier(Protocol):
    def verify(
        self,
        original_task: str,
        completed_steps: list[PlanStep],
        world_state: WorldState,
    ) -> TaskVerification: ...


class PlanCompletionTaskVerifier:
    """Software-level verifier used until physical goal verification is attached."""

    def verify(
        self,
        original_task: str,
        completed_steps: list[PlanStep],
        world_state: WorldState,
    ) -> TaskVerification:
        if not completed_steps:
            return TaskVerification(False, "No task step completed.", "plan")
        return TaskVerification(
            True,
            "All runnable plan steps completed; physical goal remains unverified.",
            "plan",
        )


class SymbolicGoalTaskVerifier:
    """Verify a trusted, grounded symbolic goal after all skills finish."""

    def verify(
        self,
        original_task: str,
        completed_steps: list[PlanStep],
        world_state: WorldState,
    ) -> TaskVerification:
        goal = world_state.get("_agent.task_goal")
        conditions = goal.get("conditions") if isinstance(goal, dict) else None
        if isinstance(conditions, list) and conditions:
            if world_state.conditions_met(conditions):
                return TaskVerification(
                    True,
                    (
                        "Trusted symbolic goal conditions are satisfied; "
                        "physical verification remains unavailable."
                    ),
                    "symbolic_goal",
                )
            return TaskVerification(
                False,
                "The completed skills did not satisfy the declared symbolic goal.",
                "symbolic_goal",
            )
        return PlanCompletionTaskVerifier().verify(
            original_task,
            completed_steps,
            world_state,
        )
